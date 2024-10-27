import numpy as np
import pandas as pd
import yahooquery as yq
from scipy.optimize import minimize_scalar
import polars as pl
from pydantic import BaseModel
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from session_model_dcf import SessionModelDCF, MARKET_CURRENCY, ERROR_SYMBOL
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

ERROR_QUERRY_PRICE = 2
OVERLAPING_DAYS_TOL = 5

EURONEXT_ID = '710'
NASDAQ_ID = '663'

INC_CODE = [
    'RTLR', # 'TotalRevenue'
    "NINC", # "NetIncome", 

]

BAL_CODE = [

    "STLD", # 'TotalDebt',
    "ACAE", # "Cash & Equivalents" 
    "QTLE", # "Total Equity"
    "QTCO", # "Total Common Shares Outstanding"
]

CASH_CODE = [
    "OTLO", # "Cash from Operating Activities",
    "SCEX", # Capital Expenditures,
    "FCDP", # Total Cash Dividends Paid
    "FPSS", # Issuance (Retirement) of Stock, Net,

            ]


FINANCIAL_ST_CODE =   INC_CODE + BAL_CODE + CASH_CODE

class ShareFinancialStatements():
    """
    Share financial statement from degiro
    
    """

    def __init__(self, session_model : SessionModelDCF):

        self.y_financial_statements : pd.DataFrame = None
        self.q_financial_statements : pd.DataFrame = None
        self.last_bal_financial_statements : pd.DataFrame = None
        self.financial_currency : str = None
        self.session_model = session_model

    def retrieve(self):

        financial_st = self.session_model.trading_api.get_financial_statements(
                        product_isin= self.isin,
                        raw= False
                    )
        
        self.financial_currency = financial_st.currency
        
        ### Retrive annual data
        data = []
        for y in financial_st.annual :
            y_dict = {
                # "fiscalYear" : y['fiscalYear'],
                      "endDate"  :  y['endDate']}
            for statement in y['statements']:
                for  item in  statement['items'] :
                    if item['code'] in FINANCIAL_ST_CODE:
                        # y_dict[item["meaning"]] = item["value"]
                        y_dict[item["code"]] = item["value"]
            data.append(y_dict)
        
        self.y_financial_statements = pd.DataFrame.from_records(
            data).iloc[::-1].set_index('endDate')[FINANCIAL_ST_CODE]
        
        ### Retrive interim data
        int_data = []
        for q in financial_st.interim :
            for statement in q['statements']:
                q_dict = {
                    # "fiscalYear" : q['fiscalYear'],
                        "endDate"  :  q['endDate']
                        }
                for k, v in statement.items():
                    if k != "items" :
                        q_dict[k] = v
                 
                for  item in  statement['items'] :
                    if item['code'] in FINANCIAL_ST_CODE:
                        # q_dict[item["meaning"]] = item["value"]
                        q_dict[item["code"]] = item["value"]             

                int_data.append(q_dict)
         
        df = pd.DataFrame.from_records(int_data).iloc[::-1]
        df['endDate'] = pd.to_datetime(df['endDate'])
        gb = df.groupby('type')

        q_inc_financial_data = gb.get_group('INC').dropna(axis=1)
        q_bal_financial_data = gb.get_group('BAL').dropna(axis=1)
        q_cas_financial_data = gb.get_group('CAS').dropna(axis=1)
        
        def get_date_shift_back(end_date : datetime, months : int):

            return end_date - relativedelta(months= months )

        ### correct interim data corresponding to period lenght if period is overlapping previous
        for p_df in [q_inc_financial_data, q_cas_financial_data] :
            value_cols = ["periodLength"] + [c for c in p_df.columns if c in FINANCIAL_ST_CODE]

            #### eval coresponding startDate from end date and periodLenght
            p_df['startDate'] = p_df.apply(lambda x : get_date_shift_back(x.endDate, x.periodLength), axis = 1)
            p_df['startDate_shift'] = p_df['startDate'].shift()
            
            #### apply overlaping tolerance of 5 days
            p_df['startDate_shift'] = p_df['startDate_shift'].apply(lambda x : x + relativedelta(days= OVERLAPING_DAYS_TOL ) if not pd.isnull(x) else x,)

            ### mark as overlaping line for which the period cover the one of previous line
            p_df["overlaping"] = p_df['startDate'] < p_df['startDate_shift']

            ### correct overlaping line by substracting from it periodLenght and values from previous line
            for i in range(len(p_df.index)) :
                if p_df.iloc[i]["overlaping"] :
                    p_df.iloc[i, p_df.columns.get_indexer(value_cols)] -= p_df[value_cols].iloc[i-1]

            p_df.drop(['startDate', 'startDate_shift', 'overlaping'], inplace = True, axis = 1)

        q_financial_statements = pd.concat([df.set_index('endDate') for df in [
                q_inc_financial_data,
                q_bal_financial_data,
                q_cas_financial_data
        ]], axis = 1)[FINANCIAL_ST_CODE]

        self.q_financial_statements = q_financial_statements     
        self.last_bal_financial_statements = q_financial_statements.iloc[-1][BAL_CODE]


class ShareRatios():

    def __init__(self):
        pass


class ShareFinancialComputer():

    """
    Cumputed infos
    """

    def __init__(self, 
                 session_model : SessionModelDCF  = None,
                 ratios : ShareRatios = None,
                 financial_statements : ShareFinancialStatements = None,
                 ) :

        self.session_model = session_model
        self.financial_statements = financial_statements
        self.fcf : float = None
        self.fcf_last : float = None
        self.cmpc :float = None
        self.mean_g_fcf : float = None
        self.net_debt : float = None

        self.price_to_fcf :float = None
        self.per : float = np.nan
        self.debt_to_equity : float = np.nan
        self.price_to_book : float = np.nan
        self.mean_g_tr = None
        self.mean_g_netinc = None
        self.capital_cost : float = None
        self.roic :float = np.nan
        
        self.beta : float = None
        self.unknown_last_fcf = None
        self.price_currency : str = None
        self.q_financial_data = None
        self.nb_shares : float = None
        self.market_cap : float = None
        self.net_market_cap : float = None
        self.g :float = np.nan
        self.g_last :float = np.nan
        self.financial_currency_price_history = None
        self.df_multiple = None
        
        self.total_payout_ratio = None


    def compute(self):
        
        market_infos = self.session_model.market_infos

        free_risk_rate = market_infos.free_risk_rate
        market_rate = market_infos.market_rate
        y_financial_statements = self.financial_statements.y_financial_statements
        y_financial_statements =  self.financial_statements.q_financial_statements
        
        stock_equity = self.last_bal_financial_statements['QTLE'] # CommonStockEquity
        market_cap = self.market_cap

        if self.capital_cost_equal_market :
            capital_cost = market_rate
        else : 
            capital_cost = free_risk_rate + self.beta * (market_rate - free_risk_rate)
        
        total_debt = self.last_bal_financial_statements['STLD'] # 'TotalDebt',

        self.cmpc = capital_cost * stock_equity/(total_debt + stock_equity) + \
                    self.market_infos.debt_cost * (1-IS) * total_debt/(total_debt + stock_equity)
        self.net_debt = total_debt - last_financial_info['CashAndCashEquivalents']


        self.net_market_cap = market_cap + self.net_debt
        if stock_equity >= 0 :
            self.debt_to_equity = total_debt / stock_equity
            self.price_to_book = market_cap / stock_equity



        # net_income = y_financial_data['NetIncome'][-3:].mean()
        net_income = y_financial_data['NetIncome'][-1]
        last_delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[-2],)

        payout = 0
        for info in PAYOUT_INFOS :
            if info in y_financial_data:
                payout -= y_financial_data[info][-3:].mean()
        
        self.total_payout_ratio = payout / y_financial_data['NetIncome'][-3:].mean()
        self.per = market_cap / net_income

        invested_capital = last_financial_info['InvestedCapital']
        if invested_capital >= 0 :
            self.roic = net_income / invested_capital

        if last_delta_t.years < 1 and \
            isinstance(q_financial_data, pd.DataFrame) and \
            ('FreeCashFlow' in q_financial_data.columns) and \
            (self.q_financial_statements.index[-1] > y_financial_data.index[-2]) :

            complement_q_financial_infos = self.q_financial_statements[self.q_financial_statements.index
                                                                > y_financial_data.index[-2]]
            complement_time = relativedelta(complement_q_financial_infos.index[-1],
                                            y_financial_data.index[-2]).months / 12
            self.fcf = (y_financial_data['FreeCashFlow'][-4:-1].sum() \
                        + complement_q_financial_infos['FreeCashFlow'].sum()
                        ) / (3 + complement_time)

        elif self.unknown_last_fcf :
            print(f"unknown last free cash flow  for {self.name}")
            self.fcf = y_financial_data['FreeCashFlow'][-4:-1].mean()

        else :
            self.fcf = y_financial_data['FreeCashFlow'][-3:].mean()

        if self.fcf < 0 :
            self.fcf = y_financial_data['FreeCashFlow'].mean()

        self.fcf_last = y_financial_data['FreeCashFlow'][-1]

        ### price to fcf multiple
        df_multiple = y_financial_data[['BasicAverageShares', 'FreeCashFlow' ]].copy()
        df_multiple.index = df_multiple.index.date
        df_multiple = pd.concat([self.financial_currency_price_history, 
                                df_multiple], axis = 1).sort_index().ffill().dropna()
        df_multiple['price_to_fcf'] = df_multiple['BasicAverageShares'] * df_multiple['close'] / df_multiple['FreeCashFlow']
        self.df_multiple = df_multiple
        self.price_to_fcf = df_multiple.loc[df_multiple['FreeCashFlow'] > 0 , 'price_to_fcf'].median()

        # print(self.df_multiple)
        # stop
        # if self.price_to_fcf < 0 :
        #     self.price_to_fcf = df_multiple['price_to_fcf'].max()
        
        # if self.price_to_fcf < 0 :
        #     print(f"negative price_to_fcf mean for {self.short_name} can not compute DCF")
        #     return(1)

        ### calculation of fcf growth
        delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[YEAR_G],)
        nb_years_fcf = delta_t.years + delta_t.months / 12

        fcf_se_ratio = y_financial_data['FreeCashFlow'].iloc[-1]\
            /y_financial_data['FreeCashFlow'].iloc[YEAR_G]
        
        if fcf_se_ratio < 0:
            self.mean_g_fcf = np.nan
        else :
            self.mean_g_fcf = (fcf_se_ratio)**(1/nb_years_fcf) - 1


        nb_year_inc = delta_t.years + delta_t.months / 12
        self.mean_g_tr = (y_financial_data['TotalRevenue'].iloc[-1]\
                        /y_financial_data['TotalRevenue'].iloc[YEAR_G])**(1/nb_year_inc) - 1
        # inc_se_ratio = y_financial_data['NetIncome'].iloc[-1]\
        #                 /y_financial_data['NetIncome'].iloc[YEAR_G]
        # if inc_se_ratio < 0 :
        #     self.mean_g_netinc = np.nan
        # else :
        #     self.mean_g_netinc = inc_se_ratio**(1/nb_year_inc) - 1

        if pr :
            print("\r")
            print(y_financial_data)
            print(f"Prix courant: {self.close_price:.2f} {self.financial_currency:s}" )
            print(f"Cout moyen pondere du capital: {self.cmpc*100:.2f}%")
            print(f"Croissance moyenne du chiffre d'affaire sur {nb_year_inc:f} \
                ans: {self.mean_g_tr*100:.2f}%")
            print(f"Croissance moyenne du free cash flow sur {nb_year_inc:f} \
                ans: {self.mean_g_fcf*100:.2f}%")
        return(0)


class Share():
    """
    Object containing a share and its financial informations
    """
    def __init__(self, s_id : int = None, s_dict : dict = None, session_model : SessionModelDCF  = None):
        self.id : str = None
        if s_id :
            self.id  = str(s_id)
        self.history : pl.DataFrame = None
        self.history_expires : datetime = None
        self.symbol : str = None
        
        self.name : str = None
        self.isin :str = None
        self.close_price : float = None
        self.product_type :str = None
        self.vwd_id : str = None
        self.vwd_id_secondary : str = None
        self.exchange_id : str = None
        self.market_cap : float = None
        self.nb_shares : int = None
        self.current_price : float = None

        self.capital_cost_equal_market : bool = False
        self.__dict__.update(s_dict)

        self.session_model : SessionModelDCF = session_model
        self.market_infos = session_model.market_infos
        self.computed_infos : ShareFinancialComputer = None
        self.ratios : ShareRatios = None
        self.financial_statements : ShareFinancialStatements = None


        # import json
        # with open("company_profile.json", "w", encoding= "utf8") as outfile: 
        #     json.dump(company_profile['data'], outfile, indent = 4)

    def retrieve_history(self) -> None :

        """
        Retrieve share history price dataframe
        """
        try :
            float(self.vwd_id)
            serie_id = self.vwd_id
        except ValueError :
            serie_id = self.vwd_id_secondary

        chart_request = ChartRequest(
        culture="fr-FR",
        # culture = "en-US",
     
        period=Interval.P5Y,
        # period=Interval.P1D,
        requestid="1",
        resolution=Interval.P1M,
        # resolution=Interval.PT60M,
        
        series=[
            # "issueid:360148977"+ self.vwd_id ,
            # "ohlc:issueid:" + self.vwd_id,
            "price:issueid:"+ serie_id,
            # "dividend:issueid:"+ self.vwd_id,
            # "volume:issueid:360148977",
        ],
        tz="Europe/Paris",
        )

        chart = self.session_model.chart_fetcher.get_chart(
            chart_request=chart_request,
            raw=False,
        )
        if isinstance(chart, BaseModel):
            df = pl.DataFrame(data=chart.series[0].data, orient="row")
            df.columns = ['month', 'close']

            self.current_price = df['close'][-1]
            self.history = df['month', 'close']
            self.history_expires = chart.series[0].expires
            # print(self.name)
            # print(self.history)
            # print(self.current_price)
            # print(self.close_price)

              
    def retrieve_company_profile(self):
        """
        Retrive company gerneral infos
        """
        
        company_profile = self.session_model.trading_api.get_company_profile(
        product_isin= self.isin,
        raw=True,
        )
        
        self.nb_shares = float(company_profile['data']['shrOutstanding'])
        self.market_cap = self.nb_shares * self.close_price

    def retrieve_financial_statements(self) -> None:
        """
        Retrieve financial data from degiro api
        """

        self.financial_statements = ShareFinancialStatements(self.session_model )

        self.financial_statements.retrieve()


        

    # def eval_beta(self) :
    #     """
    #     Compute the share beta value regarding the evolution of share price with reference market
    #     """
    #     regular_history = self.history['adjclose'].iloc[:-1].copy()

    #     if self.price_currency != MARKET_CURRENCY :
    #         self.market_infos.update_rate_dic(self.price_currency, MARKET_CURRENCY)
    #         change_rate = self.price_currency + MARKET_CURRENCY + "=X"
    #         regular_history = regular_history * self.market_infos.rate_history_dic[change_rate]

    #     month_change_share = regular_history.pct_change().rename("share")

    #     cov_df = pd.concat([self.market_infos.month_change_rate,
    #                         month_change_share], axis = 1, join= 'inner',).dropna(how = 'any')
    #     cov = cov_df.cov()['rm'].loc['share']
    #     beta = cov/ self.market_infos.var_rm
    #     self.beta = beta

    #     return beta
    

    def querry_financial_info(self,):
        """
        Get the share associated financial infos from degiro api 
        """
        
        print(f'\rquerry {self.name}    ', flush=True, end="")
  
        self.retrieve_history()
        self.retrieve_company_profile()
        self.retrieve_financial_statements()


        # self.eval_beta()

        # self.financial_currency_price_history = self.history[['close']].iloc[:-1].copy()
        # if self.price_currency != self.financial_currency:
        #     self.market_infos.update_rate_dic(self.price_currency, self.financial_currency)
        #     rate_symb = self.price_currency + self.financial_currency + "=X"
   
        #     rate = self.market_infos.rate_current_dic[rate_symb]
        #     self.close_price *= rate
        #     self.market_cap *= rate
        #     self.financial_currency_price_history = self.financial_currency_price_history.mul(self.market_infos.rate_history_dic[rate_symb],axis = 0)

        return 0

    def compute_financial_info(self, pr = True,) :
        """
        Compute intermedate financial infos used as input for dcf calculation
        """

        self.computed_infos = ShareFinancialComputer(self.session_model,
                                                self.ratios,
                                                self.financial_statements)
        
        self.computed_infos.compute()

        

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        """
        Get price corresponding to given growth rate with discouted
        cash flow methods
        """
        if self.cmpc is None :
            self.querry_financial_info()
            self.compute_financial_info(pr = pr)
        if g is None :
            g = self.mean_g_tr
        if start_fcf is None :
            start_fcf = self.fcf

        eval_dcf_(g, self, start_fcf, pr)

    def eval_g(self, start_fcf : float = None, pr=False, use_multiple = True):

        """
        Evaluate company assumed growth rate from fundamental financial data
        """

        
        if self.cmpc < 0:
            print(f"negative cmpc for {self.short_name} can not compute DCF")
            return np.nan
        if not start_fcf is None :
            self.fcf = start_fcf
        if use_multiple:
            up_bound = 2
        else : 
            up_bound = self.cmpc

        # compute g from mean fcf
        if self.fcf < 0 :
            print(f"negative free cash flow mean for {self.short_name} can not compute DCF")
        else :
            res_mean = minimize_scalar(eval_dcf_, args=(self, self.fcf, False),
                                method= 'bounded', bounds = (-1, up_bound))
            self.g = res_mean.x

        # compute g_last from last fcf
        if self.fcf_last >= 0:
            res_last = minimize_scalar(eval_dcf_, args=(self, self.fcf_last, False),
                                method= 'bounded', bounds = (-1, up_bound))
            self.g_last = res_last.x

        if pr:
            print(f"Croissance correspondant au prix courrant: {self.g*100:.2f}%")
            self.get_dcf(self.g, start_fcf= start_fcf, pr = pr)


    def eval_dcf(self, g :  float, fcf : float, pr = False, use_multiple = True):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.cmpc
        net_debt = self.net_debt
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        if use_multiple :
            vt = fcf * (1+g)**(NB_YEAR_DCF ) * self.price_to_fcf
        else :
            vt = fcf * (1+g)**(NB_YEAR_DCF ) / (cmpc - g)
        vt_act = vt / (1+cmpc)**(NB_YEAR_DCF)

        a = (1+g)/(1+cmpc)
        # fcf * sum of a**k for k from 1 to NB_YEAR_DCF 
        fcf_act_sum = fcf * ((a**NB_YEAR_DCF - 1)/(a-1) - 1 + a**(NB_YEAR_DCF))
        enterprise_value = fcf_act_sum + vt_act
     

        if pr :
            fcf_ar = np.array([fcf * (1+g)**(k) for k in range(1,1 + NB_YEAR_DCF)])
            act_vec = np.array([1/((1+cmpc)**k) for k in range(1,1 + NB_YEAR_DCF)])
            fcf_act = fcf_ar * act_vec
            print("\r")
            val_share = (enterprise_value - net_debt)/ self.nb_shares
            nyear_disp = min(10,NB_YEAR_DCF)
            annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
            table = np.array([ np.concatenate((fcf_ar[:nyear_disp] ,[vt])),
                              np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
            print(f"Prévision pour une croissance de {g*100:.2f}% :")
            print(tabulate(table, floatfmt= ".4e",
                           showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"],
                           headers= annees))

            print(f"Valeur DCF de l'action: {val_share:.2f} {self.price_currency:s}")

        # return ((enterpriseValue - netDebt)/ share.marketCap - 1)**2
        return (enterprise_value / self.net_market_cap - 1)**2

def eval_dcf_(g, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    share : Share = data[0]
    fcf : float = data[1]
    pr = data[2]

    return share.eval_dcf(g = g, fcf = fcf, pr = pr)