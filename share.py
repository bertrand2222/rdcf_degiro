import numpy as np
import pandas as pd
import json
import yahooquery as yq
from scipy.optimize import minimize_scalar
from pydantic import BaseModel
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from session_model_dcf import SessionModelDCF, MARKET_CURRENCY, ERROR_SYMBOL, IS, MarketInfos
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

ERROR_QUERRY_PRICE = 2
OVERLAPING_DAYS_TOL = 5

EURONEXT_ID = '710'
NASDAQ_ID = '663'

INC_CODES = [
    'RTLR', # 'TotalRevenue'
    "NINC", # "NetIncome", 
]

BAL_CODES = [
    "STLD", # 'TotalDebt',
    "ACAE", # "Cash & Equivalents" 
    "QTLE", # "Total Equity"
    "QTCO", # "Total Common Shares Outstanding"
]

CASH_CODES = [
    "OTLO", # "Cash from Operating Activities",
    "SCEX", # Capital Expenditures,
    "FCDP", # Total Cash Dividends Paid
    "FPSS", # Issuance (Retirement) of Stock, Net,

            ]
FINANCIAL_ST_CODES =   INC_CODES + BAL_CODES + CASH_CODES

RATIO_CODES = [
    'BETA',
    "PEEXCLXOR",  # "P/E excluding extraordinary items - TTM",
    # "APRFCFPS", # "Price to Free Cash Flow per Share - most recent fiscal year",
    "TTMROIPCT" # "Return on investment - trailing 12 month",
    ] 

PAYOUT_INFOS = ["FCDP", "FPSS" ]

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

class ShareFinancialStatements():
    """
    Share financial statement from degiro
    
    """
    y_financial_statements : pd.DataFrame = None
    q_inc_financial_statements = pd.DataFrame = None
    q_bal_financial_statements = pd.DataFrame = None     
    q_cas_financial_statements = pd.DataFrame = None
    last_bal_financial_statements : pd.DataFrame = None
    financial_currency : str = None

    def __init__(self, session_model : SessionModelDCF, isin : str):
        
        self.session_model = session_model
        self.isin = isin

    def retrieve(self):
        """
        Retrieve financial statements from degiro api
        """
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
                    if item['code'] in FINANCIAL_ST_CODES:
                        # y_dict[item["meaning"]] = item["value"]
                        y_dict[item["code"]] = item["value"]
            data.append(y_dict)

        y_financial_statements = pd.DataFrame.from_records(
            data).iloc[::-1]
        y_financial_statements['endDate'] = pd.to_datetime(y_financial_statements['endDate'])
        y_financial_statements = y_financial_statements.set_index('endDate')[FINANCIAL_ST_CODES]
        
        # free cash flow            = Cash from Operating Activities - Capital Expenditures, 
        y_financial_statements['FCFL'] = y_financial_statements["OTLO"] - y_financial_statements["SCEX"]

        self.y_financial_statements = y_financial_statements

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
                    if item['code'] in FINANCIAL_ST_CODES:
                        # q_dict[item["meaning"]] = item["value"]
                        q_dict[item["code"]] = item["value"]             

                int_data.append(q_dict)
         
        df = pd.DataFrame.from_records(int_data).iloc[::-1]
        df['endDate'] = pd.to_datetime(df['endDate'])
        gb = df.groupby('type')

        q_inc_financial_statements = gb.get_group('INC').dropna(axis=1)
        q_bal_financial_statements = gb.get_group('BAL').dropna(axis=1)
        q_cas_financial_statements = gb.get_group('CAS').dropna(axis=1)
        
        def get_date_shift_back(end_date : datetime, months : int):

            return end_date - relativedelta(months= months )

        ### correct interim data corresponding to period lenght if period is overlapping previous
        for p_df in [q_inc_financial_statements, q_cas_financial_statements] :
            value_cols = ["periodLength"] + [c for c in p_df.columns if c in FINANCIAL_ST_CODES]

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

        # q_financial_statements = pd.concat([df.set_index('endDate') for df in [
        #         q_inc_financial_statements,
        #         q_bal_financial_statements,
        #         q_cas_financial_statements
        # ]], axis = 1)[FINANCIAL_ST_CODES]

        # free cash flow            = Cash from Operating Activities - Capital Expenditures, 
        q_cas_financial_statements['FCFL'] = q_cas_financial_statements["OTLO"] - q_cas_financial_statements["SCEX"]
        # q_cas_financial_statements['annualised_FCFL'] = q_cas_financial_statements['FCFL'] * q_cas_financial_statements['periodLength'] / 12

        self.q_inc_financial_statements = q_inc_financial_statements.set_index('endDate')
        self.q_bal_financial_statements = q_bal_financial_statements.set_index('endDate')
        self.q_cas_financial_statements = q_cas_financial_statements.set_index('endDate')

        self.last_bal_financial_statements = q_bal_financial_statements.iloc[-1]


class ShareValues():
    """
    Data retrieved from degiro api with get_company_ratio
    """
    current_price : float = None
    market_cap : float = None
    nb_shares : int = None
    history : pd.DataFrame = None
    history_expires : datetime = None
    roic: float = None
    beta : float = None
    per : float = None
    price_to_fcf : float = None
    cmpc : float = None
    net_debt : float = None
    net_market_cap : float = None
    debt_to_equity : float = None
    price_to_book :float = None
    total_payout_ratio : float = None
    fcf : float = None
    fcf_ttm : float = None

    def __init__(self, session_model : SessionModelDCF,
                 financial_statements : ShareFinancialStatements, 
        isin : str,
        vwd_id : str,
        vwd_id_secondary : str
        ):
        self.session_model = session_model
        self.financial_statements = financial_statements
        self.isin = isin

        try :
            float(vwd_id)
            self.serie_id = vwd_id
        except ValueError :
            self.serie_id = vwd_id_secondary
    
    def retrieve_ratios(self):
        """
        retrive company ratios fom degiro api
        """
        ratios = self.session_model.trading_api.get_company_ratios(
            product_isin=self.isin, 
            raw = True
        )

        self.nb_shares = float(ratios['data']['sharesOut'])
        
        ratio_dic = {}
        for rg in ratios['data']['currentRatios']['ratiosGroups'] :
            for item in rg['items']:
                if item['id'] in RATIO_CODES:
                    ratio_dic[item['id']] = item['value']

        self.beta           = ratio_dic['BETA']
        self.per            = ratio_dic["PEEXCLXOR"]  # "P/E excluding extraordinary items - TTM",
        # self.price_to_fcf   = ratio_dic["APRFCFPS"] # "Price to Free Cash Flow per Share - most recent fiscal year",
        self.roic = ratio_dic['TTMROIPCT']

        if not self.current_price:
            self.retrieve_current_price()

        self.market_cap = self.nb_shares * self.current_price

        # print(self.nb_shares)
        # with open("company_ratio.json", "w", encoding= "utf8") as outfile: 
        #     json.dump(ratios['data'], outfile, indent = 4)
        # print(ratios)

    def retrieve_current_price(self):

        print("retrieve last price")

        chart_request = ChartRequest(
        culture="fr-FR",
        # culture = "en-US",
     
        period=Interval.PT5M,
        requestid="1",
        resolution=Interval.PT5M,        
        series=[
            "price:issueid:"+ self.serie_id,
        ],
        tz="Europe/Paris",
        )

        chart = self.session_model.chart_fetcher.get_chart(
            chart_request=chart_request,
            raw=False,
        )
        if isinstance(chart, BaseModel):
            # df = pl.DataFrame(data=chart.series[0].data, orient="row")
            # df.columns = ['time', 'close']
            self.current_price = chart.series[0].data[-1][-1]
    
    def retrieve_history(self):

        print("retrieve history")

        chart_request = ChartRequest(
        culture="fr-FR",
        # culture = "en-US",
     
        period=Interval.P5Y,
        requestid="1",
        resolution=Interval.P1M,
        
        series=[
            # "issueid:360148977"+ self.vwd_id ,
            # "ohlc:issueid:" + self.vwd_id,
            "price:issueid:"+ self.serie_id,
            # "volume:issueid:360148977",
        ],
        tz="Europe/Paris",
        )

        chart = self.session_model.chart_fetcher.get_chart(
            chart_request=chart_request,
            raw=False,
        )
        if isinstance(chart, BaseModel):
            history = pd.DataFrame(data=chart.series[0].data, columns = ['time', 'close'])

            last_day_current_month = last_day_of_month(chart.series[0].expires)
            for i in range(len(history)) :
                history.iloc[-1-i,0] = last_day_current_month - relativedelta(months= i)
        
            self.history = history.set_index('time').tz_localize(None)
            

    def compute(self, pr = False):
        """
        compute value and ratios from financial statements and market infos 
        """
        if self.session_model.market_infos is None:
            self.session_model.market_infos = MarketInfos()

        market_infos = self.session_model.market_infos
        free_risk_rate = market_infos.free_risk_rate

        if self.financial_statements.y_financial_statements is None:
            self.financial_statements.retrieve()

        y_financial_statements = self.financial_statements.y_financial_statements
        q_cas_financial_statements =  self.financial_statements.q_cas_financial_statements
        q_bal_financial_statements =  self.financial_statements.q_bal_financial_statements
        last_bal_financial_statements =  self.financial_statements.last_bal_financial_statements
        history_nb_year_avg = self.session_model.history_nb_year_avg

        stock_equity = self.financial_statements.last_bal_financial_statements['QTLE'] # CommonStockEquity

        if self.session_model.capital_cost_equal_market :
            capital_cost = market_infos.market_rate
        else : 
            capital_cost = free_risk_rate + self.beta * (market_infos.market_rate - free_risk_rate)
        
        total_debt = last_bal_financial_statements['STLD'] # 'TotalDebt',

        self.cmpc = capital_cost * stock_equity/(total_debt + stock_equity) + \
                    market_infos.debt_cost * (1-IS) * total_debt/(total_debt + stock_equity)
                        
        #net debt     =  total_debt - cash and cash equivalent
        self.net_debt = total_debt - last_bal_financial_statements['ACAE']

        if self.market_cap is None:
            self.retrieve_ratios()

        self.net_market_cap = self.market_cap + self.net_debt
        if stock_equity >= 0 :
            self.debt_to_equity = total_debt / stock_equity
            self.price_to_book = self.market_cap / stock_equity


        # net_income = y_financial_statements['NINC'][-3:].mean()
        # net_income = y_financial_statements['NetIncome'][-1]
        # last_delta_t = relativedelta(y_financial_statements.index[-1], y_financial_statements.index[-2],)

        payout = 0
        for info in PAYOUT_INFOS :
            payout -= y_financial_statements[info].iloc[-history_nb_year_avg:].mean()
        
        self.total_payout_ratio = payout / y_financial_statements['NINC'].iloc[-history_nb_year_avg:].mean()
        # self.per = market_cap / net_income

        # invested_capital = last_financial_info['InvestedCapital']
        # if invested_capital >= 0 :
        #     self.roic = net_income / invested_capital

        complement_q_financial_infos = q_cas_financial_statements.loc[q_cas_financial_statements.index
                                                            > y_financial_statements.index[-1]]
        
        complement_time = complement_q_financial_infos['periodLength'].sum() /12
        self.fcf = (y_financial_statements['FCFL'].iloc[-history_nb_year_avg:].sum() \
                    + complement_q_financial_infos['FCFL'].sum()
                    ) / (history_nb_year_avg + complement_time)

        # elif self.unknown_last_fcf :
        #     print(f"unknown last free cash flow  for {self.name}")
        #     self.fcf = y_financial_statements['FreeCashFlow'][-4:-1].mean()

        # else :
        #     self.fcf = y_financial_statements['FreeCashFlow'][-3:].mean()

        # if self.fcf < 0 :
        #     self.fcf = y_financial_statements['FreeCashFlow'].mean()

        ttm_fcf_start_time = q_cas_financial_statements.index[-1] - relativedelta(years= 1)
        ttm_fcf_infos = q_cas_financial_statements.loc[q_cas_financial_statements.index > ttm_fcf_start_time]
        self.fcf_ttm = ttm_fcf_infos['FCFL'].sum() / ttm_fcf_infos['periodLength'].sum() * 12

        ### price to fcf median calculation over all history
        if self.history is None:
            self.retrieve_history()

        df_multiple = pd.concat([self.history, 
                                y_financial_statements[["QTCO" , 'FCFL']]], axis = 0).sort_index().ffill().dropna()
        
        df_multiple['price_to_fcf'] = df_multiple['QTCO'] * df_multiple['close'] / df_multiple['FCFL']
        # self.df_multiple = df_multiple
        self.price_to_fcf = df_multiple.loc[df_multiple['FCFL'] > 0 , 'price_to_fcf'].median()

        print(self.price_to_fcf)
        exit(0)
        # print(self.df_multiple)
        # stop
        # if self.price_to_fcf < 0 :
        #     self.price_to_fcf = df_multiple['price_to_fcf'].max()
        
        # if self.price_to_fcf < 0 :
        #     print(f"negative price_to_fcf mean for {self.short_name} can not compute DCF")
        #     return(1)

        ### calculation of fcf growth
        delta_t = relativedelta(y_financial_statements.index[-1], y_financial_statements.index[YEAR_G],)
        nb_years_fcf = delta_t.years + delta_t.months / 12

        fcf_se_ratio = y_financial_statements['FCFL'].iloc[-1]\
            /y_financial_statements['FCFL'].iloc[YEAR_G]
        
        if fcf_se_ratio < 0:
            self.mean_g_fcf = np.nan
        else :
            self.mean_g_fcf = (fcf_se_ratio)**(1/nb_years_fcf) - 1


        nb_year_inc = delta_t.years + delta_t.months / 12
        self.mean_g_tr = (y_financial_statements['TotalRevenue'].iloc[-1]\
                        /y_financial_statements['TotalRevenue'].iloc[YEAR_G])**(1/nb_year_inc) - 1
        # inc_se_ratio = y_financial_statements['NetIncome'].iloc[-1]\
        #                 /y_financial_statements['NetIncome'].iloc[YEAR_G]
        # if inc_se_ratio < 0 :
        #     self.mean_g_netinc = np.nan
        # else :
        #     self.mean_g_netinc = inc_se_ratio**(1/nb_year_inc) - 1

        if pr :
            print("\r")
            print(y_financial_statements)
            print(f"Prix courant: {self.close_price:.2f} {self.financial_currency:s}" )
            print(f"Cout moyen pondere du capital: {self.cmpc*100:.2f}%")
            print(f"Croissance moyenne du chiffre d'affaire sur {nb_year_inc:f} \
                ans: {self.mean_g_tr*100:.2f}%")
            print(f"Croissance moyenne du free cash flow sur {nb_year_inc:f} \
                ans: {self.mean_g_fcf*100:.2f}%")
        return(0)

        

# class ShareFinancialComputer():

    # self.mean_g_fcf : float = None

    # self.price_to_fcf :float = None
    # self.debt_to_equity : float = np.nan
    # self.price_to_book : float = np.nan
    # self.mean_g_tr = None
    # self.mean_g_netinc = None
    # self.capital_cost : float = None

    # self.unknown_last_fcf = None
    # self.price_currency : str = None
    # self.net_market_cap : float = None
    # self.g :float = np.nan
    # self.g_last :float = np.nan
    # self.financial_currency_price_history = None
    # self.df_multiple = None



class Share():
    """
    Object containing a share and its financial informations
    """
    name : str = None
    isin :str = None
    vwd_id : str = None
    vwd_id_secondary : str = None
    symbol : str = None
    close_price : float = None
    product_type :str = None
    exchange_id : str = None

    financial_statements : ShareFinancialStatements = None

    def __init__(self, s_dict : dict = None,
            session_model : SessionModelDCF  = None):

        self.__dict__.update(s_dict)

        self.session_model = session_model

        self.financial_statements = ShareFinancialStatements(self.session_model,
                                                             self.isin )
        
        self.values = ShareValues(
            self.session_model,
            self.financial_statements,
            self.isin, 
            self.vwd_id, 
            self.vwd_id_secondary )
        
        self.retrieve_current_price = self.values.retrieve_current_price
        self.retrieve_history = self.values.retrieve_history
        self.retrieve_ratios = self.values.retrieve_ratios
        self.retrieve_financial_statements = self.financial_statements.retrieve
        self.compute_financial_info = self.values.compute
        
        # import json
        # with open("company_profile.json", "w", encoding= "utf8") as outfile: 
        #     json.dump(company_profile['data'], outfile, indent = 4)

  
    # def retrieve_company_profile(self):
    #     """
    #     Retrive company gerneral infos
    #     """
        
    #     company_profile = self.session_model.trading_api.get_company_profile(
    #     product_isin= self.isin,
    #     raw=True,
    #     )
        
    #     self.nb_shares = float(company_profile['data']['shrOutstanding'])
    #     self.market_cap = self.nb_shares * self.current_price


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
        self.retrieve_ratios()
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