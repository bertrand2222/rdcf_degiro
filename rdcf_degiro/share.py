import numpy as np
import pandas as pd
import json, os
from scipy.optimize import minimize_scalar
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from dateutil.relativedelta import relativedelta
from rdcf_degiro.session_model_dcf import SessionModelDCF, MarketInfos
from datetime import datetime, timedelta
from typing import Any, Callable
from tabulate import tabulate

ERROR_QUERRY_PRICE = 2
OVERLAPING_DAYS_TOL = 7

EURONEXT_ID = '710'
NASDAQ_ID = '663'

INC_CODES = [
    'RTLR', # 'TotalRevenue'
    'SIIB', # 'Total revenue (Bank)
    "SGRP", # Gross profit
    "NINC", # "NetIncome", 
]

BAL_CASH_CODES = [
    "ACAE", # "Cash & Equivalents" 
    "ACDB", # "Cash and due from bank"
    "ACSH", # "Cash"
]
BAL_CODES = BAL_CASH_CODES + [
    "STLD", # 'TotalDebt',
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

SPECIAL_CURRENCIES = {
    'GBX' : {'real' : 'GBP', 'rate_factor' : 0.01}
}

class ShareIdentity():

    name : str = None
    isin :str = None
    vwd_id : str = None
    vwd_id_secondary : str = None
    symbol : str = None
    currency : str = None
    vwd_identifier_type : str = None
    vwd_identifier_type_secondary : str = None

    def __init__(self, s_dict : dict):

        self.__dict__.update(s_dict)


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
    cash_code : str = 'ACAE'
    total_revenue_key : str = 'RTLR'

    def __init__(self, session_model : SessionModelDCF, identity : ShareIdentity):
        
        self.session_model = session_model
        self.identity = identity

        self.retrieve()

    def retrieve(self):
        """
        Retrieve financial statements from degiro api
        """
        try:
            financial_st = self.session_model.trading_api.get_financial_statements(
                        product_isin= self.identity.isin,
                        raw= True
                    )['data']
        except KeyError as e:
            raise KeyError(f'{self.identity.name} : no financial statement found for isin {self.identity.isin}') from e
                
        if self.session_model.output_value_files:
            with open(os.path.join(self.session_model.output_folder, 
                                   f"{self.identity.symbol}_company_financial.json"), 
                        "w", 
                        encoding= "utf8") as outfile: 
                json.dump(financial_st, outfile, indent = 4)

        self.financial_currency = financial_st['currency']

        ### Retrive annual data
        data = []
        for y in financial_st['annual'] :
            y_dict = {
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
        y_financial_statements = y_financial_statements.set_index('endDate')[list(
            set(FINANCIAL_ST_CODES) & set(y_financial_statements.columns))]
        
        if 'RTLR' not in y_financial_statements.columns :
            self.total_revenue_key = 'SIIB'
        
        # free cash flow            = Cash from Operating Activities + Capital Expenditures( negative), 
        y_financial_statements['FCFL'] = y_financial_statements["OTLO"] 
        if "SCEX" in y_financial_statements:
            y_financial_statements['FCFL'] += y_financial_statements["SCEX"]

        self.y_financial_statements = y_financial_statements

        ### Retrive interim data
        int_data = []
        for q in financial_st['interim'] :
            for statement in q['statements']:
                q_dict = {
                        "endDate"  :  q['endDate']
                        }
                for k, v in statement.items():
                    if k != "items" :
                        q_dict[k] = v
                 
                for  item in  statement['items'] :
                    if item['code'] in FINANCIAL_ST_CODES:
                        q_dict[item["code"]] = item["value"]             

                int_data.append(q_dict)
         
        df = pd.DataFrame.from_records(int_data).iloc[::-1]
        df['endDate'] = pd.to_datetime(df['endDate'])
        gb = df.groupby('type')

        q_inc_financial_statements = gb.get_group('INC').dropna(axis=1, how= 'all')
        q_bal_financial_statements = gb.get_group('BAL').dropna(axis=1, how= 'all').ffill()
        q_cas_financial_statements = gb.get_group('CAS').dropna(axis=1, how= 'all')
        
        for key in BAL_CASH_CODES:
            if key  in q_bal_financial_statements.columns:
                self.cash_code = key

        def get_date_shift_back(end_date : datetime, months : int = 0, weeks :int = 0):

            return end_date - relativedelta(months= months, weeks= weeks )

        ### correct interim data corresponding to period lenght if period is overlapping previous
        for p_df in [q_inc_financial_statements, q_cas_financial_statements] :
            value_cols = ["periodLength"] + [c for c in p_df.columns if c in FINANCIAL_ST_CODES]

            #### eval coresponding startDate from end date and periodLenght

            p_df['startDate'] = p_df.apply(lambda x : get_date_shift_back(x.endDate,
                                                                         months = (x.periodType == 'M') * x.periodLength,
                                                                         weeks = (x.periodType == 'W') * x.periodLength,
                                                                           ), axis = 1)
            p_df['startDate_shift'] = p_df['startDate'].shift()
            
            #### apply overlaping tolerance of OVERLAPING_DAYS_TOL days
            p_df['startDate_shift'] = p_df['startDate_shift'].apply(lambda x : x + relativedelta(days= OVERLAPING_DAYS_TOL ) if not pd.isnull(x) else x,)

            ### mark as overlaping line for which the period cover the one of previous line
            p_df["overlaping"] = p_df['startDate'] < p_df['startDate_shift']

            ### correct overlaping line by substracting from it periodLenght and values from previous line
            p_df.loc[p_df["overlaping"], value_cols] -= p_df[value_cols].shift()
            # p_old_df = p_df.copy()
            # for i in range(len(p_df.index)) :
            #     if p_df.iloc[i]["overlaping"] :
            #         p_df.iloc[i, p_df.columns.get_indexer(value_cols)] -= p_old_df[value_cols].iloc[i-1]

            p_df.drop(['startDate', 'startDate_shift', 'overlaping'], inplace = True, axis = 1)


        # free cash flow            = Cash from Operating Activities - Capital Expenditures, 
        q_cas_financial_statements['FCFL'] = q_cas_financial_statements["OTLO"] 
   
        if "SCEX" in q_cas_financial_statements:
            q_cas_financial_statements['FCFL'] += q_cas_financial_statements["SCEX"]
        # q_cas_financial_statements['annualised_FCFL'] = q_cas_financial_statements['FCFL'] * q_cas_financial_statements['periodLength'] / 12

        self.q_inc_financial_statements = q_inc_financial_statements.set_index('endDate')
        self.q_bal_financial_statements = q_bal_financial_statements.set_index('endDate')
        self.q_cas_financial_statements = q_cas_financial_statements.set_index('endDate')

        self.last_bal_financial_statements = q_bal_financial_statements.iloc[-1]

        return(0)
class ShareValues():
    """
    Data retrieved from degiro api with get_company_ratio
    """
    current_price : float = None
    market_cap : float = None
    nb_shares : int = None
    history : pd.DataFrame = None
    history_expires : datetime = None
    roic: float = np.nan
    beta : float = np.nan
    per : float = np.nan
    price_to_fcf : float = None
    price_to_fcf_terminal : float = None
    cmpc : float = None
    net_debt : float = None
    net_market_cap : float = None
    debt_to_equity : float = None
    price_to_book :float = None
    total_payout_ratio : float = None
    fcf : float = None
    fcf_ttm : float = None
    mean_g_fcf : float = None
    mean_g_tr : float = None
    capital_cost : float = None
    g : float = np.nan
    g_from_ttm : float = np.nan
    history_in_financial_currency : pd.DataFrame = None

    # mean_g_netinc = None

    # price_currency : str = None
    # financial_currency_price_history = None

    def __init__(self, session_model : SessionModelDCF,
                 financial_statements : ShareFinancialStatements, 
                 identity : ShareIdentity,
        ):
        self.session_model = session_model
        self.financial_statements = financial_statements
        self.identity = identity

        series_id_name = "issueid"
        serie_id = self.identity.vwd_id

        if self.identity.vwd_identifier_type_secondary =='issueid':
            serie_id = self.identity.vwd_id_secondary

        elif self.identity.vwd_identifier_type =='vwdkey':
            series_id_name = "vwdkey"

        try :
            self.price_series_str = f"price:{series_id_name}:{serie_id}"
        except TypeError as e:
            raise TypeError(
            f"not valid {series_id_name} type : {type(serie_id)}  history\
                             chart not available for the quote") from e
        self.retrieve_values()
    
        self.retrieve_history()
    
    
    def retrieve_values(self):
        """
        retrive company ratios fom degiro api
        """
        ratios = self.session_model.trading_api.get_company_ratios(
            product_isin=self.identity.isin, 
            raw = True
        )

        self.nb_shares = float(ratios['data']['sharesOut'])
        
        ratio_dic = {}
        for rg in ratios['data']['currentRatios']['ratiosGroups'] :
            for item in rg['items']:
                if item['id'] in RATIO_CODES:
                    if 'value' in item:
                        ratio_dic[item['id']] = item['value']
                    else:
                        print(f"{self.identity.name} : Warning {item['id']} value not found")


        if 'BETA' in ratio_dic:
            self.beta           = float(ratio_dic['BETA'])

        if "PEEXCLXOR" in ratio_dic : 
            self.per     = float(ratio_dic["PEEXCLXOR"])  # "P/E excluding extraordinary items - TTM",

        if "TTMROIPCT" in ratio_dic : 
            self.roic = float(ratio_dic['TTMROIPCT']) / 100 # return on investec capital - TTM

        # if not self.current_price:
        #     self.retrieve_current_price()

        

        if self.session_model.output_value_files:
            with open(os.path.join(self.session_model.output_folder, 
                                   f"{self.identity.symbol}_company_ratio.json"), 
                        "w", 
                        encoding= "utf8") as outfile: 
                json.dump(ratios['data'], outfile, indent = 4)

    def retrieve_intra_day_price(self):
        """
        retrive current price from charts
        """

        chart_request = ChartRequest(
            culture="fr-FR",
            # culture = "en-US",
        
            period=Interval.P1D,
            requestid="1",
            resolution=Interval.PT5M,        
            series=[
                self.price_series_str,
            ],
            tz="Europe/Paris",
            )
        chart = self.session_model.chart_fetcher.get_chart(
            chart_request=chart_request,
            raw=False,
        )
        try :
            self.current_price = chart.series[0].data[-1][-1]
        except IndexError :
            print(f'{self.identity.name} : warning, not enought data to retrieve intra day price')
            
    def retrieve_history(self):
        """
        retrive current history from charts
        """

        chart_request = ChartRequest(
            culture="fr-FR",
            # culture = "en-US",
        
            period=Interval.P5Y,
            requestid="1",
            resolution=Interval.P1M,

            series=[
                # "issueid:360148977"+ self.vwd_id ,
                self.price_series_str,
            ],
            tz="Europe/Paris",
            )

        chart = self.session_model.chart_fetcher.get_chart(
            chart_request=chart_request,
            raw=False,
        )

        history = pd.DataFrame(data=chart.series[0].data, columns = ['time', 'close'])
        self.current_price = history['close'].iloc[-1]

        if self.session_model.use_last_intraday_price:
                self.retrieve_intra_day_price()

        self.market_cap = self.nb_shares * self.current_price / 1e6

        last_day_current_month = last_day_of_month(chart.series[0].expires)
        for i in range(len(history)) :
            history.iloc[-1-i,0] = last_day_current_month - relativedelta(months= i)
    
        self.history = history.set_index('time').tz_localize(None)
    
        self.history_in_financial_currency = self.history.iloc[:-1].copy()

        ### convert price history in financial statement currency  if needed

        if self.identity.currency != self.financial_statements.financial_currency:
            
            rate_factor = 1
            share_currency = self.identity.currency
            if self.identity.currency in  SPECIAL_CURRENCIES :
                share_currency = SPECIAL_CURRENCIES[self.identity.currency]['real']
                rate_factor = SPECIAL_CURRENCIES[self.identity.currency]['rate_factor']
            if self.session_model.market_infos is None:
                self.session_model.market_infos = MarketInfos()
            
            if share_currency != self.financial_statements.financial_currency:
                self.session_model.market_infos.update_rate_dic(share_currency,
                                                                self.financial_statements.financial_currency,
                                                                )
                rate_symb = share_currency + self.financial_statements.financial_currency + "=X"
    
                rate = self.session_model.market_infos.rate_current_dic[rate_symb] * rate_factor

                self.market_cap *= rate

                history_in_financial_currency  = pd.concat(
                    [
                    self.history_in_financial_currency,
                        self.session_model.market_infos.rate_history_dic[rate_symb] * rate_factor
                        ], axis = 0
                        ).sort_index().ffill().dropna()
                        
                history_in_financial_currency['close'] *= history_in_financial_currency['change_rate']
                self.history_in_financial_currency = history_in_financial_currency
            else :
                self.market_cap *= rate_factor
                self.history_in_financial_currency  *= rate_factor
            

    def compute_complementary_values(self, pr = False):
        """
        compute value and ratios from financial statements and market infos 
        before dcf calculation
        """
        

        if self.session_model.market_infos is None:
            self.session_model.market_infos = MarketInfos()

        market_infos = self.session_model.market_infos
        free_risk_rate = market_infos.free_risk_rate

        if self.financial_statements.y_financial_statements is None:
            self.financial_statements.retrieve()

        y_financial_statements = self.financial_statements.y_financial_statements
        q_cas_financial_statements =  self.financial_statements.q_cas_financial_statements
        # q_bal_financial_statements =  self.financial_statements.q_bal_financial_statements
        last_bal_financial_statements =  self.financial_statements.last_bal_financial_statements
        history_avg_nb_year = self.session_model.history_avg_nb_year

        stock_equity = self.financial_statements.last_bal_financial_statements['QTLE'] # CommonStockEquity

        if self.session_model.capital_cost_equal_market :
            self.capital_cost = market_infos.market_rate
        else : 
            self.capital_cost = free_risk_rate + self.beta * (market_infos.market_rate - free_risk_rate)
        
        total_debt = last_bal_financial_statements['STLD'] # 'TotalDebt',

        self.cmpc = self.capital_cost * stock_equity/(total_debt + stock_equity) + \
                    market_infos.debt_cost * (1-self.session_model.taxe_rate) * total_debt/(total_debt + stock_equity)
                        
        #net debt     =  total_debt - cash and cash equivalent
        self.net_debt = total_debt - last_bal_financial_statements[self.financial_statements.cash_code]

        # if self.market_cap is None:
        #     self.retrieve_values()

        self.net_market_cap = self.market_cap + self.net_debt
        if stock_equity >= 0 :
            self.debt_to_equity = total_debt / stock_equity
            self.price_to_book = self.market_cap / stock_equity


        # net_income = y_financial_statements['NINC'][-3:].mean()
        # net_income = y_financial_statements['NetIncome'][-1]
        # last_delta_t = relativedelta(y_financial_statements.index[-1], y_financial_statements.index[-2],)

        payout = 0
        for info in list(set(PAYOUT_INFOS) & set(y_financial_statements.columns)) :
            payout -= y_financial_statements[info].iloc[-history_avg_nb_year:].mean()
        
        self.total_payout_ratio = payout / y_financial_statements['NINC'].iloc[-history_avg_nb_year:].mean()

        complement_q_financial_infos = q_cas_financial_statements.loc[q_cas_financial_statements.index
                                                            > y_financial_statements.index[-1]]
        
        complement_time =  ((complement_q_financial_infos['periodType'] == 'M') * complement_q_financial_infos['periodLength']).sum() /12 + \
                            ((complement_q_financial_infos['periodType'] == 'W') * complement_q_financial_infos['periodLength']).sum() /53

            
        nb_year_avg = history_avg_nb_year + complement_time
        self.fcf = (y_financial_statements['FCFL'].iloc[-history_avg_nb_year:].sum() \
                    + complement_q_financial_infos['FCFL'].sum()
                    ) / (nb_year_avg)

        # if self.fcf < 0 :
        #     self.fcf = y_financial_statements['FreeCashFlow'].mean()

        ttm_fcf_start_time = q_cas_financial_statements.index[-1] - relativedelta(years= 1)
        ttm_fcf_infos = q_cas_financial_statements.loc[q_cas_financial_statements.index > ttm_fcf_start_time]
        self.fcf_ttm = ttm_fcf_infos['FCFL'].sum() / (((ttm_fcf_infos['periodType'] == 'M') * ttm_fcf_infos['periodLength']).sum() /12 + \
                            ((ttm_fcf_infos['periodType'] == 'W') * ttm_fcf_infos['periodLength']).sum() /53)

        ### terminal price to fcf calculation from  history
        if self.history is None:
            self.retrieve_history()

        df_multiple = pd.concat([self.history_in_financial_currency, 
                                y_financial_statements[["QTCO" , 'FCFL']]], axis = 0).sort_index().ffill().dropna()

        df_multiple['price_to_fcf'] = df_multiple['QTCO'] * df_multiple['close'] / df_multiple['FCFL']

        if self.session_model.price_to_fcf_avg_method == 'harmonic':
            self.price_to_fcf = len(df_multiple) / (1 / df_multiple['price_to_fcf']).sum()
        elif self.session_model.price_to_fcf_avg_method == 'median':
            self.price_to_fcf = df_multiple['price_to_fcf'].median()
        else: # arithmetic
            self.price_to_fcf = df_multiple['price_to_fcf'].mean()
        
        self.price_to_fcf_terminal = min(max(self.price_to_fcf, self.session_model.terminal_price_to_fcf_bounds[0]),
                                          self.session_model.terminal_price_to_fcf_bounds[1])


        ### calculation of fcf growth
        fcf_se_ratio = self.fcf_ttm\
            /y_financial_statements['FCFL'].iloc[-history_avg_nb_year]
        
        if fcf_se_ratio < 0:
            self.mean_g_fcf = np.nan
        else :
            self.mean_g_fcf = (fcf_se_ratio)**(1/nb_year_avg) - 1


        self.mean_g_tr = (y_financial_statements[self.financial_statements.total_revenue_key].iloc[-1]\
                        /y_financial_statements[self.financial_statements.total_revenue_key].iloc[-history_avg_nb_year])**(1/nb_year_avg) - 1
       
        # inc_se_ratio = y_financial_statements['NetIncome'].iloc[-1]\
        #                 /y_financial_statements['NetIncome'].iloc[YEAR_G]
        # if inc_se_ratio < 0 :
        #     self.mean_g_netinc = np.nan
        # else :
        #     self.mean_g_netinc = inc_se_ratio**(1/nb_year_inc) - 1

        # if pr :
        #     print("\r")
        #     print(y_financial_statements)
        #     print(f"Prix courant: {self.close_price:.2f} {self.financial_currency:s}" )
        #     print(f"Cout moyen pondere du capital: {self.cmpc*100:.2f}%")
        #     print(f"Croissance moyenne du chiffre d'affaire sur {nb_year_inc:f} \
        #         ans: {self.mean_g_tr*100:.2f}%")
        #     print(f"Croissance moyenne du free cash flow sur {nb_year_inc:f} \
        #         ans: {self.mean_g_fcf*100:.2f}%")
        return(0)

    def eval_g(self, start_fcf : float = None, pr=False,):

        """
        Evaluate company assumed growth rate from fundamental financial data
        """
        
        
        if self.cmpc < 0:
            print(f"{self.identity.name} : negative cmpc can not compute DCF")
            return 
        if not start_fcf is None :
            self.fcf = start_fcf
        if self.session_model.use_multiple:
            up_bound = 2
        else : 
            up_bound = self.cmpc

        # compute g from mean fcf
        
        if self.fcf < 0 :
            print(f"{self.identity.name} : negative free cash flow mean can not compute RDCF")
        else :
            res_mean = minimize_scalar(eval_dcf_, args=(self, self.fcf, False),
                                method= 'bounded', bounds = (-1, up_bound))
            self.g = res_mean.x

        # compute g_from_ttm from last fcf

        if self.session_model.use_multiple :
            if self.price_to_fcf_terminal < 0:
                print(f"{self.identity.name} negative terminal price to fcf multiple, can not compute RDCF")
                return
            
        if self.fcf_ttm < 0:
            print(f"{self.identity.name} : negative TTM free cash flow mean can not compute TTM RDCF")
        else :
            res_last = minimize_scalar(eval_dcf_, args=(self, self.fcf_ttm, False),
                                method= 'bounded', bounds = (-1, up_bound))
            self.g_from_ttm = res_last.x

        if pr:
            print(f"Croissance correspondant au prix courrant: {self.g*100:.2f}%")
            self.get_dcf(self.g, start_fcf= start_fcf, pr = pr)

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        """
        Get price corresponding to given growth rate with discouted
        cash flow methods
        """
        if self.cmpc is None :
            self.compute_complementary_values()
        if g is None :
            g = self.mean_g_tr
        if start_fcf is None :
            start_fcf = self.fcf

        eval_dcf_(g, self, start_fcf, pr)

    
    def eval_dcf(self, g :  float, fcf : float, pr = False,):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.cmpc
        net_debt = self.net_debt
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        nb_year_dcf = self.session_model.nb_year_dcf
        if self.session_model.use_multiple :
            vt = fcf * (1+g)**(nb_year_dcf ) * self.price_to_fcf_terminal
        else :
            vt = fcf * (1+g)**(nb_year_dcf ) / (cmpc - g)
        vt_act = vt / (1+cmpc)**(nb_year_dcf)

        a = (1+g)/(1+cmpc)
        # fcf * sum of a**k for k from 1 to nb_year_dcf 
        fcf_act_sum = fcf * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf))
        enterprise_value = fcf_act_sum + vt_act
     

        if pr :
            fcf_ar = np.array([fcf * (1+g)**(k) for k in range(1,1 + nb_year_dcf)])
            act_vec = np.array([1/((1+cmpc)**k) for k in range(1,1 + nb_year_dcf)])
            fcf_act = fcf_ar * act_vec
            print("\r")
            val_share = (enterprise_value - net_debt)/ self.nb_shares
            nyear_disp = min(10,nb_year_dcf)
            annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
            table = np.array([ np.concatenate((fcf_ar[:nyear_disp] ,[vt])),
                              np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
            print(f"Prévision pour une croissance de {g*100:.2f}% :")
            print(tabulate(table, floatfmt= ".4e",
                           showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"],
                           headers= annees))

            print(f"Valeur DCF de l'action: {val_share:.2f} {self.identity.currency:s}")

        # return ((enterpriseValue - netDebt)/ share.marketCap - 1)**2
        return (enterprise_value / self.net_market_cap - 1)**2

class Share():
    """
    Object containing a share and its financial informations
    """

    # close_price : float = None
    product_type :str = None
    exchange_id : str = None
    currency :str = None
    financial_statements : ShareFinancialStatements = None
    values : ShareValues = None
    compute_complementary_values : Callable = None
    eval_g :  Callable = None

    def __init__(self, s_dict : dict = None,
            session_model : SessionModelDCF  = None):

        self.__dict__.update(s_dict)
        self.session_model = session_model
        self.identity = ShareIdentity(s_dict)
        

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
    

    def retrieves_all_values(self,):
        """
        Get all the share associated financial infos from degiro api 
        """

        self.financial_statements = ShareFinancialStatements(
            self.session_model,
            self.identity )
        
        self.values = ShareValues(
            self.session_model,
            self.financial_statements,
            self.identity,
        )
        # self.retrieve_current_price = self.values.retrieve_current_price
        # self.retrieve_history = self.values.retrieve_history
        # self.retrieve_values = self.values.retrieve_values
        # self.retrieve_financial_statements = self.financial_statements.retrieve

        self.compute_complementary_values = self.values.compute_complementary_values
        self.eval_g = self.values.eval_g

        # self.eval_beta()

        


    
    
    # def __getattr__(self, item) -> Callable[..., Any]:

    #     if item in self._action_list:
    #         action = item
    #         self.setup_one_action(action=action)

    #         return getattr(self, action)

    #     raise AttributeError(
    #         f"'{self.__class__.__name__}' object has no attribute '{item}'"
    #     )

def eval_dcf_(g, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    sharevalues : ShareValues = data[0]
    fcf : float = data[1]
    pr = data[2]

    return sharevalues.eval_dcf(g = g, fcf = fcf, pr = pr)