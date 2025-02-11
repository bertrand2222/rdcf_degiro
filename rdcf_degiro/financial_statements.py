import pandas as pd
import yahooquery as yq
from datetime import datetime, timedelta
import json, os
from dateutil.relativedelta import relativedelta
from rdcf_degiro.session_model_dcf import SessionModelDCF, MarketInfos
from rdcf_degiro.share_identity import ShareIdentity
from sklearn.linear_model import LinearRegression
# import matplotlib.pylab as plt
import numpy as np

OVERLAPING_DAYS_TOL = 7

YA_INCOME_INFOS = [ 'FreeCashFlow','TotalRevenue', "NetIncome", "OperatingCashFlow", 'CapitalExpenditure', 'ChangeInWorkingCapital']
YA_PAYOUT_INFOS = [ 'RepurchaseOfCapitalStock', 'CashDividendsPaid', 'CommonStockIssuance']
YA_BALANCE_INFOS = ['currencyCode', 'TotalDebt', 'CashAndCashEquivalents',
                 'CommonStockEquity', "InvestedCapital",  "BasicAverageShares"]

TYPES = YA_INCOME_INFOS + YA_BALANCE_INFOS + YA_PAYOUT_INFOS

INC_CODES = [
    'RTLR', # 'TotalRevenue'
    'SIIB', # 'Total revenue (Bank)
    "SGRP", # Gross profit
    "SOPI", # Operating income
    "SDPR", # depreciation and amortizing
    "EIBT", # "Net Income Before Taxes",
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
    "QTCO", # "Total Common Shares Outstanding",
    'InvestedCapital'
]
CASH_CODES = [
    "OTLO", # "Cash from Operating Activities",
    "SOCF", # "Changes in Working Capital",
    "SCEX", # Capital Expenditures,
    "FCDP", # Total Cash Dividends Paid
    "FPSS", # Issuance (Retirement) of Stock, Net,
    "FCFL"
            ]
PAYOUT_INFOS = ["FCDP", "FPSS" ]

FINANCIAL_ST_CODES =   INC_CODES + BAL_CODES + CASH_CODES

RENAME_DIC = {
    "OperatingCashFlow" :           "OTLO",
    "NetIncome" :                   "NINC",
    'RepurchaseOfCapitalStock' :    "FPSS",
    'CashDividendsPaid' :           "FCDP",
    "CapitalExpenditure" :          "SCEX",
    "ChangeInWorkingCapital":       "SOCF",

    "CashAndCashEquivalents":   "ACAE", # "Cash & Equivalents" 
    "TotalDebt" :               "STLD", # 'TotalDebt',
    "CommonStockEquity" :       "QTLE", # "Total Equity"
    "BasicAverageShares" :      "QTCO", # "Total Common Shares Outstanding"
    'FreeCashFlow':             "FCFL"
}
SPECIAL_CURRENCIES = {
    'GBX' : {'real' : 'GBP', 'rate_factor' : 0.01}
}

class ShareFinancialStatements():
    """
    Share financial statement from degiro
    
    """
    y_financial_statements : pd.DataFrame = None
    q_inc_financial_statements = pd.DataFrame = None
    q_bal_financial_statements = pd.DataFrame = None     
    q_cas_financial_statements = pd.DataFrame = None
    q_inc_ttm_statements = pd.DataFrame = None
    last_bal_financial_statements : pd.DataFrame = None
    nb_shares = None
    financial_currency : str = None
    cash_code : str = 'ACAE'
    total_revenue_code : str = 'RTLR'
    # gross_profit_code : str = 'SGRP'
    total_payout_ratio : float = None
    fcf : float = None
    fcf_ttm : float = None
    incf : float = None
    incf_ttm : float = None
    nincf : float = None
    focf_cagr : float = np.nan
    rate_factor = 1
    rate_symb = None

    def __init__(self, session_model : SessionModelDCF, identity : ShareIdentity):
        
        self.session_model = session_model
        self.identity = identity
        self.retrieve()

        self.q_cashflow_available = ('FCFL' in self.q_cas_financial_statements)
        # compute cash flow from income
        self.y_financial_statements['FOCF'] = self.y_financial_statements["OTLO"] - self.y_financial_statements["SOCF"]
        if "OTLO" in self.q_cas_financial_statements.columns:
            self.q_cas_financial_statements['FOCF'] = self.q_cas_financial_statements["OTLO"] - self.q_cas_financial_statements["SOCF"]

        
        self.last_bal_financial_statements = self.q_bal_financial_statements.iloc[-1]
        self.compute_avg_values()

    def retrieve(self):
        """
        Retrieve financial statements
        """
        try:
            self.degiro_retrieve()
        except KeyError as e:
            print(f'{self.identity.name} : can not retrieve financial data from degiro, {e}')
        else:
            return
        try :
            self.yahoo_retrieve()
        except (AttributeError, KeyError) as e:
            raise KeyError(f'{self.identity.name} : error while retrieving financial from yahoo, {e}') from e
        except (TypeError) as e:
            raise TypeError(f'{self.identity.name} : error while retrieving financial from yahoo, {e}') from e

    def degiro_retrieve(self):
        """
        Retrieve financial statements from degiro api
        """
        try:
            financial_st = self.session_model.trading_api.get_financial_statements(
                        product_isin= self.identity.isin,
                        raw= True
                    )['data']
        except KeyError as e:
            raise KeyError(f'no financial statement found for isin {self.identity.isin}') from e
                
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
            self.total_revenue_code = 'SIIB'
        
        y_financial_statements['periodLength'] = 12
        y_financial_statements['periodType'] = 'M'
        # free cash flow            = Cash from Operating Activities + Capital Expenditures( negative), 
        bal_cols = list(set(BAL_CODES) & set(y_financial_statements.columns))
        y_financial_statements[bal_cols] = y_financial_statements[bal_cols].ffill().bfill()
        inc_cols = list(set(CASH_CODES + INC_CODES) & set(y_financial_statements.columns))
        y_financial_statements[inc_cols] = y_financial_statements[inc_cols].fillna(0)
        
        # compute free cash flow
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

        q_inc_financial_statements = gb.get_group('INC').dropna(axis=1, how= 'all').fillna(0)
        q_bal_financial_statements = gb.get_group('BAL').dropna(axis=1, how= 'all').ffill()
        q_cas_financial_statements = gb.get_group('CAS').dropna(axis=1, how= 'all').fillna(0)


        for key in BAL_CASH_CODES:
            if key  in q_bal_financial_statements.columns:
                self.cash_code = key
                break

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
            p_df['startDate_shift'] = p_df['startDate_shift'].apply(
                lambda x : x + relativedelta(days= OVERLAPING_DAYS_TOL ) if not pd.isnull(x) else x,)

            ### mark as overlaping line for which the period cover the one of previous line
            p_df["overlaping"] = p_df['startDate'] < p_df['startDate_shift']

            ### correct overlaping line by substracting from it periodLenght and values from previous line
            p_df.loc[p_df["overlaping"], value_cols] -= p_df[value_cols].shift()
            p_df.drop(['startDate', 'startDate_shift', 'overlaping'], inplace = True, axis = 1)


        # free cash flow            = Cash from Operating Activities - Capital Expenditures, 
        q_cas_financial_statements['FCFL'] = q_cas_financial_statements["OTLO"] 
        if "SCEX" in q_cas_financial_statements:
            q_cas_financial_statements['FCFL'] += q_cas_financial_statements["SCEX"]


        self.q_inc_financial_statements = q_inc_financial_statements.set_index('endDate')
        self.q_bal_financial_statements = q_bal_financial_statements.set_index('endDate')
        self.q_cas_financial_statements = q_cas_financial_statements.set_index('endDate')


       
        if self.financial_currency != self.identity.currency:
            self.convert_to_price_currency()
   
        return(0)
    
    def yahoo_retrieve(self):

        """
        Get the share associated financial infos from yahoo finance api 
        """
        symb = self.identity.symbol
        if self.identity.symbol in self.session_model.yahoo_symbol_cor:
            symb = self.session_model.yahoo_symbol_cor[symb]
        tk = yq.Ticker(symb)
        
        y_financial_statements : pd.DataFrame = tk.get_financial_data(TYPES)
        if not isinstance(y_financial_statements, pd.DataFrame):
            raise TypeError('financial data is not a valid dataframe')
        financial_currency = y_financial_statements.dropna(subset="FreeCashFlow")["currencyCode"].value_counts().idxmax()
        y_financial_statements = y_financial_statements.sort_values(by = ['asOfDate' , 'BasicAverageShares'], ascending=[1, 0])

        y_financial_statements['BasicAverageShares'] = y_financial_statements['BasicAverageShares'].ffill(axis = 0, )
        y_financial_statements = y_financial_statements.loc[y_financial_statements['currencyCode'] == financial_currency ]

        # self.unknown_last_fcf = np.isnan(y_financial_statements["FreeCashFlow"].iloc[-1])
        
        y_financial_statements = y_financial_statements.ffill(axis = 0
                                                            ).drop_duplicates(
                                                                subset = ['asOfDate', ],
                                                                keep= 'last').set_index('asOfDate')
      

        q_financial_statements = tk.get_financial_data(TYPES , frequency= "q",
                                                      trailing= False).set_index('asOfDate')
        q_financial_statements = q_financial_statements.loc[q_financial_statements['currencyCode'] == financial_currency ]

        if isinstance(q_financial_statements, pd.DataFrame) :
            q_financial_statements.ffill(axis = 0, inplace = True)
            last_date_y = y_financial_statements.index[-1]
            last_date_q = q_financial_statements.index[-1]

            for d in YA_BALANCE_INFOS:
                if d in q_financial_statements.columns :
                    y_financial_statements.loc[last_date_y , d] = q_financial_statements.loc[last_date_q , d]

        self.nb_shares = y_financial_statements["BasicAverageShares"].iloc[-1]

        y_financial_statements = y_financial_statements.rename(columns= RENAME_DIC)
        q_financial_statements = q_financial_statements.rename(columns= RENAME_DIC)
        self.y_financial_statements = y_financial_statements

        self.q_inc_financial_statements = q_financial_statements[list(set(INC_CODES) & set(q_financial_statements.columns))]
        self.q_bal_financial_statements = q_financial_statements[list(set(BAL_CODES) & set(q_financial_statements.columns))]
        self.q_cas_financial_statements = q_financial_statements[list(set(CASH_CODES) & set(q_financial_statements.columns))]

        
        self.financial_currency = y_financial_statements['currencyCode'].iloc[-1]
        if self.financial_currency != self.identity.currency:
            self.convert_to_price_currency()
            
    def convert_to_price_currency(self):
        """
        Convert financial statement from statement curency to share price history 
        curency considering the change rate history over the period
        """
        
        share_currency = self.identity.currency
        if self.identity.currency in  SPECIAL_CURRENCIES :
            share_currency = SPECIAL_CURRENCIES[self.identity.currency]['real']
            self.rate_factor = SPECIAL_CURRENCIES[self.identity.currency]['rate_factor']
        if self.session_model.market_infos is None:
            self.session_model.market_infos = MarketInfos()
        
        
        if share_currency != self.financial_currency:
            self.session_model.market_infos.update_rate_dic(
                                                            self.financial_currency, share_currency
                                                            )
            self.rate_symb = self.financial_currency +  share_currency   + "=X"
        self.y_financial_statements = self.convert_to_price_curency_df(self.y_financial_statements)
        self.q_inc_financial_statements = self.convert_to_price_curency_df(self.q_inc_financial_statements)
        self.q_cas_financial_statements = self.convert_to_price_curency_df(self.q_cas_financial_statements)
        self.q_bal_financial_statements = self.convert_to_price_curency_df(self.q_bal_financial_statements)
        

    def convert_to_price_curency_df(self, p_df):
        """
        convert a single dataframe currency
        """
        convert_value_cols = [c for c in p_df.columns if (c in FINANCIAL_ST_CODES + ['FCFL']) and (c != "QTCO")]
        if not convert_value_cols:
            return p_df
        if  self.rate_symb :
            new_df  = pd.concat(
                [
                    p_df,
                    self.session_model.market_infos.rate_history_dic[self.rate_symb] / self.rate_factor
                    ], axis = 0
                    ).sort_index()
            new_df['change_rate'] = new_df['change_rate'].ffill()
            new_df = new_df.dropna(axis=0, subset= convert_value_cols )     
            new_df[convert_value_cols] = new_df[convert_value_cols].multiply(new_df['change_rate'], axis= 0)

            return new_df
            
        p_df.loc[:,convert_value_cols] /= self.rate_factor
        return p_df

    def compute_avg_values(self):
        """
        Evaluate averaged values and ratios from financial statements
        """
        q_cas_financial_statements =  self.q_cas_financial_statements
        q_inc_financial_statements =  self.q_inc_financial_statements
        y_financial_statements = self.y_financial_statements
        
        history_avg_nb_year = min(self.session_model.history_avg_nb_year, len(self.y_financial_statements.index) - 1)
        # payout ratio calculation
        payout = 0
        for info in list(set(PAYOUT_INFOS) & set(y_financial_statements.columns)) :
            payout -= y_financial_statements[info].iloc[-self.session_model.history_avg_nb_year:].mean()
        self.total_payout_ratio = payout / y_financial_statements['NINC'].iloc[-self.session_model.history_avg_nb_year:].mean()

        complement_q_cas_fcfl = 0
        complement_q_cas_focf = 0
        complement_q_cas_otlo = 0
        q_cas_complement_time = 0
        if self.q_cashflow_available :
            complement_q_cas = q_cas_financial_statements.loc[q_cas_financial_statements.index
                                                                > y_financial_statements.index[-1]]
            # complement_q_inc_financial_infos = q_inc_financial_statements.loc[ q_inc_financial_statements.index > y_financial_statements.index[-1]]
            
            if 'periodType' in complement_q_cas:
                q_cas_complement_time =  complement_q_cas.loc[complement_q_cas['periodType'] == 'M','periodLength'].sum() /12 + \
                                    complement_q_cas.loc[complement_q_cas['periodType'] == 'W','periodLength'].sum() /52
            else :
                q_cas_complement_time = len(complement_q_cas.index) * 3 / 12

            
            complement_q_cas_fcfl = complement_q_cas['FCFL'].sum()
            complement_q_cas_focf = complement_q_cas['FOCF'].sum()
            complement_q_cas_otlo = complement_q_cas['OTLO'].sum()

        q_cas_nb_year_avg = history_avg_nb_year + q_cas_complement_time
        self.fcf = (y_financial_statements['FCFL'].iloc[-history_avg_nb_year:].sum() \
                    + complement_q_cas_fcfl
                    ) / q_cas_nb_year_avg
        self.incf = (y_financial_statements['FOCF'].iloc[-history_avg_nb_year:].sum() \
                    + complement_q_cas_focf
                    ) / q_cas_nb_year_avg
        self.nincf = self.fcf - self.incf

        
        # compound annual growth rate
        delta_avg_nb_year = min(self.session_model.delta_avg_nb_year, len(y_financial_statements.index) - history_avg_nb_year)
        end_start = ((y_financial_statements['OTLO'].iloc[-history_avg_nb_year + 1:].sum() + complement_q_cas_otlo) /\
                     y_financial_statements['OTLO'].iloc[-delta_avg_nb_year-history_avg_nb_year:-delta_avg_nb_year].sum()) *\
                        history_avg_nb_year / (history_avg_nb_year - 1 + q_cas_complement_time)
        if end_start >= 0:
            self.focf_cagr = end_start**(1/delta_avg_nb_year) - 1
        
        # TTM values
        ttm_start_time = q_inc_financial_statements.index[-1] - relativedelta(years= 1)
        self.q_inc_ttm_statements = q_inc_financial_statements.loc[
            q_inc_financial_statements.index > ttm_start_time]
        
        if self.q_cashflow_available :
            ttm_start_time = q_cas_financial_statements.index[-1] - relativedelta(years= 1)
            q_cas_ttm_infos = q_cas_financial_statements.loc[
                q_cas_financial_statements.index > ttm_start_time]

            # ttm real period in years
            if 'periodType' in q_cas_ttm_infos:
                ttm_period = (((q_cas_ttm_infos['periodType'] == 'M') * q_cas_ttm_infos['periodLength']).sum() /12 + (
                        (q_cas_ttm_infos['periodType'] == 'W') * q_cas_ttm_infos['periodLength']).sum() /52)
            else:
                ttm_period = len(q_cas_ttm_infos.index) * 3 / 12

            self.fcf_ttm = q_cas_ttm_infos['FCFL'].sum() / ttm_period
            self.incf_ttm = q_cas_ttm_infos['FOCF'].sum() / ttm_period
