import os
import json
import re
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import yahooquery as yq
from dateutil.relativedelta import relativedelta
from rdcf_degiro.share_identity import ShareIdentity
# from sklearn.linear_model import LinearRegression
# import matplotlib.pylab as plt
OVERLAPING_DAYS_TOL = 7

YA_INCOME_INFOS = [ 'FreeCashFlow','TotalRevenue', "NetIncome",
                    "OperatingCashFlow", 'CapitalExpenditure', 'ChangeInWorkingCapital']
YA_PAYOUT_INFOS = [ 'RepurchaseOfCapitalStock', 'CashDividendsPaid', 'CommonStockIssuance']
YA_BALANCE_INFOS = ['currencyCode', 'TotalDebt', 'CashAndCashEquivalents',
                 'CommonStockEquity', "InvestedCapital",  "BasicAverageShares"]

TYPES = YA_INCOME_INFOS + YA_BALANCE_INFOS + YA_PAYOUT_INFOS

INC_CODES = [
    'RTLR', # 'TotalRevenue'
    'SIIB', # 'Total revenue (Bank)
    # "SGRP", # Gross profit
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

FINANCIAL_FCST_CODES =  ['NET', 'SAL', 'PRE', 'CPX', 'CPS']
RENAME_DIC = {
    "TotalRevenue" :                "RTLR",
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

class YahooRetrieveError(Exception):
    """
    Exception relative to yahooretrieve function
    """

class Statements(ShareIdentity):
    """
    class containing statements
    """
    y_statements : pd.DataFrame = None
    ocf : float = None # operating cash flow
    cex : float = None # capital expenditure
    rate_factor = 1
    nb_shares : float = None

    def convert_to_price_currency(self, df_attr_list : List[str], statements_currency: str = None):

        """
        Convert financial statement from statement curency to share price history 
        curency considering the change rate history over the period
        """
        if statements_currency == self.share_currency:
            return
        print(f'{self.name} : convert statement to price currency                               ')
        share_currency_special = self.share_currency
        if self.share_currency in  SPECIAL_CURRENCIES :
            share_currency_special = SPECIAL_CURRENCIES[self.share_currency]['real']
            self.rate_factor = SPECIAL_CURRENCIES[self.share_currency]['rate_factor']

        rate_symb = None
        if share_currency_special != statements_currency:
            print(f'{self.name} : retrieve {statements_currency}/{share_currency_special} history       ')
            self.session_model.update_rate_dic(  statements_currency, share_currency_special
                                                            )
            rate_symb = statements_currency +  share_currency_special   + "=X"
        for  attr in df_attr_list:
            self.__dict__[attr] = self.convert_to_price_curency_df(self.__dict__[attr], rate_symb)

    def convert_to_price_curency_df(self, p_df: pd.DataFrame, rate_symb: str = None) -> pd.DataFrame:
        """
        convert a single dataframe currency
        """
        convert_value_cols = [c for c in p_df.columns if (
            c in FINANCIAL_ST_CODES + FINANCIAL_FCST_CODES ) and (c != "QTCO")]
        if not convert_value_cols:
            return p_df
        if  rate_symb :
            new_df  = pd.concat(
                [
                    p_df,
                    self.session_model.rate_history_dic[rate_symb] / self.rate_factor
                    ], axis = 0
                    ).sort_index()
            new_df['change_rate'] = new_df['change_rate'].ffill()
            new_df = new_df.dropna(axis=0, subset= convert_value_cols )
            new_df[convert_value_cols] = new_df[convert_value_cols].multiply(
                new_df['change_rate'], axis= 0)

            return new_df

        p_df.loc[:,convert_value_cols] /= self.rate_factor
        return p_df

class FinancialForcast(Statements):
    """
    Class computing and containing fincancial forcast
    """

    y_forcasts : pd.DataFrame = None
    _forcasted_ocf_growth : float = None
    _forcasted_cex_growth : float = None
    _forcasted_ocf : np.ndarray = None
    _forcasted_cex : np.ndarray = None

    def retrieve_forcasts(self):
        """
        retrive forcated financial data
        """

        print(f'{self.name} : retrieves forcast                    ',
                flush= True, end = "\r")
        statement_path = os.path.join(self.session_model.output_folder,
                                    f"{self.symbol}_company_forcast.json")
        if self.session_model.update_statements_need(statement_path):
            try:
                estimates_summaries = self.session_model.get_estimates_summaries(
                product_isin= self.isin,
                raw=True,
                )['data']

            except KeyError as e:
                raise KeyError(f'no forcast statement found for isin {self.isin}') from e

            with open(statement_path,
                        "w", 
                        encoding= "utf8") as json_file: 
                json.dump(estimates_summaries, json_file, indent = 4)
        else :
            with open(statement_path, "r", encoding= "utf8") as json_file:
                estimates_summaries = json.load(json_file)

         ### Retrieve annual data
        data = []
        for y in estimates_summaries['annual'] :
            y_dict = {
                      "endDate"  :  pd.Timestamp(year = int(y['year']), month = 12, day = 31)}
            for statement in y['statements']:
                for  item in  statement['items'] :
                    if item['code'] in FINANCIAL_FCST_CODES:
                        y_dict[item["code"]] = item["value"]
            data.append(y_dict)

        self.y_forcasts = pd.DataFrame.from_records(
            data).set_index('endDate').sort_index()
        statements_currency = estimates_summaries['currency']

        self.convert_to_price_currency(['y_forcasts'], statements_currency)

    @property
    def forcasted_ocf_growth(self):
        """
        forcasted annual growth rate
        """
        if self._forcasted_ocf_growth is None :
            self._set_forcasted_ocf_growth()
        return self._forcasted_ocf_growth

    def _set_forcasted_ocf_growth(self):
        """
        retruned growth rate fited from estimate 
        """

        if self.y_forcasts is None:
            return

        for val in ['CPS',  'NET', 'PRE', 'SAL', ] :
            if val in self.y_forcasts :
                ys  = self.y_forcasts[val].dropna()
                if len(ys) >1 and  ys.min() > 0 :
                    y = np.log(ys.values / ys.iloc[0])
                    x = np.arange(len(y))
                    z = np.polyfit(x,y, deg = 1)
                    g = np.exp(z[0]) - 1
                    self._forcasted_ocf_growth = g
                    return
        
        print(f"{self.symbol} no valid value to compute OCF growth estimate")
        return

    def _set_forcasted_ocf(self):
        """
        retruned forcasted ocf array
        """
        if self.y_forcasts is None:
            return

        ys = None
        for val in ['CPS', 'NET', 'PRE', 'SAL' ] :
            if val in self.y_forcasts:
                ys = self.y_forcasts[val].dropna()
                ratio = self.y_statements['OTLO'].iloc[-1]/ys.iloc[0]
                if (ys.index[0].year == self.y_statements.index[-1].year) and ratio > 0:
                    ys *= ratio
                elif val == 'CPS' :
                    ys *= self.nb_shares
                else:
                    continue

                # complete forcasted ocf array with value extrapolated from forcasted growth rate
                ys = np.concat([
                        ys,
                        ys[-1] * (1+ self.forcasted_ocf_growth)**np.arange(
                            1,
                            1 + self.session_model.nb_year_dcf - len(ys))])
                
                self._forcasted_ocf = ys
                return

        # no forcasted cash flow per share provided
        self._forcasted_ocf = self.ocf * (1 + self.forcasted_ocf_growth)**np.arange(1,1 +self.session_model.nb_year_dcf)

    def _set_forcasted_cex(self):
        """
        retruned growth rate fited from estimate 
        """
        if self.y_forcasts is None:
            return

        if 'CPX' in self.y_forcasts:
            ys = self.y_forcasts['CPX'].dropna()
            if len(ys)> 1 and ys.min() > 0 :
                y = np.log(ys.values / ys.iloc[0])
                # complete forcasted cex array with value extrapolated from forcasted growth rate
                x = np.arange(len(y))
                # x = x[:, np.newaxis]
                # a, _, _, _ = np.linalg.lstsq(x,y)
                # g= np.exp(a[0]) - 1

                z = np.polyfit(x,y, deg = 1)
                g = np.exp(z[0]) - 1
                g = min(g,2)  #bound growth rate to 2
                self._forcasted_cex_growth = g
                ys = np.concat([
                        ys,
                        ys[-1] * (1+ g)**np.arange(
                            1,
                            1 + self.session_model.nb_year_dcf - len(ys))])
                self._forcasted_cex = ys
                return

        # can not compute forcasted capital expenditure growth from itself
        self._forcasted_cex_growth = self.forcasted_ocf_growth
        self._forcasted_cex = self.cex * (1 + self._forcasted_cex_growth)**np.arange(1,1 +self.session_model.nb_year_dcf)

        

    @property
    def forcasted_cex_growth(self):
        """
        forcasted annual growth rate
        """
        if self._forcasted_cex_growth is None :
            self._set_forcasted_cex()
        return self._forcasted_cex_growth

    @property
    def forcasted_ocf(self) -> np.ndarray:
        """
        array of forcasted cpx
        """
        if self._forcasted_ocf is None:
            self._set_forcasted_ocf()
        return self._forcasted_ocf
    
    @property
    def forcasted_focf(self):
        return self.forcasted_ocf - self.forcasted_cex
    
    @property
    def forcasted_cex(self) -> np.ndarray:
        """
        array of forcasted cpx
        """
        if self._forcasted_cex is None:
            self._set_forcasted_cex()
        return self._forcasted_cex

class FinancialStatements(Statements):
    """
    Share financial statement from degiro
    
    """

    q_inc_statements : pd.DataFrame = None
    q_bal_statements : pd.DataFrame = None     
    q_cas_statements : pd.DataFrame = None
    _inc_ttm_statements_df : pd.DataFrame = None
    _cas_ttm_statements_df : pd.DataFrame = None
    cash_code : str = 'ACAE'
    total_revenue_code : str = 'RTLR'
    # gross_profit_code : str = 'SGRP'
    fcf : float = None # free cash flow
    nincf : float = None
    _history_growth : float = None
    last_bal_statements : pd.Series = None 
    q_cashflow_available : bool = None
    _y_inc_complete_statements : pd.DataFrame = None

    def retrieve_financials(self):
        """
        Retrieve financial statements
        """

        try :
            self.degiro_financial_retrieve()
        except KeyError as e:
            print(f'{self.name} : can not retrieve financial data from degiro, {e}')
            self.yahoo_financial_retrieve()

        self.y_statements.loc[:,'periodLength'] = 12
        self.y_statements.loc[:,'periodType'] = 'M'

        self.q_cashflow_available = 'OTLO' in self.q_cas_statements.columns
        # compute cash flow from income
        self._y_inc_complete_statements = pd.concat([self.y_statements,
                                         self._inc_ttm_statements_df])

        self.compute_avg_values()


    def compute_avg_values(self):
        """
        Evaluate averaged values and ratios from financial statements
        """
        q_cas_statements =  self.q_cas_statements
        # q_inc_statements =  self.q_inc_statements
        y_statements = self.y_statements

        history_avg_nb_year = min(self.session_model.history_avg_nb_year, len(y_statements.index))

        all_cas = pd.concat([y_statements.iloc[-history_avg_nb_year:],
                            q_cas_statements.loc[q_cas_statements.index > y_statements.index[-1]]
                            ]
                            , axis = 0).dropna(subset = 'OTLO')

        all_cas_time = all_cas.loc[all_cas['periodType'] == 'M','periodLength'].sum() /12 + \
                                all_cas.loc[all_cas['periodType'] == 'W','periodLength'].sum() /52

        self.fcf = all_cas.loc[:,'FCFL'].sum() / all_cas_time
        self.ocf = all_cas.loc[:,'OTLO'].sum() / all_cas_time
        self.cex = self.ocf - self.fcf


    @property
    def inc_ttm_statements(self) -> pd.Series :
        """
        ttm income statements
        """
        return self._inc_ttm_statements_df.iloc[-1]
    @property
    def cas_ttm_statements(self) -> pd.Series :
        """
        ttm cash flow statements
        """
        return self._cas_ttm_statements_df.iloc[-1]
    @property
    def fcf_ttm(self):
        """
        ttm free cash flow
        """
        return self.cas_ttm_statements['FCFL']

    @property
    def total_payout_ratio(self) -> float:
        """
        Return : (dividend + repurchase) / income
        """
        payout = 0
        for info in list(set(PAYOUT_INFOS) & set(self.y_statements.columns)) :
            payout -= self.y_statements.loc[:,info].iloc[
                -self.session_model.history_avg_nb_year:].sum()
        return payout / self.y_statements.loc[:,'NINC'].iloc[
            -self.session_model.history_avg_nb_year:].sum()

    @property
    def history_growth(self):
        """
        history growth rate
        """
        if self._history_growth is None :
            self._history_growth = self._get_history_growth()
        return self._history_growth

    def _get_history_growth(self) -> float:
        """
        Return : revenu history growth calculated by logaritmic fit
        """

        y = None
        # df = self._y_cas_complete_statements
        # df['year'] = (df.index - df.index[0]).days / 365

        # if df['OTLO'].min() > 0:
        #     y = np.log(df['OTLO'].astype('float64'))
        # else:
        df = self._y_inc_complete_statements
        df['year'] = (df.index - df.index[0]).days / 365
     
        df = df.dropna(subset = self.total_revenue_code)
        y = np.log(df[self.total_revenue_code].astype('float64'))
        if y is None:
            print(f'{self.name} no valid values to compute history growth')
            return np.nan

        z = np.polyfit(df['year'],y, deg = 1)
        return np.exp(z[0]) - 1

    def yahoo_financial_retrieve(self):

        """
        Get the share associated financial infos from yahoo finance api 
        """
        print(f'{self.name} : retrieves financial statement from yahoo             ', flush= True, end = "\r")

        symb = self.symbol
        if self.symbol in self.session_model.yahoo_symbol_cor:
            symb = self.session_model.yahoo_symbol_cor[symb]

        y_statements_path = os.path.join(self.session_model.output_folder,
                                         f"{self.symbol}_y_statement.pckl")
        q_statements_path = os.path.join(self.session_model.output_folder,
                                         f"{self.symbol}_q_statement.pckl")

        if self.session_model.update_statements_need(y_statements_path):
            tk = yq.Ticker(symb, asynchronous=True)
            y_statements : pd.DataFrame = tk.get_financial_data(TYPES)
            if not isinstance(y_statements, pd.DataFrame):
                raise YahooRetrieveError(
                    f"{symb} symbol not available in yahoo database")
            y_statements.to_pickle(y_statements_path)
            q_statements : pd.DataFrame = tk.get_financial_data(TYPES , frequency= "q",
                                                        trailing= False).set_index('asOfDate')

            q_statements.to_pickle(q_statements_path)
        else:
            y_statements : pd.DataFrame = pd.read_pickle(y_statements_path)
            q_statements : pd.DataFrame = pd.read_pickle(q_statements_path)

        if not isinstance(y_statements, pd.DataFrame):
            raise TypeError('financial data is not a valid dataframe')

        currency = y_statements.dropna(
            subset="FreeCashFlow")["currencyCode"].value_counts().idxmax()
        y_statements = y_statements.sort_values(
            by = ['asOfDate' , 'BasicAverageShares'], ascending=[1, 0])

        y_statements['BasicAverageShares'] = y_statements['BasicAverageShares'].ffill(axis = 0, )
        y_statements = y_statements.loc[y_statements['currencyCode'] == currency ]

        y_statements = y_statements.rename(columns= RENAME_DIC)
        q_statements = q_statements.rename(columns= RENAME_DIC)

        y_statements = y_statements.ffill(axis = 0).drop_duplicates(
                                                                subset = ['asOfDate', ],
                                                                keep= 'last').set_index('asOfDate')

        q_statements = q_statements.loc[q_statements['currencyCode'] == currency ]
        y_statements[[c for c in y_statements.columns if c not in ['periodType', 'currencyCode']]] *= 1e-6
        q_statements[[c for c in q_statements.columns if c not in ['periodType', 'currencyCode']]] *= 1e-6

        self._inc_ttm_statements_df = y_statements.loc[
            y_statements['periodType'] == 'TTM'].iloc[[-1]]
        self._cas_ttm_statements_df = self._inc_ttm_statements_df

        max_occur_month = y_statements.index.month.value_counts().idxmax()
        y_statements = y_statements.loc[y_statements.index.month == max_occur_month ]

        if isinstance(q_statements, pd.DataFrame) :
            q_statements.ffill(axis = 0, inplace = True)
            last_date_y = y_statements.index[-1]
            last_date_q = q_statements.index[-1]

            for d in YA_BALANCE_INFOS:
                if d in q_statements.columns :
                    y_statements.loc[last_date_y , d] = q_statements.loc[last_date_q , d]

        self.y_statements = y_statements

        q_statements['periodLength'] = q_statements['periodType'].apply(
            lambda x : int(re.search('^[0-9]+',x).group()))
        q_statements['periodType'] = q_statements['periodType'].apply(
            lambda x : re.search('[A-Z]+$',x).group())

        self.q_inc_statements = q_statements[
            ['periodLength' , 'periodType'] + list(set(INC_CODES) & set(q_statements.columns))]
        self.q_bal_statements = q_statements[list(set(BAL_CODES) & set(q_statements.columns))]
        self.q_cas_statements = q_statements[
            ['periodLength' , 'periodType'] + list(set(CASH_CODES) & set(q_statements.columns))]

        self.nb_shares = self.y_statements["QTCO"].iloc[-1]

        statements_currency = y_statements['currencyCode'].iloc[-1]
        self.convert_to_price_currency(['y_statements',
                                        'q_inc_statements',
                                        'q_bal_statements', 
                                        'q_cas_statements',
                                        '_inc_ttm_statements_df',
                                        '_cas_ttm_statements_df'
                                        ], statements_currency )

        self.last_bal_statements = self.y_statements.iloc[-1]


    def degiro_financial_retrieve(self):
        """
        Retrieve financial statements from degiro api
        """
        print(f'{self.name} : retrieves financial statement from degiro            ',
              flush= True, end = "\r")
        statement_path = os.path.join(self.session_model.output_folder, 
                                    f"{self.symbol}_company_financial.json")
        # print(os.path.isfile(statement_path))
        if self.session_model.update_statements_need(statement_path):

            r_financial_st = self.session_model.get_financial_statements(
                        product_isin= self.isin,
                        raw= True
                    )
            if r_financial_st is None:
                raise KeyError('no financial statement available from degiro ')
            try:
                financial_st = r_financial_st['data']
            except KeyError as e:
                raise KeyError(f'no financial statement found for isin {self.isin}') from e

            with open(statement_path,
                        "w", 
                        encoding= "utf8") as outfile: 
                json.dump(financial_st, outfile, indent = 4)
        else :
            with open(statement_path, "r", encoding= "utf8") as json_file:
                financial_st = json.load(json_file)

        statements_currency = financial_st['currency']

        self.degiro_retrieve_annual(financial_st= financial_st)
        self.degiro_retrieve_quarterly(financial_st= financial_st)

        self.convert_to_price_currency(['y_statements',
                                        'q_inc_statements', 
                                        'q_bal_statements', 
                                        'q_cas_statements',
                                        '_inc_ttm_statements_df',
                                        '_cas_ttm_statements_df'
                                        ], statements_currency)

        self.last_bal_statements = self.q_bal_statements.iloc[-1]


    def degiro_retrieve_annual(self, financial_st : dict):
        """
        Retrieve annual data
        """
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

        y_statements = pd.DataFrame.from_records(
            data).iloc[::-1]
        y_statements['endDate'] = pd.to_datetime(y_statements['endDate'])
        y_statements = y_statements.set_index('endDate')[list(
            set(FINANCIAL_ST_CODES) & set(y_statements.columns))]
        if 'RTLR' not in y_statements.columns :
            self.total_revenue_code = 'SIIB'

        # free cash flow = Cash from Operating Activities + Capital Expenditures( negative),
        bal_cols = list(set(BAL_CODES) & set(y_statements.columns))
        y_statements[bal_cols] = y_statements[bal_cols].ffill().bfill()
        inc_cols = list(set(CASH_CODES + INC_CODES) & set(y_statements.columns))
        y_statements[inc_cols] = y_statements[inc_cols].fillna(0)

        # compute free cash flow
        y_statements['FCFL'] = y_statements["OTLO"]
        if "SCEX" in y_statements:
            y_statements['FCFL'] += y_statements["SCEX"]

        self.y_statements = y_statements

    def degiro_retrieve_quarterly(self, financial_st :dict[str, list[dict[str, list[dict[str, str]]]]]):
        """
        Retrieve quarterly data
        """
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

        q_inc_statements = gb.get_group('INC').dropna(axis=1, how= 'all').fillna(0)
        q_bal_statements = gb.get_group('BAL').dropna(axis=1, how= 'all').ffill()
        q_cas_statements = gb.get_group('CAS').dropna(axis=1, how= 'all').fillna(0)

        for key in BAL_CASH_CODES:
            if key  in q_bal_statements.columns:
                self.cash_code = key
                break

        q_inc_statements = remove_overlapping_data(q_inc_statements)
        q_cas_statements = remove_overlapping_data(q_cas_statements)

        # free cash flow            = Cash from Operating Activities - Capital Expenditures,
        q_cas_statements['FCFL'] = q_cas_statements["OTLO"] 
        if "SCEX" in q_cas_statements:
            q_cas_statements['FCFL'] += q_cas_statements["SCEX"]


        q_inc_statements.set_index('endDate', inplace= True)
        q_bal_statements.set_index('endDate', inplace= True)
        q_cas_statements.set_index('endDate', inplace= True)

        self.nb_shares = q_bal_statements["QTCO"].iloc[-1]

        # TTM values
        ### income
        ttm_start_time = q_inc_statements.index[-1] - relativedelta(years= 1)
        q_inc_ttm_statements  = q_inc_statements.loc[
            q_inc_statements.index > ttm_start_time]

        self._inc_ttm_statements_df = q_inc_ttm_statements.sum().to_frame().T
        self._inc_ttm_statements_df.index = [q_inc_statements.index[-1]]

        ttm_period = (((q_inc_ttm_statements['periodType'] == 'M') * q_inc_ttm_statements['periodLength']).sum() /12 + (
                    (q_inc_ttm_statements['periodType'] == 'W') * q_inc_ttm_statements['periodLength']).sum() /52)
        self._inc_ttm_statements_df[list(set(INC_CODES) & set(self._inc_ttm_statements_df.columns))] /= ttm_period

        #### cash flows
        ttm_start_time = q_cas_statements.index[-1] - relativedelta(years= 1)
        q_cas_ttm_statements = q_cas_statements.loc[
            q_cas_statements.index > ttm_start_time]

        self._cas_ttm_statements_df = q_cas_ttm_statements.sum().to_frame().T
        self._cas_ttm_statements_df.index = [q_cas_ttm_statements.index[-1]]

        ttm_period = (((q_cas_ttm_statements['periodType'] == 'M') * q_cas_ttm_statements['periodLength']).sum() /12 + (
                (q_cas_ttm_statements['periodType'] == 'W') * q_cas_ttm_statements['periodLength']).sum() /52)

        self._cas_ttm_statements_df[
            list(set(CASH_CODES) & set(self._cas_ttm_statements_df.columns))] /= ttm_period

        self.q_inc_statements = q_inc_statements
        self.q_bal_statements = q_bal_statements
        self.q_cas_statements = q_cas_statements

def remove_overlapping_data(p_df:pd.DataFrame):
    """
    correct period length dependant data in dataframe if period  of one row is overlapping previous
    """
    value_cols = ["periodLength"] + [c for c in p_df.columns if c in FINANCIAL_ST_CODES]

    #### eval coresponding startDate from end date and periodLength
    p_df['startDate'] = p_df.apply(lambda x : get_date_shift_back(
            x.endDate,
            months = (x.periodType == 'M') * x.periodLength,
            weeks = (x.periodType == 'W') * x.periodLength,
            ), axis = 1)
    p_df['startDate_shift'] = p_df['startDate'].shift()
    #### apply overlaping tolerance of OVERLAPING_DAYS_TOL days
    p_df['startDate_shift'] = p_df['startDate_shift'].apply(
        lambda x : x + relativedelta(days= OVERLAPING_DAYS_TOL ) if not pd.isnull(x) else x,)

    ### mark as overlaping line for which the period cover the one of previous line
    p_df["overlaping"] = p_df['startDate'] < p_df['startDate_shift']

    ### correct overlaping line by substracting from it periodLength and values from previous line
    p_df.loc[p_df["overlaping"], value_cols] -= p_df[value_cols].shift()
    p_df.drop(['startDate', 'startDate_shift', 'overlaping'], inplace = True, axis = 1)

    return p_df

def get_date_shift_back(end_date : datetime, months : int = 0, weeks :int = 0):
    return end_date - relativedelta(months= months, weeks= weeks )
