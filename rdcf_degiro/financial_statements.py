from dataclasses import dataclass
import os
import json
import re
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from requests.exceptions import RetryError
import yfinance as yf
from rdcf_degiro.session_model_dcf import SessionModelDCF

# from sklearn.linear_model import LinearRegression
# import matplotlib.pylab as plt
OVERLAPING_DAYS_TOL = 7

YA_INCOME_INFOS = [ 'FreeCashFlow','TotalRevenue', "NetIncome",
                    "OperatingCashFlow", 'CapitalExpenditure', 'ChangeInWorkingCapital', 
                    "InterestExpense",
                    "NetInterestIncome",
                    "EBITDA", 
                    "PretaxIncome",
                    "DepreciationAndAmortization"]
YA_PAYOUT_INFOS = [ 'RepurchaseOfCapitalStock', 'CashDividendsPaid', 'CommonStockIssuance']
YA_BALANCE_INFOS = ['TotalDebt', 'CurrentLiabilities', 'CashAndCashEquivalents',
                 'CommonStockEquity', "InvestedCapital",  "BasicAverageShares"]

YA_DATA = YA_INCOME_INFOS + YA_BALANCE_INFOS + YA_PAYOUT_INFOS

INC_CODES = [
    'RTLR', # 'TotalRevenue'
    'SIIB', # 'Total revenue (Bank)
    # "SGRP", # Gross profit
    "SOPI", # Operating income
    "SDPR", # depreciation and amortizing
    "SUIE", # unusual expense
    "EIBT", # "Net Income Before Taxes",
    "NINC", # "NetIncome", 
    "SNIN", # "Non operating net interest income"
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
    "SDED", # depreciation and amortizing
    "SOCF", # "Changes in Working Capital",
    "SCEX", # Capital Expenditures,
    "FCDP", # Total Cash Dividends Paid
    "FPSS", # Issuance (Retirement) of Stock, Net,
    "FCFF"
            ]
PAYOUT_INFOS = ["FCDP", "FPSS" ]

FINANCIAL_ST_CODES =   INC_CODES + BAL_CODES + CASH_CODES

RENAME_DIC_YAHOO = {
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
    'FreeCashFlow':             "FCFF",
    "NetInterestIncome" : "SNIN"
}

RENAME_DIC_DEGIRO = {
    "SOPI" : "EBIT",}
SPECIAL_CURRENCIES = {
    'GBX' : {'real' : 'GBP', 'rate_factor' : 0.01}
}

def multiply_df_scalar(df: pd.DataFrame, scalar: float, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    multiply a dataframe by a scalar
    """
    new_df = df.copy()
    cols = [c for c in new_df.columns if exclude_cols is None or c not in exclude_cols]
    new_df.loc[:, cols] *= scalar

    return new_df

class YahooRetrieveError(Exception):
    """
    Exception relative to yahooretrieve function
    """

class DegiroRetrieveError(Exception):
    """
    Exception relative to Degirotrieve function
    """
    pass

class MissingDataError(Exception):
    """
    Exception relative to missing data in financial statements
    """
    pass


class CurrencyConvertrer():
        def __init__(self, session_model : SessionModelDCF,
                     share_params : dict,
                     currency_convertible_cols : list[str] = None):
            
            self.session_model = session_model
            self.price_currency : str = share_params.get("price_currency")
            self.currency_convertible_cols = currency_convertible_cols
            

        def convert_to_price_currency_attr(self, val: float | pd.DataFrame, statements_currency: str = None) -> pd.DataFrame:
            """
            convert a single dataframe currency
            """
    
            if statements_currency == self.price_currency:
                return val
            rate_factor = 1
            price_currency = self.price_currency
            if self.price_currency in  SPECIAL_CURRENCIES :
                price_currency = SPECIAL_CURRENCIES[self.price_currency]['real']
                rate_factor = SPECIAL_CURRENCIES[self.price_currency]['rate_factor']
    
            rate_symb = None
            if price_currency != statements_currency:
                self.session_model.update_rate_dic(  statements_currency, price_currency
                                                                )
                rate_symb = statements_currency +  price_currency   + "=X"
    
            if isinstance(val, float):
                if  rate_symb:
                    return val * self.session_model.rate_history_dic[rate_symb]['change_rate'].iloc[-1] / rate_factor
                return val / rate_factor

            convert_value_cols = [c for c in val.columns if c in self.currency_convertible_cols]
            if not convert_value_cols:
                return val
            if  rate_symb :
                new_df  = pd.concat([
                        val,
                        self.session_model.rate_history_dic[rate_symb] / rate_factor
                        ], axis = 0
                        )
                new_df = new_df.sort_index()
                new_df['change_rate'] = new_df['change_rate'].ffill()
                new_df = new_df.dropna(axis=0, subset= convert_value_cols, how = "all" )
                new_df[convert_value_cols] = new_df[convert_value_cols].multiply(
                    new_df['change_rate'], axis= 0)
                return new_df
    
            val.loc[:,convert_value_cols] /= rate_factor
            return val


class Statements():

    currency_convertible_cols = None

    def __init__(self,
                session_model : SessionModelDCF,
                share_params : dict
                ):

        self.session_model = session_model
        self.logger = self.session_model.logger
        self.share_params = share_params
        self.price_currency :str = share_params.get("price_currency")
        self.yahoo_tk : yf.Ticker = share_params.get("yahoo_tk")
        self.symbol : str = share_params.get("symbol")
        self.name : str  = share_params.get("name")
        self.isin : str = share_params.get('isin')

        self.retrieve_from = "degiro"
        self.statements_currency : str = None
        self.ebitda : float = None # ebitda
        self.capex : float = None # capital expenditure
        self.ocfe : float = None # operating cash flow

        self.currency_converter = CurrencyConvertrer(session_model=session_model, 
                                                     share_params=share_params, 
                                                     currency_convertible_cols = self.currency_convertible_cols)


    def convert_to_price_currency(self, attr_list : List[str], rate_symb: str = None):

        """
        Convert financial statement from statement currency to share price history 
        currency considering the change rate history over the period
        """
        
        for  attr in attr_list:
            setattr(self, attr, 
                    self.currency_converter.convert_to_price_currency_attr(getattr(self, attr), rate_symb))



class FinancialStatements(Statements):
    """
    Share financial statement from degiro
    
    """

    currency_convertible_cols = [c for c in  FINANCIAL_ST_CODES  if c != "QTCO"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
        self._history_growth : float = None
        self.y_statements : pd.DataFrame = None

        self.q_inc_statements : pd.DataFrame = None
        self.q_bal_statements : pd.DataFrame = None     
        self.q_cas_statements : pd.DataFrame = None
        
        self._inc_ttm_statements_df : pd.DataFrame = None
        self._cas_ttm_statements_df : pd.DataFrame = None
        self.total_revenue_code : str = 'RTLR'

        self.roic: float = np.nan
        self.fcff : float = None # free cash flow

        self.cash_code : str = ['ACAE']

        self.last_bal_statements : pd.Series = None 
        self.q_cashflow_available : bool = None
        self._y_inc_complete_statements : pd.DataFrame = None


    @property
    def stock_equity(self):
        return self.last_bal_statements['QTLE']

    
    def _retrive_financials_src(self):
        
        if (self.retrieve_from == "degiro") and (self.share_params.get("finance_from", "degiro") == "degiro"):
            try :
                self.degiro_financial_retrieve()
                return
            except DegiroRetrieveError as e:
                self.logger.warning(f'{self.name} : can not retrieve financial data from degiro, {e}')

        self.yahoo_financial_retrieve()
        
    def retrieve_financials(self):
        """
        Retrieve financial statements
        """

        self._retrive_financials_src()

        self.y_statements.loc[:,'periodLength'] = 12
        self.y_statements.loc[:,'periodType'] = 'M'

        self.q_cashflow_available =  (self.q_cas_statements is not None) and ('OTLO' in self.q_cas_statements.columns)

        self._y_inc_complete_statements = pd.concat([self.y_statements,
                                         self._inc_ttm_statements_df],axis= 0)

        self.compute_avg_values()

        self.total_debt =  self.last_bal_statements['STLD']
        self.net_debt = self.total_debt - self.last_bal_statements[self.cash_code].sum()
        self.debt_cost = - self.interest_expense_income / self.net_debt 

        if 'EBIT' in self.inc_ttm_statements:
            self.roic = self.inc_ttm_statements['EBIT'] * (1-self.session_model.taxe_rate) / (self.stock_equity + self.net_debt)  


    def compute_avg_values(self):
        """
        Evaluate averaged values and ratios from financial statements
        """
        q_cas_statements =  self.q_cas_statements
        y_statements = self.y_statements

        history_avg_nb_year = min(self.session_model.history_avg_nb_year, len(y_statements.index))

        all_cas = pd.concat([y_statements.iloc[-history_avg_nb_year:],
                            q_cas_statements.loc[q_cas_statements.index > y_statements.index[-1]]
                            ]
                            , axis = 0).dropna(subset = 'OTLO')

        all_cas_time = all_cas.loc[all_cas['periodType'] == 'M','periodLength'].sum() /12 + \
                                all_cas.loc[all_cas['periodType'] == 'W','periodLength'].sum() /52

        self.fcff = all_cas.loc[:,'FCFF'].sum() / all_cas_time # free cash flow to firm
        self.fcfe = all_cas.loc[:,'FCFE'].sum() / all_cas_time # free cash flow to equity
        self.ocfe = all_cas.loc[:,'OCFE'].sum() / all_cas_time
        self.ocff = all_cas.loc[:,'OCFF'].sum() / all_cas_time
        self.capex = self.ocfe - self.fcfe

        if "EBITDA" in y_statements.columns:
            self.ebitda = y_statements['EBITDA'].iloc[-1]


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
    def fcff_ttm(self):
        """
        ttm free cash flow
        """
        return self.cas_ttm_statements['FCFF']

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

        df = self._y_inc_complete_statements
        df['year'] = (df.index - df.index[0]).days / 365
     
        df = df.dropna(subset = self.total_revenue_code)
        y = np.log(df[self.total_revenue_code].astype('float64'))
        if y is None:
            self.logger.warning(f'{self.name} no valid values to compute history growth')
            return np.nan

        z = np.polyfit(df['year'],y, deg = 1)
        return np.exp(z[0]) - 1

    def get_yahoo_financial_src(self):

        symb = self.symbol
        if "yahoo_symb" not in self.share_params:
            raise YahooRetrieveError(f"{symb} : yahoo_symb not defined in share_params")
        symb = self.share_params["yahoo_symb"]

        y_statements_path = os.path.join(self.session_model.output_folder,
                                                 f"{self.symbol}_y_statement.pckl")
        q_statements_path = os.path.join(self.session_model.output_folder,
                                                 f"{self.symbol}_q_statement.pckl")
        
        if self.session_model.update_statements_need(y_statements_path):
            try :
                y_statements : pd.DataFrame = pd.concat([self.yahoo_tk.get_income_stmt(pretty=False), 
                                          self.yahoo_tk.get_balance_sheet(pretty=False),
                                          self.yahoo_tk.get_cash_flow(pretty=False)
                                          ], axis = 0).T

                if y_statements.empty:
                    raise YahooRetrieveError(
                        f"{symb} symbol not available in yahoo database")
                y_statements.to_pickle(y_statements_path)

                q_statements = pd.concat([self.yahoo_tk.get_income_stmt(pretty=False, freq='quarterly'), 
                                        self.yahoo_tk.get_balance_sheet(pretty=False, freq='quarterly'),
                                        self.yahoo_tk.get_cash_flow(pretty=False, freq='quarterly')
                                                          ], axis = 0).T

                q_statements.to_pickle(q_statements_path)
                return y_statements, q_statements

            except RetryError as e:
                self.logger.warning(f'{self.name} : failed to update financial data from yahoo {e}')

        y_statements : pd.DataFrame = pd.read_pickle(y_statements_path)
        q_statements : pd.DataFrame = pd.read_pickle(q_statements_path)

        return y_statements, q_statements


    def yahoo_financial_retrieve(self):

        """
        Get the share associated financial infos from yahoo finance api 
        """
        self.logger.info(f'{self.name} : retrieves financial statement from yahoo             ')

        y_statements, q_statements = self.get_yahoo_financial_src()
        y_statements = y_statements.sort_index()
        q_statements = q_statements.sort_index()

        if not isinstance(y_statements, pd.DataFrame):
            raise TypeError('financial data is not a valid dataframe')

        y_statements = y_statements.ffill(axis = 0)

        for st in [y_statements, q_statements]:
            if 'TotalDebt' not in st:
                st['TotalDebt'] = st['CurrentLiabilities']

            st.rename(columns= RENAME_DIC_YAHOO, inplace = True)
            st['OCFF'] = st['OTLO']
            st['OCFE'] = st['OTLO']
            if 'SNIN' in st:
                st['OCFF'] -= st['SNIN'] * (1-self.session_model.taxe_rate)
            st['FCFE'] = st['OCFE'] + st['SCEX']
            st['FCFF'] = st['OCFF'] + st['SCEX']

        y_statements = multiply_df_scalar(y_statements, 1e-6, exclude_cols = ['TaxRateForCalcs'])
        q_statements = multiply_df_scalar(q_statements, 1e-6, exclude_cols = ['TaxRateForCalcs'])

        self._inc_ttm_statements_df = multiply_df_scalar(self.yahoo_tk.get_income_stmt(pretty=False, freq='trailing').T.iloc[::-1], 1e-6, exclude_cols = ['TaxRateForCalcs'])
        self._cas_ttm_statements_df = multiply_df_scalar(self.yahoo_tk.get_cash_flow(pretty=False, freq='trailing').T.iloc[::-1], 1e-6, exclude_cols = ['TaxRateForCalcs'])

        if isinstance(q_statements, pd.DataFrame) :
            q_statements.ffill(axis = 0, inplace = True)
            last_date_y = y_statements.index[-1]
            last_date_q = q_statements.index[-1]

            for d in YA_BALANCE_INFOS:
                if d in q_statements.columns :
                    y_statements.loc[last_date_y , d] = q_statements.loc[last_date_q , d]

        self.y_statements = y_statements


        self.q_inc_statements = q_statements[list(set(INC_CODES) & set(q_statements.columns))]
        self.q_bal_statements = q_statements[list(set(BAL_CODES) & set(q_statements.columns))]
        self.q_cas_statements = q_statements[list(set(CASH_CODES) & set(q_statements.columns))]

        self.nb_shares = self.y_statements["QTCO"].iloc[-1]
        self.statements_currency = self.yahoo_tk.info['financialCurrency']

        self.convert_to_price_currency(['y_statements',
                                            'q_inc_statements',
                                            'q_bal_statements', 
                                            'q_cas_statements',
                                            '_inc_ttm_statements_df',
                                            '_cas_ttm_statements_df'
                                            ], self.statements_currency )

        self.last_bal_statements = self.q_bal_statements.iloc[-1]

        y_period_ratio = pd.Timedelta(days=365) / self.q_inc_statements.index.diff()[-1]
        self.interest_expense_income = self.q_inc_statements["SNIN"].iloc[-1] * y_period_ratio

    def degiro_financial_retrieve(self):
        """
        Retrieve financial statements from degiro api
        """
        self.logger.info(f'{self.name} : retrieves financial statement from degiro            ')
        statement_path = os.path.join(self.session_model.output_folder, 
                                    f"{self.symbol}_company_financial.json")
        # print(os.path.isfile(statement_path))
        if self.session_model.update_statements_need(statement_path):

            r_financial_st = self.session_model.degiro_api.get_financial_statements(
                        product_isin= self.isin,
                        raw= True
                    )
            if r_financial_st is None:
                raise DegiroRetrieveError('no financial statement available from degiro ')
            try:
                financial_st = r_financial_st['data']
            except KeyError as e:
                raise DegiroRetrieveError(f'no financial statement found for isin {self.isin}') from e

            with open(statement_path,
                        "w", 
                        encoding= "utf8") as outfile: 
                json.dump(financial_st, outfile, indent = 4)
        else :
            with open(statement_path, "r", encoding= "utf8") as json_file:
                financial_st = json.load(json_file)

        self.statements_currency = financial_st['currency']

        
        self.degiro_retrieve_annual(financial_st= financial_st)
        self.degiro_retrieve_quarterly(financial_st= financial_st)

        self.convert_to_price_currency(['y_statements',
                                        'q_inc_statements', 
                                        'q_bal_statements', 
                                        'q_cas_statements',
                                        '_inc_ttm_statements_df',
                                        '_cas_ttm_statements_df'
                                        ], self.statements_currency)

        self.last_bal_statements = self.q_bal_statements.iloc[-1]

        perod_interest = 0
        for info in ["SNIN", "SONT"]:
            if info in self.q_inc_statements:
                perod_interest += self.q_inc_statements[info].iloc[-1]
        self.interest_expense_income = perod_interest * 12 / self.q_inc_statements['periodLength'].iloc[-1]

    def _compute_free_cash_flow(self, st : pd.DataFrame):
        # st['FCFF'] = st[op_keys[0]] * (1-self.session_model.taxe_rate) + da  # Free cash flow to firm
        st['OCFF'] = st["OTLO"]
        st['OCFE'] = st["OTLO"]
        for info in ["SNIN", "SONT"]:
            if info in st:
                # Add net non operating interest
                st['OCFF'] -= st[info] * (1-self.session_model.taxe_rate)

        st['FCFF'] = st['OCFF']
        st['FCFE'] = st["OCFE"]

        if "SCEX" in st:
            st['FCFF'] += st["SCEX"]
            st['FCFE'] += st["SCEX"]

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

        # compute EBITDA        = Operating income + depreciation and amortizing
        y_statements = y_statements.rename(columns= RENAME_DIC_DEGIRO)

        if 'EBIT' in y_statements:
            y_statements['EBITDA'] = y_statements['EBIT'] 
            da_keys = [k for k in ['SDPR', 'SDED'] if k in y_statements]
            if da_keys :
                # Add depreciation and amortization
                y_statements['EBITDA'] += y_statements[da_keys[0]]
            if 'SUIE' in y_statements :
                # Add unusual expense
                y_statements['EBITDA'] += y_statements['SUIE']

        # compute free cash flow
        self._compute_free_cash_flow(y_statements)

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

        self.cash_code = [key for key in BAL_CASH_CODES if key  in q_bal_statements.columns]

        q_inc_statements = remove_overlapping_data(q_inc_statements)

        # da_keys = [k for k in ['SDPR'] if k in q_inc_statements]
        # da = q_inc_statements[da_keys[0]] if da_keys else 0
        q_inc_statements = q_inc_statements.rename(columns= RENAME_DIC_DEGIRO)

        q_inc_statements.set_index('endDate', inplace= True)
        q_bal_statements.set_index('endDate', inplace= True)

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

        self.q_inc_statements = q_inc_statements
        self.q_bal_statements = q_bal_statements

        #### cash flows
        if 'CAS' not in gb.groups:
            raise MissingDataError(f'{self.name} : no cash flow statement found in degiro data')

        q_cas_statements = gb.get_group('CAS').dropna(axis=1, how= 'all').fillna(0)
        q_cas_statements = remove_overlapping_data(q_cas_statements)

        # free cash flow            = Cash from Operating Activities - Capital Expenditures,
        # q_cas_statements['FCFF'] = q_cas_statements[op_keys[0]] * (1-self.session_model.taxe_rate) + da  # Free cash flow to firm
        q_cas_statements['FCFF'] = q_cas_statements["OTLO"] 
        if "SCEX" in q_cas_statements:
            # q_cas_statements['FCFF'] += q_cas_statements["SCEX"]
            q_cas_statements['FCFF'] += q_cas_statements["SCEX"]

        q_cas_statements.set_index('endDate', inplace= True)
        ttm_start_time = q_cas_statements.index[-1] - relativedelta(years= 1)
        q_cas_ttm_statements = q_cas_statements.loc[
            q_cas_statements.index > ttm_start_time]

        self._cas_ttm_statements_df = q_cas_ttm_statements.sum().to_frame().T
        self._cas_ttm_statements_df.index = [q_cas_ttm_statements.index[-1]]

        ttm_period = (((q_cas_ttm_statements['periodType'] == 'M') * q_cas_ttm_statements['periodLength']).sum() /12 + (
                (q_cas_ttm_statements['periodType'] == 'W') * q_cas_ttm_statements['periodLength']).sum() /52)

        self._cas_ttm_statements_df[
            list(set(CASH_CODES) & set(self._cas_ttm_statements_df.columns))] /= ttm_period

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
