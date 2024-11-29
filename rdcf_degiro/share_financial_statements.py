import pandas as pd
from datetime import datetime, timedelta
import json, os
from dateutil.relativedelta import relativedelta
from rdcf_degiro.session_model_dcf import SessionModelDCF, MarketInfos
from rdcf_degiro.share_identity import ShareIdentity
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt

OVERLAPING_DAYS_TOL = 7

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
    "QTCO", # "Total Common Shares Outstanding"
]
CASH_CODES = [
    "OTLO", # "Cash from Operating Activities",
    "SOCF", # "Changes in Working Capital",
    "SCEX", # Capital Expenditures,
    "FCDP", # Total Cash Dividends Paid
    "FPSS", # Issuance (Retirement) of Stock, Net,
            ]
PAYOUT_INFOS = ["FCDP", "FPSS" ]

FINANCIAL_ST_CODES =   INC_CODES + BAL_CODES + CASH_CODES

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
    last_bal_financial_statements : pd.DataFrame = None
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

        # compute cash flow from income
        y_financial_statements['INCF'] = y_financial_statements["OTLO"] - y_financial_statements["SOCF"]
        # if "SGRP" in y_financial_statements:
        #     self.gross_profit_code = "SGRP"
        # elif "SOPI" in y_financial_statements :
        #     self.gross_profit_code = "SOPI"
        # else :
        #     self.gross_profit_code = 'EIBT'
        # compute difference between gross profit and free cash flow
        # y_financial_statements['G_M_FCFL'] = y_financial_statements[self.gross_profit_code] - y_financial_statements['FCFL']
            
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

        # if "SGRP" in q_inc_financial_statements:
        #     self.gross_profit_code = "SGRP"
        # elif "SOPI" in q_inc_financial_statements :
        #     self.gross_profit_code = "SOPI"
        # else :
        #     self.gross_profit_code = 'EIBT'

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

        # cash flow from income
        q_cas_financial_statements['INCF'] = q_cas_financial_statements["OTLO"] - q_cas_financial_statements["SOCF"]

        self.q_inc_financial_statements = q_inc_financial_statements.set_index('endDate')
        self.q_bal_financial_statements = q_bal_financial_statements.set_index('endDate')
        self.q_cas_financial_statements = q_cas_financial_statements.set_index('endDate')

        # # compute gross profit or EIBT minus free cash flow 
        # q_inc_adj_cas =  pd.concat([q_cas_financial_statements,
        #                     self.q_inc_financial_statements], axis = 1, keys = ['cas', 'inc'])
        # ids_cors = (q_inc_adj_cas[('inc','periodLength')] < q_inc_adj_cas[('cas','periodLength')])
        # value_cols = ["periodLength"] + [c for c in q_inc_adj_cas['inc'].columns if c in FINANCIAL_ST_CODES]
        # q_inc_adj_cas.loc[ids_cors, ('inc', value_cols)]  += q_inc_adj_cas.shift()
        # q_inc_adj_cas = q_inc_adj_cas.dropna()
        # q_inc_adj_cas.loc[:,('cas','G_M_FCFL')] = q_inc_adj_cas[('inc',self.gross_profit_code)] - q_inc_adj_cas[('cas','FCFL')]
     
        # self.q_cas_financial_statements = q_inc_adj_cas['cas']
        # print(self.q_cas_financial_statements)

       
        if self.financial_currency != self.identity.currency:
            self.convert_to_price_currency()
   
        self.last_bal_financial_statements = self.q_bal_financial_statements.iloc[-1]


        self.compute_avg_values()
        return(0)
    
    def convert_to_price_currency(self):
        """
        Convert financial statement from statement curency to share price history 
        curency considering the change rate history over the period
        """
        rate_factor = 1
        share_currency = self.identity.currency
        if self.identity.currency in  SPECIAL_CURRENCIES :
            share_currency = SPECIAL_CURRENCIES[self.identity.currency]['real']
            rate_factor = SPECIAL_CURRENCIES[self.identity.currency]['rate_factor']
        if self.session_model.market_infos is None:
            self.session_model.market_infos = MarketInfos()
        
        if share_currency != self.financial_currency:
            self.session_model.market_infos.update_rate_dic(
                                                            self.financial_currency, share_currency
                                                            )
            rate_symb = self.financial_currency +  share_currency   + "=X"

        for p_df in [
            self.y_financial_statements,
            self.q_inc_financial_statements, 
            self.q_cas_financial_statements,
            self.q_bal_financial_statements,
            ] :
            convert_value_cols = [c for c in p_df.columns if (c in FINANCIAL_ST_CODES + ['FCFL']) and (c != "QTCO")]
         
            if share_currency != self.financial_currency:
                new_df  = pd.concat(
                    [
                        p_df,
                        self.session_model.market_infos.rate_history_dic[rate_symb] / rate_factor
                        ], axis = 0
                        ).sort_index()
                
                new_df['change_rate'] = new_df['change_rate'].ffill()
                new_df = new_df.dropna(axis=0)     
                # if 'BY6' in self.identity.symbol :
                #     print(new_df)
                new_df[convert_value_cols] = new_df[convert_value_cols].multiply(new_df['change_rate'], axis= 0)
                p_df.loc[:,convert_value_cols] = new_df[convert_value_cols]           
                
            else :
                p_df.loc[:,convert_value_cols]  /= rate_factor

    def compute_avg_values(self):
        """
        Evaluate averaged values and ratios from financial statements
        """
        q_cas_financial_statements =  self.q_cas_financial_statements
        # q_inc_financial_statements = self.q_inc_financial_statements
        y_financial_statements = self.y_financial_statements
        
        # payout ratio calculation
        payout = 0
        for info in list(set(PAYOUT_INFOS) & set(y_financial_statements.columns)) :
            payout -= y_financial_statements[info].iloc[-self.session_model.history_avg_nb_year:].mean()
        self.total_payout_ratio = payout / y_financial_statements['NINC'].iloc[-self.session_model.history_avg_nb_year:].mean()

        complement_q_cas_financial_infos = q_cas_financial_statements.loc[q_cas_financial_statements.index
                                                            > y_financial_statements.index[-1]]
        # complement_q_inc_financial_infos = q_inc_financial_statements.loc[ q_inc_financial_statements.index > y_financial_statements.index[-1]]
        
        q_cas_complement_time =  complement_q_cas_financial_infos.loc[complement_q_cas_financial_infos['periodType'] == 'M','periodLength'].sum() /12 + \
                            complement_q_cas_financial_infos.loc[complement_q_cas_financial_infos['periodType'] == 'W','periodLength'].sum() /52
        # q_inc_complement_time =  complement_q_inc_financial_infos.loc[complement_q_inc_financial_infos['periodType'] == 'M','periodLength'].sum() /12 + \
        #                     complement_q_inc_financial_infos.loc[complement_q_inc_financial_infos['periodType'] == 'W','periodLength'].sum() /52

        history_avg_nb_year = self.session_model.history_avg_nb_year
            
        q_cas_nb_year_avg = history_avg_nb_year + q_cas_complement_time
        self.fcf = (y_financial_statements['FCFL'].iloc[-history_avg_nb_year:].sum() \
                    + complement_q_cas_financial_infos['FCFL'].sum()
                    ) / q_cas_nb_year_avg
        self.incf = (y_financial_statements['INCF'].iloc[-history_avg_nb_year:].sum() \
                    + complement_q_cas_financial_infos['INCF'].sum()
                    ) / q_cas_nb_year_avg
        self.nincf = self.fcf - self.incf

        # q_inc_nb_year_avg = history_avg_nb_year + q_inc_complement_time
        # self.gp = (y_financial_statements[self.gross_profit_code].iloc[-history_avg_nb_year:].sum() \
        #             + complement_q_inc_financial_infos[self.gross_profit_code].sum()
        #             ) / q_inc_nb_year_avg
        # ninc = (y_financial_statements['NINC'].iloc[-history_avg_nb_year:].sum() \
        #             + complement_q_inc_financial_infos['NINC'].sum()
        #             ) / q_inc_nb_year_avg
        # self.ninc_m_fcf = ninc - self.fcf
        


        # lregr = LinearRegression()
        # length = len(y_financial_statements.index)
        # lregr.fit(y_financial_statements[self.gross_profit_code].values.reshape(length, 1), 
        #             y_financial_statements['FCFL'].values.reshape(length, 1),
        #             )
                    
        # print(self.identity.name, lregr.score(y_financial_statements[self.gross_profit_code].values.reshape(length, 1), 
        #             y_financial_statements['FCFL'].values.reshape(length, 1),
        #             )
        #             )
        # plt.figure()
        # if 'SDPR' in y_financial_statements.columns :
        #     plt.scatter(y_financial_statements[self.gross_profit_code], 
        #                 y_financial_statements['NINC'] )
        #     plt.scatter(y_financial_statements[self.gross_profit_code], 
        #                 y_financial_statements['NINC'] + y_financial_statements['SDPR'] )
        # plt.show()    

        ttm_start_time = q_cas_financial_statements.index[-1] - relativedelta(years= 1)
        q_cas_ttm_infos = q_cas_financial_statements.loc[
            q_cas_financial_statements.index > ttm_start_time]
        # ttm_start_time = q_inc_financial_statements.index[-1] - relativedelta(years= 1)
        # q_inc_ttm_infos = q_inc_financial_statements.loc[
        #     q_inc_financial_statements.index > ttm_start_time]
        
        # ttm real period in years
        ttm_period = (((q_cas_ttm_infos['periodType'] == 'M') * q_cas_ttm_infos['periodLength']).sum() /12 + (
                (q_cas_ttm_infos['periodType'] == 'W') * q_cas_ttm_infos['periodLength']).sum() /52)
        
        self.fcf_ttm = q_cas_ttm_infos['FCFL'].sum() / ttm_period
        self.incf_ttm = q_cas_ttm_infos['INCF'].sum() / ttm_period

        # self.gp_ttm = q_inc_ttm_infos[self.gross_profit_code].sum() / (((q_inc_ttm_infos['periodType'] == 'M') * q_inc_ttm_infos['periodLength']).sum() /12 + (
        #         (q_inc_ttm_infos['periodType'] == 'W') * q_inc_ttm_infos['periodLength']).sum() /52)