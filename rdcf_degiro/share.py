import json, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from dateutil.relativedelta import relativedelta

from rdcf_degiro.session_model_dcf import SessionModelDCF
from rdcf_degiro.financial_statements import FinancialStatements, FinancialForcast
from rdcf_degiro.share_identity import ShareIdentity

ERROR_QUERRY_PRICE = 2

# EURONEXT_ID = '710'
# NASDAQ_ID = '663'


RATIO_CODES = [
    'BETA',
    "PEEXCLXOR",  # "P/E excluding extraordinary items - TTM",
    # "APRFCFPS", # "Price to Free Cash Flow per Share - most recent fiscal year",
    "TTMROIPCT" ,# "Return on investment - trailing 12 month",
    "TTMROEPCT", # return on equity
    # "FOCF_AYr5CAGR" #"Free Operating Cash Flow, 5 Year CAGR",
    ]


SPECIAL_CURRENCIES = {
    'GBX' : {'real' : 'GBP', 'rate_factor' : 0.01}
}

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

class SharePrice(ShareIdentity):
    history : pd.DataFrame = None
    history_in_financial_currency : pd.DataFrame = None
    current_price : float = None
    price_series_str : str = None

    
    def retrieve_history(self):
        """
        retrieve current history from charts
        """
        print(f'{self.name} : retrieves price history                      ', flush= True, end = "\r")

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

        if chart is None:
            raise ValueError('error while fetching price history chart')

        history = pd.DataFrame(data=chart.series[0].data, columns = ['time', 'close'])
        self.current_price = history['close'].iloc[-1]

        if self.session_model.use_last_intraday_price:
                self.retrieve_intra_day_price()

        last_day_current_month = last_day_of_month(chart.series[0].expires)
        for i in range(len(history)) :
            history.iloc[-1-i,0] = last_day_current_month - relativedelta(months= i)
    
        self.history = history.set_index('time').tz_localize(None)
    
        self.history_in_financial_currency = self.history.iloc[:-1].copy()
    
    def retrieve_intra_day_price(self):
        """
        retrieve current price from charts
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
            print(f'{self.name} : warning, not enought data to retrieve intra day price')


class ShareValues(SharePrice, FinancialStatements, FinancialForcast):
    """
    Data retrieved from degiro api with get_company_ratio
    """
    market_cap : float = None
    roic: float = np.nan
    roe : float = np.nan
    beta : float = None
    per : float = np.nan
    price_to_fcf : float = None
    price_to_fcf_terminal : float = None
    _market_wacc : float = None

    # history_growth : float = None # free oerating cash flow compound annual  growth

        
    def retrieve_values(self):
        """
        retrieve ratio values
        """
        try:
            self.degiro_values_retrieve()
        except KeyError as e:
            print(f'{self.name} : can not retrieve value ratios from degiro api, {e}     ')
        else:
            return
        
        self.yahoo_values_retrieve()
        # try:
        #     self.yahoo_retrieve()
        # except (KeyError,TypeError) as e:
        #     raise TypeError(f'{self.name} : error while retrieving value ratios from yahoo, {e}     ') from e
        

    def degiro_values_retrieve(self):
        """
        retrieve company ratios fom degiro api
        """
        print(f'{self.name} : retrieves ratio from degiro              ', flush= True, end = "\r")

        statement_path = os.path.join(self.session_model.output_folder, 
                                   f"{self.symbol}_company_ratio.json")
        if self.session_model.update_statements_need(statement_path):
            _ratios = self.session_model.get_company_ratios(
                product_isin=self.isin, 
                raw = True
            )
            if 'data' not in _ratios:
                raise KeyError(f'ratio data not available for isin {self.isin}')
            ratios = _ratios['data']
            with open(statement_path, 
                        "w", 
                        encoding= "utf8") as outfile: 
                json.dump(ratios, outfile, indent = 4)
        else:
            with open(statement_path, "r", encoding= "utf8") as json_file:
                ratios = json.load(json_file) 

        self.market_cap = self.nb_shares * self.current_price 

        ratio_dic = {}
        for rg in ratios['currentRatios']['ratiosGroups'] :
            for item in rg['items']:
                if item['id'] in RATIO_CODES:
                    if 'value' in item:
                        ratio_dic[item['id']] = item['value']
                    else:
                        print(f"{self.name} : Warning {item['id']} value not found")


        if 'BETA' in ratio_dic:
            self.beta    = float(ratio_dic['BETA'])

        if "PEEXCLXOR" in ratio_dic : 
            self.per     = float(ratio_dic["PEEXCLXOR"])  # "P/E excluding extraordinary items - TTM",

        if "TTMROIPCT" in ratio_dic : 
            self.roic = float(ratio_dic['TTMROIPCT']) / 100 # return on investec capital - TTM
        if "TTMROEPCT" in ratio_dic : 
            self.roe = float(ratio_dic['TTMROEPCT']) / 100 # return on equity - TTM

        # if "FOCF_AYr5CAGR" in ratio_dic : 
        #     self.history_growth = float(ratio_dic['FOCF_AYr5CAGR']) / 100 # return on investec capital - TTM
        # if not self.current_price:
        #     self.retrieve_current_price()


    def yahoo_values_retrieve(self):
        """
        compute ratios from degiro statements
        """
        print(f'{self.name} : retrieves ratio from yahoo              ', flush= True, end = "\r")
        net_income = self.inc_ttm_statements['NINC']

        self.market_cap = self.nb_shares * self.current_price
        self.per = self.market_cap / net_income
        
        # return on invested capital
        invested_capital = self.last_bal_statements['InvestedCapital']
        if invested_capital > 0 :
            self.roic = net_income / invested_capital

        # return on equity
        if self.stock_equity > 0 :
            self.roe = net_income / self.stock_equity

    @property
    def total_debt(self):
        return self.last_bal_statements['STLD']
    
    @property
    def net_debt(self):
                #net debt     =  total_debt - cash and cash equivalent
        return  self.total_debt - self.last_bal_statements[
                self.cash_code]
    @property
    def stock_equity(self):
        return self.last_bal_statements['QTLE']
    
    @property
    def market_capital_cost(self):
        market_infos = self.session_model.rate_info
        if (not self.session_model.use_beta) or (not self.beta):
            return market_infos.market_rate
        
        free_risk_rate = market_infos.free_risk_rate
        return free_risk_rate + self.beta * (market_infos.market_rate - free_risk_rate)
    
    @property
    def market_wacc(self):
        if self._market_wacc is None:
            self._market_wacc = self._get_market_wacc()
        return self._market_wacc
    
    @property
    def net_market_cap(self):
        return self.market_cap + self.net_debt
    
    @property
    def debt_to_equity(self) :
        return self.total_debt / self.stock_equity

    @property
    def price_to_book(self):
        return  self.market_cap / self.stock_equity
    
    def _get_market_wacc(self) :
        stock_equity = self.stock_equity
        total_debt = self.total_debt
        if stock_equity >= 0 :
            return self.market_capital_cost * stock_equity/(total_debt + stock_equity) + \
                    self.session_model.rate_info.debt_cost * (1-self.session_model.taxe_rate) * total_debt/(total_debt + stock_equity)
       
        return  self.market_capital_cost * -stock_equity/total_debt + \
                self.session_model.rate_info.debt_cost * (1-self.session_model.taxe_rate) * (total_debt + stock_equity)/ total_debt

    def compute_complementary_values(self,):
        """
        compute value and ratios from financial statements and market infos 
        before dcf calculation
        """
        print(f'{self.name} : compute complementary values                     ', flush= True, end='\r')
        if self.y_statements is None:
            self.retrieve_financials()

        y_statements = self.y_statements

        df_multiple = pd.concat([self.history_in_financial_currency, 
                                y_statements[["QTCO" , 'FCFL']]
                                ], axis = 0).sort_index().ffill().dropna()

        df_multiple['price_to_fcf'] = df_multiple['QTCO'] * df_multiple['close'] / df_multiple['FCFL']

        # price to fcf multilple calculated as harmonic mean of history:
        self.price_to_fcf = len(df_multiple) / (1 / df_multiple['price_to_fcf']).sum()
            
        self.price_to_fcf_terminal = max(
            self.session_model.terminal_price_to_fcf_bounds[0],
            1 / max(1/self.price_to_fcf, 1/self.session_model.terminal_price_to_fcf_bounds[1])
            )

        return(0)


class ShareDCFModule(ShareValues):

    g           : float = np.nan
    g_ttm       : float = np.nan
    # g_incf : float = np.nan
    # g_incf_ttm : float = np.nan
    g_delta_forcasted_assumed : float = np.nan
    forcasted_wacc : float = np.nan

    @property
    def forcasted_capital_cost(self):
        total_debt =  self.total_debt
        stock_equity =  self.stock_equity
        if np.isnan(self.forcasted_wacc) :
            return np.nan
        if stock_equity >= 0 :
            return (self.forcasted_wacc - \
                    self.session_model.rate_info.debt_cost * (1-self.session_model.taxe_rate) *\
                        total_debt/(total_debt + stock_equity)) *\
                        (total_debt + stock_equity)/stock_equity
        
        return (self.forcasted_wacc - \
                    self.session_model.rate_info.debt_cost * (1-self.session_model.taxe_rate) *\
                        (total_debt + stock_equity) /total_debt ) *\
                        - stock_equity /(total_debt + stock_equity)


    def compute_dcf(self, start_fcf : float = None):

        """
        Evaluate company assumed growth rate from fundamental financial data
        """
        print(f'{self.name} : compute dcf values                     ', )
        fcf = start_fcf  if start_fcf else self.fcf
 
        if self.session_model.use_multiple:
            up_bound = 2
        else :
            up_bound = self.market_wacc

        if self.session_model.use_multiple :
            if self.price_to_fcf_terminal < 0:
                print(f"{self.name} negative terminal price to fcf multiple, can not compute RDCF")
                return
        # compute g from mean fcf
        if fcf < 0 :
            print(f"{self.name} : negative free cash flow mean, can not compute RDCF")
        else :
            self.g = minimize_scalar(_residual_dcf_on_g, args=(self, fcf,  False),
                                method= 'bounded', bounds = (-1, up_bound)).x
            if not np.isnan(self.forcasted_sal_growth)  :
                self.forcasted_wacc = minimize_scalar(_residual_dcf_on_wacc, args=(self, fcf,  False),
                                    method= 'bounded', bounds = (-1, 2)).x

                self.g_delta_forcasted_assumed = self.forcasted_sal_growth - self.g

        # # compute g from income cash flow
        
        # if self.financial_statements.focf < 0 :
        #     print(f"{self.name} : negative cash flow from income mean, can not compute RDINCF")
        # else :
        #     self.g_incf = minimize_scalar(_residual_dincf_on_g, args=(self, self.financial_statements.focf),
        #                         method= 'bounded', bounds = (-1, up_bound)).x

        if  self.q_cashflow_available :
            # compute g_from_ttm from last fcf
            if self.fcf_ttm < 0:
                print(f"{self.name} : negative TTM free cash flow, can not compute TTM RDCF")
            else :
                self.g_ttm = minimize_scalar(_residual_dcf_on_g, args=(self, self.fcf_ttm,  False),
                                    method= 'bounded', bounds = (-1, up_bound)).x
        
            # # compute g_from_ttm from last incf
            # if self.financial_statements.focf_ttm < 0:
            #     print(f"{self.name} : negative TTM cash flow from income mean, can not compute TTM RDINCF")
            # else :
            #     self.g_incf_ttm = minimize_scalar(_residual_dincf_on_g, args=(self, self.financial_statements.focf_ttm),
            #                     method= 'bounded', bounds = (-1, up_bound)).x
            

    def residual_dcf(self, g :  float, fcf : float, wacc : float , pr = False,):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        nb_year_dcf = self.session_model.nb_year_dcf
        if self.session_model.use_multiple :
            vt = fcf * (1+g)**(nb_year_dcf ) * self.price_to_fcf_terminal
        else :
            vt = fcf * (1+g)**(nb_year_dcf ) / (wacc - g)
        vt_act = vt / (1+wacc)**(nb_year_dcf)

        a = (1+g)/(1+wacc)
        # fcf * sum of a**k for k from 1 to nb_year_dcf 
        fcf_act_sum = fcf * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf))
        enterprise_value = fcf_act_sum + vt_act
     

        # if pr :
        #     fcf_ar = np.array([fcf * (1+g)**(k) for k in range(1,1 + nb_year_dcf)])
        #     act_vec = np.array([1/((1+cmpc)**k) for k in range(1,1 + nb_year_dcf)])
        #     fcf_act = fcf_ar * act_vec
        #     print("\r")
        #     val_share = (enterprise_value - self.net_debt)/ self.financial_statements.nb_shares
        #     nyear_disp = min(10,nb_year_dcf)
        #     annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
        #     table = np.array([ np.concatenate((fcf_ar[:nyear_disp] ,[vt])),
        #                       np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
        #     print(f"Prévision pour une croissance de {g*100:.2f}% :")
        #     print(tabulate(table, floatfmt= ".4e",
        #                    showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"],
        #                    headers= annees))

            # print(f"Valeur DCF de l'action: {val_share:.2f} {self.currency:s}")

        return (enterprise_value / self.net_market_cap - 1)**2
    
    def eval_dincf(self, g :  float, incf : float):
        """
        computes company value regarding its actuated free cash flows assuming growing gross profit 
        and constant difference between gross profit and fcf and compares it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.market_wacc
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        nb_year_dcf = self.session_model.nb_year_dcf

        if self.session_model.use_multiple :
            # fcf = incf + nincf
            vt = (incf * (1+g)**(nb_year_dcf)  + self.nincf) * self.price_to_fcf_terminal
        else :
            vt = (incf * (1+g)**(nb_year_dcf)  + self.nincf) / (cmpc - g)
        vt_act = vt / (1+cmpc)**(nb_year_dcf)

        a = (1+g)/(1+cmpc)
        b = 1/(1+cmpc)
        fcf_act_sum = incf * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf)) + \
                    self.nincf * ((b**nb_year_dcf - 1)/(b-1) - 1 + b**(nb_year_dcf))

        # a = (1+g)/(1+cmpc)
        # b = 1/(1+cmpc)
        # fcf_act_sum = gp * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf)) - self.gp_m_fcf * ((b**nb_year_dcf - 1)/(b-1) - 1 + b**(nb_year_dcf))
        # fcf_act_sum = np.array([(gp * (1+g)**k - self.gp_m_fcf)/(1+cmpc)**k for k in range(1,1 + nb_year_dcf)]).sum()


        enterprise_value = fcf_act_sum + vt_act

        return (enterprise_value / self.net_market_cap - 1)**2

def _residual_dcf_on_g(g, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    dcf : ShareDCFModule = data[0]
    fcf : float = data[1]
    pr = data[2]

    return dcf.residual_dcf(g = g,
                        fcf = fcf,
                        wacc = dcf.market_wacc, 
                        pr = pr)

def _residual_dcf_on_wacc(wacc, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    dcf : ShareDCFModule = data[0]
    fcf : float = data[1]
    pr = data[2]
    return dcf.residual_dcf(g = dcf.forcasted_sal_growth,
                        fcf = fcf,
                        wacc = wacc,
                        pr = pr)

# def _residual_dincf_on_g(g, *data):
#     """
#     reformated Share.eval_dcf() function for compatibility with minimize_scalar
#     """
#     sharevalues : ShareValues = data[0]
#     incf : float = data[1]

#     return sharevalues.eval_dincf(g = g, incf = incf)

class Share(ShareDCFModule):
    """
    Object containing a share and its financial informations
    """

    product_type :str = None
    exchange_id : str = None
    share_currency :str = None

    def __init__(self, s_dict : dict = None,
            session_model : SessionModelDCF  = None):

        self.__dict__.update(s_dict)
        self.share_currency = s_dict['currency']
        self.session_model = session_model
        # if self.vwd_identifier_type_secondary =='issueid':
        #     serie_id = self.vwd_id_secondary

        # if self.vwd_identifier_type =='vwdkey':
        #     series_id_name = "vwdkey"
        #     serie_id = self.vwd_id

        series_id_name = self.vwd_identifier_type
        serie_id = self.vwd_id
        try :
            self.price_series_str = f"price:{series_id_name}:{serie_id}"
        except TypeError as e:
            raise TypeError(
            f"not valid {series_id_name} type : {type(serie_id)}  history\
                             chart not available for the quote") from e

    def retrieves_all_values(self,):
        """
        Get all the share associated financial infos from degiro api 
        """

        self.retrieve_financials()

        ## forcast
        try :
            self.retrieve_forcasts()
        except KeyError as e:
            print(f'{self.name} : can not retrieve financial forcast from degiro, {e}')

        try:
            self.retrieve_history()
        except (KeyError) as e:
            raise KeyError(f'error while retrieving price history, {e}') from e
        
        try :
            self.retrieve_values()
            self.compute_complementary_values()
        except (KeyError, TypeError) as e:
            raise KeyError(f"{self.name} : error while generating values \n {type(e).__name__} : {e}") from e
        
        self.compute_dcf()

        # self.eval_beta()