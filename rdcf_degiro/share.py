from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json, os
from scipy.optimize import minimize_scalar
from typing import Any, Callable
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from dateutil.relativedelta import relativedelta
from tabulate import tabulate

from rdcf_degiro.session_model_dcf import SessionModelDCF, MarketInfos
from rdcf_degiro.share_financial_statements import ShareFinancialStatements
from rdcf_degiro.share_identity import ShareIdentity

ERROR_QUERRY_PRICE = 2

EURONEXT_ID = '710'
NASDAQ_ID = '663'


RATIO_CODES = [
    'BETA',
    "PEEXCLXOR",  # "P/E excluding extraordinary items - TTM",
    # "APRFCFPS", # "Price to Free Cash Flow per Share - most recent fiscal year",
    "TTMROIPCT" # "Return on investment - trailing 12 month",
    ] 



SPECIAL_CURRENCIES = {
    'GBX' : {'real' : 'GBP', 'rate_factor' : 0.01}
}

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

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
    # mean_g_fcf : float = None
    mean_g_tr : float = None
    capital_cost : float = None
    g : float = np.nan
    g_ttm : float = np.nan
    # g_gp : float = np.nan
    # g_gp_ttm : float = np.nan
    g_incf : float = np.nan
    g_incf_ttm : float = np.nan
    
    history_in_financial_currency : pd.DataFrame = None
    stock_equity :float = None

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

            
    def compute_complementary_values(self,):
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
        
        last_bal_financial_statements =  self.financial_statements.last_bal_financial_statements
        
        stock_equity = self.financial_statements.last_bal_financial_statements['QTLE'] # CommonStockEquity
        self.stock_equity = stock_equity
        if self.session_model.capital_cost_equal_market :
            self.capital_cost = market_infos.market_rate
        else : 
            self.capital_cost = free_risk_rate + self.beta * (market_infos.market_rate - free_risk_rate)
        
        total_debt = last_bal_financial_statements['STLD'] # 'TotalDebt',

        self.cmpc = self.capital_cost * stock_equity/(total_debt + stock_equity) + \
                    market_infos.debt_cost * (1-self.session_model.taxe_rate) * total_debt/(total_debt + stock_equity)
                        
        #net debt     =  total_debt - cash and cash equivalent
        self.net_debt = total_debt - last_bal_financial_statements[
            self.financial_statements.cash_code]

        # if self.market_cap is None:
        #     self.retrieve_values()
        
        self.net_market_cap = self.market_cap + self.net_debt
        self.debt_to_equity = total_debt / stock_equity
        self.price_to_book = self.market_cap / stock_equity

        ### terminal price to fcf calculation from  history
        if self.history is None:
            self.retrieve_history()

        df_multiple = pd.concat([self.history_in_financial_currency, 
                                y_financial_statements[["QTCO" , 'FCFL']]
                                ], axis = 0).sort_index().ffill().dropna()


        df_multiple['price_to_fcf'] = df_multiple['QTCO'] * df_multiple['close'] / df_multiple['FCFL']

        if self.session_model.price_to_fcf_avg_method == 'harmonic':
            self.price_to_fcf = len(df_multiple) / (1 / df_multiple['price_to_fcf']).sum()
            
        elif self.session_model.price_to_fcf_avg_method == 'median':
            self.price_to_fcf = df_multiple['price_to_fcf'].median()
        else: # arithmetic
            self.price_to_fcf = df_multiple['price_to_fcf'].mean()
        
        self.price_to_fcf_terminal = max(
            self.session_model.terminal_price_to_fcf_bounds[0],
            1 / max(1/self.price_to_fcf, 1/self.session_model.terminal_price_to_fcf_bounds[1])
            )

        
        ### calculation of fcf growth
        # history_avg_nb_year = self.session_model.history_avg_nb_year
        # fcf_se_ratio = self.fcf_ttm\
        #     /y_financial_statements['FCFL'].iloc[-history_avg_nb_year]
        
        # if fcf_se_ratio < 0:
        #     self.mean_g_fcf = np.nan
        # else :
        #     self.mean_g_fcf = (fcf_se_ratio)**(1/q_cas_nb_year_avg) - 1


        # self.mean_g_tr = (y_financial_statements[self.financial_statements.total_revenue_code].iloc[-1]\
        #                 /y_financial_statements[self.financial_statements.total_revenue_code].iloc[-history_avg_nb_year])**(1/q_cas_nb_year_avg) - 1
       
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
  
        if self.stock_equity < 0:
            print(f"{self.identity.name} : negative stock equity can not compute DCF")
            return
        
        fcf = start_fcf  if start_fcf else self.financial_statements.fcf
 
        if self.session_model.use_multiple:
            up_bound = 2
        else :
            up_bound = self.cmpc

        if self.session_model.use_multiple :
            if self.price_to_fcf_terminal < 0:
                print(f"{self.identity.name} negative terminal price to fcf multiple, can not compute RDCF")
                return
        
        # compute g from mean fcf
        if fcf < 0 :
            print(f"{self.identity.name} : negative free cash flow mean, can not compute RDCF")
        else :
            self.g = minimize_scalar(eval_dcf_, args=(self, fcf, False),
                                method= 'bounded', bounds = (-1, up_bound)).x

        # compute g_from_ttm from last fcf
        if self.financial_statements.fcf_ttm < 0:
            print(f"{self.identity.name} : negative TTM free cash flow, can not compute TTM RDCF")
        else :
            self.g_ttm = minimize_scalar(eval_dcf_, args=(self, self.financial_statements.fcf_ttm, False),
                                method= 'bounded', bounds = (-1, up_bound)).x
        if pr:
            print(f"Croissance correspondant au prix courrant: {self.g*100:.2f}%")
            self.get_dcf(self.g, start_fcf= start_fcf, pr = pr)

        # # compute g from income cash flow
        
        if self.financial_statements.incf < 0 :
            print(f"{self.identity.name} : negative cash flow from income mean, can not compute RDINCF")
        else :
            self.g_incf = minimize_scalar(eval_dincf_, args=(self, self.financial_statements.incf),
                                method= 'bounded', bounds = (-1, up_bound)).x

        # # compute g_from_ttm from last fcf
        if self.financial_statements.incf_ttm < 0:
            print(f"{self.identity.name} : negative TTM cash flow from income mean, can not compute TTM RDINCF")
        else :
            self.g_incf_ttm = minimize_scalar(eval_dincf_, args=(self, self.financial_statements.incf_ttm),
                                method= 'bounded', bounds = (-1, up_bound)).x
            

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        """
        Get price corresponding to given growth rate with discouted
        cash flow methods
        """
        if self.cmpc is None :
            self.compute_complementary_values()
        if g is None :
            g = self.mean_g_tr
        if not start_fcf  :
            start_fcf = self.financial_statements.fcf
        eval_dcf_(g, self, start_fcf, pr)

    
    def eval_dcf(self, g :  float, fcf : float, pr = False,):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.cmpc
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
            val_share = (enterprise_value - self.net_debt)/ self.nb_shares
            nyear_disp = min(10,nb_year_dcf)
            annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
            table = np.array([ np.concatenate((fcf_ar[:nyear_disp] ,[vt])),
                              np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
            print(f"Prévision pour une croissance de {g*100:.2f}% :")
            print(tabulate(table, floatfmt= ".4e",
                           showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"],
                           headers= annees))

            print(f"Valeur DCF de l'action: {val_share:.2f} {self.identity.currency:s}")

        return (enterprise_value / self.net_market_cap - 1)**2
    
    def eval_dincf(self, g :  float, incf : float):
        """
        computes company value regarding its actuated free cash flows assuming growing gross profit 
        and constant difference between gross profit and fcf and compares it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.cmpc
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        nb_year_dcf = self.session_model.nb_year_dcf

        if self.session_model.use_multiple :
            # fcf = incf + nincf
            vt = (incf * (1+g)**(nb_year_dcf)  + self.financial_statements.nincf) * self.price_to_fcf_terminal
        else :
            vt = (incf * (1+g)**(nb_year_dcf)  + self.financial_statements.nincf) / (cmpc - g)
        vt_act = vt / (1+cmpc)**(nb_year_dcf)

        a = (1+g)/(1+cmpc)
        b = 1/(1+cmpc)
        fcf_act_sum = incf * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf)) + \
                    self.financial_statements.nincf * ((b**nb_year_dcf - 1)/(b-1) - 1 + b**(nb_year_dcf))
 

        # a = (1+g)/(1+cmpc)
        # b = 1/(1+cmpc)
        # fcf_act_sum = gp * ((a**nb_year_dcf - 1)/(a-1) - 1 + a**(nb_year_dcf)) - self.gp_m_fcf * ((b**nb_year_dcf - 1)/(b-1) - 1 + b**(nb_year_dcf))
        # fcf_act_sum = np.array([(gp * (1+g)**k - self.gp_m_fcf)/(1+cmpc)**k for k in range(1,1 + nb_year_dcf)]).sum()


        enterprise_value = fcf_act_sum + vt_act

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

def eval_dincf_(g, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    sharevalues : ShareValues = data[0]
    incf : float = data[1]

    return sharevalues.eval_dincf(g = g, incf = incf)