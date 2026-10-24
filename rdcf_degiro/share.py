import json, os
import numpy as np
import pandas as pd
import numpy_financial as npf
from scipy.optimize import minimize_scalar
import yfinance as yf
from rdcf_degiro.session_model_dcf import SessionModelDCF
from rdcf_degiro.financial_statements import  DegiroRetrieveError, FinancialStatements
from rdcf_degiro.price_history import    PriceHistory
from rdcf_degiro.forcast import FinancialForcast

ERROR_QUERRY_PRICE = 2
TOLERANCE_MINIMIZE = 1e-2
# EURONEXT_ID = '710'
# NASDAQ_ID = '663'
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'

RATIO_CODES = [
    'BETA',
    "PEEXCLXOR",  # "P/E excluding extraordinary items - TTM",
    'TTMPR2REV', # price to sale
    # "APRFCFPS", # "Price to Free Cash Flow per Share - most recent fiscal year",
    "TTMROIPCT" ,# "Return on investment - trailing 12 month",
    "TTMROEPCT", # return on equity
    "MKTCAP", # market cap
    # "FOCF_AYr5CAGR" #"Free Operating Cash Flow, 5 Year CAGR",
    ]


class MarketCapError(Exception):
    pass

class Share():
    """
    Object containing a share and its financial informations
    """


    def __init__(self, s_dict : dict = None,
            session_model : SessionModelDCF  = None):

        self.name : str = None
        self.isin :str = None
        self.vwd_id : str = None
        self.vwd_id_secondary : str = None
        self.symbol : str = None
        self.vwd_identifier_type : str = None
        self.vwd_identifier_type_secondary : str = None

        self.retrieve_from = "degiro"
        self.valid_retrieve = False

        self.assumed_g           : float = np.nan
        self.assumed_g_ttm       : float = np.nan
    
        self.g_delta_forcasted_assumed : float = np.nan
        self.forcasted_wacc_multiple : float = np.nan
        self.forcasted_wacc_perpetual : float = np.nan

        self.product_type :str = None
        self.exchange_id : str = None

        self.market_cap : float = None
        self.market_cap_reported : float = None
        self.roe : float = np.nan
        self.beta : float = None
        self.per : float = np.nan
        self.price_to_sales : float = np.nan
        self.value_to_ebitda : float = None
        self.value_to_ebitda_bounded : float = None

        self.forcasted_capital_cost_multiple :float = None
        self.forcasted_capital_cost_perpetual :float = None
        self.target_market_value_perpetual : float = None
        self.target_market_price_multiple : float = None

        self.use_fcfe = False
        self.error_msg = ""
        

        self.__dict__.update(s_dict)

        self.session_model = session_model


        series_id_name = self.vwd_identifier_type
        serie_id = self.vwd_id
        self.logger = self.session_model.logger
        try :
            self.price_series_str = f"price:{series_id_name}:{serie_id}"
        except TypeError as e:
            raise TypeError(
            f"not valid {series_id_name} type : {type(serie_id)}  history\
                             chart not available for the quote") from e
        
        self.share_params :dict = {"name" : self.name, 
                                   "symbol" : self.symbol, 
                                   "price_currency" : s_dict['currency'], 
                                   "price_from" : None,
                                   "price_series_str" : self.price_series_str,
                                   "isin" : self.isin
                                   }
        
        # Set special params for shares if defined in config_dict
        if self.symbol in self.session_model.symbol_params.keys():
            self.share_params.update(self.session_model.symbol_params[self.symbol])
            self.yahoo_symb = ""
            if "yahoo_tk" in self.share_params:
                self.yahoo_symb = self.share_params["yahoo_tk"]
                self.yahoo_tk = yf.Ticker(self.share_params["yahoo_tk"])
                self.share_params.update({"price_currency" : self.yahoo_tk.info['currency']})
        
        self.price_currency = self.share_params["price_currency"]
        
        self.statements = FinancialStatements(session_model, self.share_params)
        self.forcast = FinancialForcast(self.statements, session_model, self.share_params)
        self.price_history = PriceHistory(session_model, self.share_params)

    def retrieve_values(self):
        """
        retrieve ratio values
        """
        if self.retrieve_from == "degiro" :
            try:
                self.degiro_values_retrieve()
            except DegiroRetrieveError as e:
                self.logger.warning(f'{self.name} : can not retrieve value ratios from degiro api, {e}     ')
                self.yahoo_values_retrieve()
        

        self.enterprise_value = self.market_cap + self.statements.net_debt

        # if enterprise value is negative or very low use cash flow to equity
        self.use_fcfe = (self.enterprise_value / self.market_cap) < 0.05 


    def check_market_cap(self):
        """ Check consistency between reported market cap and price calculated maket cap """

        if self.statements.statements_currency == "EUR":
            self.market_cap_reliable = True
            return

        err_market_cap = abs(self.market_cap - self.market_cap_reported)/min(self.market_cap_reported, self.market_cap )
        if err_market_cap > 0.8 :
            msg = f"{self.name}  err_market_cap: {err_market_cap:.2f} reported: {self.market_cap_reported:.2f} calculated: {self.market_cap:.2f} nb shares {self.nb_shares}"
            self.market_cap_reliable = False
            raise MarketCapError(msg)

        self.market_cap_reliable = True

    
    def degiro_values_retrieve(self):
        """
        retrieve company ratios fom degiro api
        """
        self.logger.info(f'{self.name} : retrieves ratio from degiro              ')

        statement_path = os.path.join(self.session_model.output_folder, 
                                    f"{self.symbol}_company_ratio.json")
        if self.session_model.update_statements_need(statement_path):
            _ratios = self.session_model.degiro_api.get_company_ratios(
                product_isin=self.isin, 
                raw = True
            )
            if (not _ratios) or ('data' not in _ratios):
                raise DegiroRetrieveError(f'ratio data not available for isin {self.isin}')
            ratios = _ratios['data']
            with open(statement_path, "w", 
                        encoding= "utf8") as outfile:
                json.dump(ratios, outfile, indent = 4)
        else:
            with open(statement_path, "r", encoding= "utf8") as json_file:
                ratios = json.load(json_file)

        self.market_cap = self.statements.nb_shares * self.price_history.current_price

        statement_currency = ratios['currentRatios']['priceCurrency']
        ratio_dic = {}
        for rg in ratios['currentRatios']['ratiosGroups'] :
            for item in rg['items']:
                if item['id'] in RATIO_CODES:
                    if 'value' in item:
                        ratio_dic[item['id']] = item['value']
                    # else:
                    #     self.logger.warning(f"{self.name} : Warning {item['id']} value not found")


        if 'BETA' in ratio_dic:
            self.beta    = float(ratio_dic['BETA'])

        self.per  = float(ratio_dic.get("PEEXCLXOR", self.per))  # "P/E excluding extraordinary items - TTM",
        self.price_to_sales = float(ratio_dic.get('TTMPR2REV', self.price_to_sales))

        if "TTMROEPCT" in ratio_dic : 
            self.roe = float(ratio_dic['TTMROEPCT']) / 100 # return on equity - TTM

        if "MKTCAP" in ratio_dic : 
            self.market_cap_reported = float(ratio_dic['MKTCAP']) *1e-6# market cap
            self.statements.convert_to_price_currency(["market_cap_reported"], statement_currency)
            self.check_market_cap()
                

    def yahoo_values_retrieve(self):
        """
        compute ratios from degiro statements
        """
        self.logger.info(f'{self.name} : retrieves ratio from yahoo              ')
        self.logger.handlers[0].flush()
        # self.logger.info(f'{self.name} : retrieves ratio from yahoo              ', flush= True, end = "\r")
        net_income = self.statements.inc_ttm_statements['NINC']

        self.market_cap = self.nb_shares * self.current_price
        self.check_market_cap()
        self.per = self.market_cap / net_income
        self.price_to_sales = self.market_cap / self.statements.inc_ttm_statements['RTLR']

        # return on equity
        if self.statements.stock_equity > 0 :
            self.roe = net_income / self.statements.stock_equity
    
    @property
    def market_capital_cost(self):
        market_infos = self.session_model.rate_info
        if (not self.session_model.use_beta) or (not self.beta):
            return market_infos.market_rate
        
        free_risk_rate = market_infos.free_risk_rate
        return free_risk_rate + self.beta * (market_infos.market_rate - free_risk_rate)
    
    @property
    def debt_to_equity(self) :
        return self.statements.total_debt / self.market_cap

    @property
    def price_to_book(self):
        return  self.market_cap / self.statements.stock_equity
    
    def _get_market_wacc(self) :
        """ Get weight averaged capital cost """

        cc = self.market_capital_cost
        if self.use_fcfe :
            return cc
        eq = self.market_cap
        debt = self.statements.net_debt
        if debt < 0:
            eq = self.enterprise_value
            debt = self.statements.total_debt
        
        tr = self.session_model.taxe_rate
        dc = self.statements.debt_cost

        return cc * eq/(debt + eq) +  dc * (1-tr) * debt/(debt + eq)
        

    def compute_multiples(self,):
        """
        compute value and ratios from financial statements and market infos 
        before dcf calculation
        """
        self.logger.info(f'{self.name} : compute complementary values                     ')
        # self.logger.info(f'{self.name} : compute complementary values                     ', flush= True, end='\r')

        y_statements = self.statements.y_statements

        df_multiple = pd.concat([self.price_history.history_in_financial_currency, 
                                y_statements[["QTCO" , 'EBITDA', 'STLD', 'OCFE', 'OCFF'] + self.statements.cash_code]
                                ], axis = 0).sort_index().ffill().dropna()

        
        df_multiple['ENT_PRICE'] = df_multiple['QTCO'] * df_multiple['close']
        df_multiple['ENT_VALUE'] = df_multiple['ENT_PRICE'] + df_multiple['STLD'] - df_multiple[self.statements.cash_code].sum(axis = 1)
        
        df_multiple['value_to_ebitda'] = df_multiple['ENT_VALUE'] / df_multiple['EBITDA']
        df_multiple['value_to_ocff'] = df_multiple['ENT_VALUE'] / df_multiple['OCFF']
        df_multiple['price_to_ocfe'] = df_multiple['ENT_PRICE'] / df_multiple['OCFE']


        # price to fcf multilple calculated as harmonic mean of history:
        self.price_to_ocfe = len(df_multiple) / (1 / df_multiple['price_to_ocfe']).sum()
        self.value_to_ocff = len(df_multiple) / (1 / df_multiple['value_to_ocff']).sum()

        # self.value_to_ebitda = np.median(df_multiple['value_to_ebitda'])

        boud_i, bound_s = self.session_model.terminal_value_to_ebitda_bounds
        self.price_to_ocfe_bounded = max( boud_i, 1 / max(1/self.price_to_ocfe, 1/bound_s))
        self.value_to_ocff_bounded = max( boud_i, 1 / max(1/self.value_to_ocff, 1/bound_s))

        if "value_to_ebitda" in self.share_params:
            self.value_to_ebitda = self.share_params["value_to_ebitda"]
            self.value_to_ebitda_bounded = self.share_params["value_to_ebitda"]
        else :
            self.value_to_ebitda = len(df_multiple) / (1 / df_multiple['value_to_ebitda']).sum()
            self.value_to_ebitda_bounded = max( boud_i, 1 / max(1/self.value_to_ebitda, 1/bound_s))

        return(0)
        

    def retrieve_share_data(self,):
        """
        Get all the share associated financial infos from degiro api 
        """

        self.valid_retrieve = False
        
        self.statements.retrieve_financials()

        ## forcast
        try :
            self.forcast.retrieve_forcasts()
        except KeyError as e:
            self.logger.warning(f'{self.name} : can not retrieve financial forcast from degiro, {e}')

        self.price_history.retrieve_price_history()
        
        try :
            self.retrieve_values()
        except (
            KeyError, 
                # TypeError
                ) as e:
            raise KeyError(f"{self.name} : error while generating values \n {type(e).__name__} : {e}") from e
        
        self.valid_retrieve = True
        # self.eval_beta()

    def _get_cc_from_wacc(self, wacc: float):
            
        if np.isnan(wacc) :
            return np.nan
        if self.use_fcfe :
            return wacc        
        eq = self.market_cap
        debt = self.statements.net_debt
        if debt < 0:
            eq = self.enterprise_value
            debt = self.statements.total_debt
        tr = self.session_model.taxe_rate
        dc = self.statements.debt_cost

        return (wacc - dc * (1-tr) * debt/(debt + eq)) * (debt + eq)/eq

    def _compute_forcasted_wacc_perpetual(self):

        if np.isnan(self.forcast.forcasted_ebitda_growth)  :
            return

        self.g_delta_forcasted_assumed = self.forcast.forcasted_ebitda_growth - self.assumed_g

        # Perpetual method
        optimized_res = minimize_scalar(self._residual_dcf_on_wacc_perpetual,
                            method= 'bounded', bounds = (self.forcast.forcasted_ebitda_growth, self.forcast.forcasted_ebitda_growth + 6))
        err =  optimized_res.fun
        if err <= TOLERANCE_MINIMIZE :
            self.forcasted_wacc_perpetual = optimized_res.x
        

    def _compute_forcasted_wacc_multiple(self):
        """Compute forcasted wacc from ebitda multiple terminal value
        """
        
        if self.forcast._forcasted_capex_growth is None :
            return

        current_value = self.market_cap if self.use_fcfe else self.enterprise_value

        if self.use_fcfe:
            vt_multiple = max(self.forcast._forcasted_ocfe[-1]* self.price_to_ocfe_bounded,0)
        else :
            forcasted_ebitda = self.forcast._get_forcasted_ebidta()
            vt_multiple = max(forcasted_ebitda[-1]* self.value_to_ebitda_bounded,0)

        fcf = self.forcast.forcasted_fcfe if self.use_fcfe else self.forcast.forcasted_fcff

        arr = np.concatenate([np.array([-current_value]), 
                                fcf[:-1], 
                                np.array([vt_multiple])])

        if np.isnan(arr).any():
            self.logger.error(f'{self.name} : nan fcf values forcasted found {arr}, can not compute forcasted wacc multiple: ')
            return

        self.forcasted_wacc_multiple = npf.irr(arr)

        # Compute target price from market wacc and forcasted fcf
        vt_act = vt_multiple / (1+self.market_wacc)**(self.session_model.nb_year_dcf)
        fcf_act = fcf[:-1] / (1+self.market_wacc)**np.arange(1,self.session_model.nb_year_dcf)
        fcf_act_sum = fcf_act.sum()
        target_current_value = fcf_act_sum + vt_act
        target_market_value = target_current_value
        if not self.use_fcfe :
            target_market_value -= self.statements.net_debt
        self.target_market_price_multiple = target_market_value / self.statements.nb_shares

    def _compute_assumed_g(self, ocf :float, up_bound : float):
        """
        compute g from mean fcf
        """
        if ocf < 0 :
            self.logger.info(f"{self.name} : negative operating cash flow mean, can not compute assumed growth")
            return
        self.assumed_g = minimize_scalar(self._residual_dcf_on_g, args=(ocf,  False),
                            method= 'bounded', bounds = (-1, up_bound)).x

    def _compute_assumed_g_ttm(self, up_bound :float):
        """
        compute g_from_ttm from last fcf
        """

        if  not self.statements.q_cashflow_available :
            return
        if self.statements.fcff_ttm < 0:
            self.logger.info(f"{self.name} : negative TTM free cash flow, can not compute TTM assumed growth")
            return

        self.assumed_g_ttm = minimize_scalar(self._residual_dcf_on_g, args=(self.statements.fcff_ttm,  False),
                                method= 'bounded', bounds = (-1, up_bound)).x

        # # compute g_from_ttm from last incf
        # if self.financial_statements.focf_ttm < 0:
        #     print(f"{self.name} : negative TTM cash flow from income mean, can not compute TTM RDINCF")
        # else :
        #     self.g_incf_ttm = minimize_scalar(_residual_dincf_on_g, args=(self, self.financial_statements.focf_ttm),
        #                     method= 'bounded', bounds = (-1, up_bound)).x

    def compute_share_dcf(self,):
        """
        Evaluate from fundamental financial data and stock value:
        - market assumed growth rate 

        Evaluate from fundamental forcast:
        - target return on capital
        - target price

        
        """

        self.compute_multiples()

        self.logger.info(f'{self.name} : compute dcf values                        ')
        if self.use_fcfe:
            ocf = self.statements.ocfe
        else :
            ocf = self.statements.ocff

        self.market_wacc = self._get_market_wacc()
        up_bound = 2 if self.session_model.use_multiple else self.market_wacc

        if self.session_model.use_multiple and (self.value_to_ebitda_bounded < 0) :
            self.logger.info(f"{self.name} negative terminal price to fcf multiple, can not compute RDCF")
            return


        self._compute_assumed_g(ocf, up_bound= up_bound)
        # self._compute_assumed_g_ttm(up_bound= up_bound)

        self.forcast.compute_forcasted_data()

        self._compute_forcasted_wacc_perpetual()
        self._compute_forcasted_wacc_multiple()

        # compute return on capital from wacc
        self.forcasted_capital_cost_multiple = self._get_cc_from_wacc(self.forcasted_wacc_multiple)
        self.forcasted_capital_cost_perpetual = self._get_cc_from_wacc(self.forcasted_wacc_perpetual)

        # compute taget price from market capital cost
        # self.target_market_value_perpetual = (self._compute_value_perpetual(self.market_wacc)    -  self.net_debt
        #                                       ) / self.nb_shares



    def residual_dcf(self, g :  float, ocf : float, wacc : float, vt : float = None):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        fcf = ocf - self.statements.capex
        nb_year_dcf = self.session_model.nb_year_dcf
        if vt is None:
            if self.session_model.use_multiple :
                terminal_value_multiple = self.price_to_ocfe_bounded if self.use_fcfe else self.value_to_ocff_bounded
                vt = ocf * (1+g)**(nb_year_dcf ) * terminal_value_multiple
            else :
                vt = ocf * (1+g)**(nb_year_dcf ) / (wacc - g)
        vt_act = vt / (1+wacc)**(nb_year_dcf)

        # fcf * sum of a**k for k from 1 to nb_year_dcf 
        fcf_ar = fcf * (1+g) ** np.arange(1,1 + nb_year_dcf)
        fcf_act_sum = fcf_ar.sum()
        value = fcf_act_sum + vt_act

        # Compare value to the market price of the company if use Free Cash Flow to Equity (FCFE) or 
        # Compare value to the total value of the company if use Free Cash Flow to Firm (FCFF) 
        if self.use_fcfe:
            return (value/self.market_cap -1)**2
        return (value/self.enterprise_value -1)**2


    def _compute_value_perpetual(self, wacc : float):
        """
        compute company value regarding its actuated operating cash flows and capital expenditure

        Returns:
            float : company value
        """
        # ocf_g = self.forcasted_ocf_growth

        forcast = self.forcast
        vt_perpetual = forcast.forcasted_fcfe[-1] / (wacc - forcast.forcasted_ebitda_growth)
        
        nb_year_dcf = self.session_model.nb_year_dcf
        vt_act = vt_perpetual / (1+wacc)**(nb_year_dcf)

        # ocf_a = (1+ocf_g)/(1+wacc)
        #               fcf * sum of a**k for k from 1 to nb_year_dcf
        # fcf_act_sum = self.ocf * ((ocf_a**nb_year_dcf - 1)/(ocf_a-1) - 1 + ocf_a**(nb_year_dcf))
        # fcf_act_sum -= self.capex * ((capex_a**nb_year_dcf - 1)/(capex_a-1) - 1 + capex_a**(nb_year_dcf))
        fcf = self.forcast.forcasted_fcfe if self.use_fcfe else self.forcast.forcasted_fcff
        focf_act = fcf[:-1] / (1+wacc)**np.arange(1,nb_year_dcf)
        fcf_act_sum = (focf_act).sum()
        enterprise_value = fcf_act_sum + vt_act
        return enterprise_value
    
    def _residual_dcf_on_wacc_perpetual(self, wacc : float ):
        """
        compute company value regarding its actuated operating cash flows and capital expenditure
        compare it to the market value of the company

        Returns:
            square error between enterprise actuated value corresponds 
                    to the one assumed by the market price.
        """

        value = self._compute_value_perpetual(wacc)
        if not self.use_fcfe:
            return (value/self.enterprise_value -1)**2
        return (value/self.market_cap -1)**2
            

    def _residual_dcf_on_g(self, g, *data):
        """
        reformated Share.eval_dcf() function for compatibility with minimize_scalar
        """
        ocf : float = data[0]

        return self.residual_dcf(g = g,
                            ocf = ocf,
                            wacc = self.market_wacc,
                            )

