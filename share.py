import numpy as np
import pandas as pd
import yahooquery as yq
from scipy.optimize import minimize_scalar
import polars as pl
from pydantic import BaseModel
from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from session_model_dcf import SessionModelDCF, MARKET_CURRENCY, ERROR_SYMBOL
from datetime import datetime

ERROR_QUERRY_PRICE = 2

class ShareComputedInfos():

    """
    Cumputed infos
    """

    def __init__(self) :
        self.fcf : float = None
        self.fcf_last : float = None
        self.cmpc :float = None
        self.mean_g_fcf : float = None
        self.net_debt : float = None
        self.y_financial_data : pd.DataFrame = None
        self.financial_currency : str = None
        self.price_to_fcf :float = None
        self.per : float = np.nan
        self.debt_to_equity : float = np.nan
        self.price_to_book : float = np.nan
        self.mean_g_tr = None
        self.mean_g_netinc = None
        self.capital_cost : float = None
        self.roic :float = np.nan
        
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
        self.capital_cost_equal_market = False
        self.total_payout_ratio = None


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
        self.beta : float = None
        self.name : str = None
        self.isin :str = None
        self.close_price : float = None
        self.product_type :str = None
        self.vwd_id : str = None
        self.current_price : float = None
        self.__dict__.update(s_dict)

        self.session_model : SessionModelDCF = session_model
        self.market_infos = session_model.market_infos
        self.computed_infos = ShareComputedInfos()
    
    def retrieve_company_profile(self):
        
        
        company_profile = self.session_model.trading_api.get_company_profile(
        product_isin= self.isin,
        raw=True,
        )
        self.computed_infos.market_cap = float(company_profile['data']['shrOutstanding']) * self.current_price

        # import json
        # with open("company_profile.json", "w", encoding= "utf8") as outfile: 
        #     json.dump(company_profile['data'], outfile, indent = 4)

    def retrieve_history(self) -> None :

        """
        Retrieve share history price dataframe
        """
        chart_request = ChartRequest(
        culture="fr-FR",
     
        period=Interval.P5Y,
        requestid="1",
        resolution=Interval.P1M,
        series=[
            # "issueid:360148977",
            # "price:issueid:360148977",
            "ohlc:issueid:" + self.vwd_id,
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
            df.columns = ['month', 'open', 'high', 'low', 'close']

            self.current_price = df['close'][-1]
            self.history = df['month', 'close']
            self.history_expires = chart.series[0].expires
            
        
    def eval_beta(self) :
        """
        Compute the share beta value regarding the evolution of share price with reference market
        """
        regular_history = self.history['adjclose'].iloc[:-1].copy()

        if self.price_currency != MARKET_CURRENCY :
            self.market_infos.update_rate_dic(self.price_currency, MARKET_CURRENCY)
            change_rate = self.price_currency + MARKET_CURRENCY + "=X"
            regular_history = regular_history * self.market_infos.rate_history_dic[change_rate]

        month_change_share = regular_history.pct_change().rename("share")

        cov_df = pd.concat([self.market_infos.month_change_rate,
                            month_change_share], axis = 1, join= 'inner',).dropna(how = 'any')
        cov = cov_df.cov()['rm'].loc['share']
        beta = cov/ self.market_infos.var_rm
        self.beta = beta

        return beta
    

    def querry_financial_info(self, pr = False):
        """
        Get the share associated financial infos from degiro api 
        """
        
        print(f'\rquerry {self.name}    ', flush=True, end="")
  
        self.retrieve_history()
        self.retrieve_company_profile()

        
        self.y_financial_data : pd.DataFrame = tk.get_financial_data(TYPES,)
        self.financial_currency = self.y_financial_data.dropna(subset="FreeCashFlow")["currencyCode"].value_counts().idxmax()
        self.y_financial_data = self.y_financial_data.sort_values(by = ['asOfDate' , 'BasicAverageShares'], ascending=[1, 0])

        self.y_financial_data['BasicAverageShares'] = self.y_financial_data['BasicAverageShares'].ffill(axis = 0, )
        self.y_financial_data = self.y_financial_data.loc[self.y_financial_data['currencyCode'] == self.financial_currency ]

        self.unknown_last_fcf = np.isnan(self.y_financial_data["FreeCashFlow"].iloc[-1])
        
        self.y_financial_data = self.y_financial_data.ffill(axis = 0
                                                            ).drop_duplicates(
                                                                subset = ['asOfDate', ],
                                                                keep= 'last').set_index('asOfDate')


        self.q_financial_data = tk.get_financial_data(TYPES , frequency= "q",
                                                      trailing= False).set_index('asOfDate')
        print(self.q_financial_data)
        self.q_financial_data = self.q_financial_data.loc[self.q_financial_data['currencyCode'] == self.financial_currency ]

        if isinstance(self.q_financial_data, pd.DataFrame) :
            self.q_financial_data.ffill(axis = 0, inplace = True)
            last_date_y = self.y_financial_data.index[-1]
            last_date_q = self.q_financial_data.index[-1]

            for d in BALANCE_INFOS:
                if d in self.q_financial_data.columns :
                    self.y_financial_data.loc[last_date_y , d] = self.q_financial_data.loc[last_date_q , d]

        self.nb_shares = self.y_financial_data["BasicAverageShares"].iloc[-1]
        # if self.market_cap is None :
            # self.market_cap = self.nb_shares * self.close_price

        # self.financial_currency = self.y_financial_data['currencyCode'].iloc[-1]
        self.eval_beta()

        self.financial_currency_price_history = self.history[['close']].iloc[:-1].copy()
        if self.price_currency != self.financial_currency:
            self.market_infos.update_rate_dic(self.price_currency, self.financial_currency)
            rate_symb = self.price_currency + self.financial_currency + "=X"
   
            rate = self.market_infos.rate_current_dic[rate_symb]
            self.close_price *= rate
            self.market_cap *= rate
            self.financial_currency_price_history = self.financial_currency_price_history.mul(self.market_infos.rate_history_dic[rate_symb],axis = 0)
            

        if self.compute_financial_info(pr = pr) :
            return 1
        return 0

    def compute_financial_info(self, pr = True,) :
        """
        Compute intermedate financial infos used as input for dcf calculation
        """
        free_risk_rate = self.market_infos.free_risk_rate
        market_rate = self.market_infos.market_rate
        y_financial_data = self.y_financial_data
        q_financial_data =  self.q_financial_data

        last_financial_info =  y_financial_data.iloc[-1]

        # self.nb_shares = int(self.market_cap / self.close_price)
        stock_equity = last_financial_info['CommonStockEquity']
        market_cap = self.market_cap

        if self.capital_cost_equal_market :
            self.capital_cost = market_rate
        else : 
            self.capital_cost = free_risk_rate + self.beta * (market_rate - free_risk_rate)
        
        try :
            total_debt = last_financial_info['TotalDebt']
        except KeyError:
            print(f"no total debt available for {self.name}")
            return 1
        self.cmpc = self.capital_cost * stock_equity/(total_debt + stock_equity) + \
                    self.market_infos.debt_cost * (1-IS) * total_debt/(total_debt + stock_equity)
        self.net_debt = total_debt - last_financial_info['CashAndCashEquivalents']


        self.net_market_cap = market_cap + self.net_debt
        if stock_equity >= 0 :
            self.debt_to_equity = total_debt / stock_equity
            self.price_to_book = market_cap / stock_equity

        y_financial_data['date'] = y_financial_data.index
        y_financial_data['year'] = y_financial_data['date'].apply(lambda  x : x.year )
        y_financial_data.drop_duplicates(subset= "year", keep= "last", inplace= True)

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
            (self.q_financial_data.index[-1] > y_financial_data.index[-2]) :

            complement_q_financial_infos = self.q_financial_data[self.q_financial_data.index
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

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        """
        Get price corresponding to given growth rate with discouted
        cash flow methods
        """
        if self.cmpc is None :
            self.querry_financial_info(pr = pr)
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