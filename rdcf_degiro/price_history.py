from abc import abstractmethod
import os
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
# from degiro_connector.quotecast.models.chart import ChartRequest, Interval
from datetime import  timedelta, datetime

from rdcf_degiro.financial_statements import CurrencyConvertrer
from rdcf_degiro.session_model_dcf import SessionModelDCF


def last_day_of_month(any_day: datetime):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

class PriceRetrieveError(Exception):
    pass

class PriceHistory:

    def __init__(self, session_model : SessionModelDCF, share_params : dict):
        self.session_model = session_model
        self.share_params = share_params

        self.logger = self.session_model.logger
        self.symbol : str = share_params.get("symbol")
        self.name : str  = share_params.get("name")
        self.price_currency : str = share_params.get("price_currency")
        self.yahoo_tk : yf.Ticker = share_params.get("yahoo_tk")
        self.price_series_str : str = share_params.get("price_series_str")

        self.history_in_financial_currency : pd.DataFrame = None
        self.history :pd.DataFrame = None
        self.market_chg : pd.DataFrame = None

        self.currency_converter = CurrencyConvertrer(session_model=session_model, 
                                                             share_params=share_params,
                                                             currency_convertible_cols=['close'])

    @abstractmethod
    def _retrieve_price_history(self) :
        pass

    def retrieve_price_history(self):
        """
        retrieve current history from charts
        """
        
        self.logger.info(f'{self.name} : retrieves price history                      ')
        self._retrieve_price_history()
        self.history_in_financial_currency = self.history.copy()
        self.history_usd = self.currency_converter.convert(self.history, statements_currency=self.price_currency, target_currency= 'USD')
        self.eval_beta()

    def eval_beta(self):
        """
        Compute unlevered beta from share and market history
        """

        cov_df = pd.concat([self.history_usd.reset_index()['close'].pct_change(), 
                            self.market_chg.iloc[-len(self.history):].reset_index()['close']], 
                            axis=1, 
                            keys=['share', 'market']).dropna().cov()
        market_var = cov_df.loc['market', 'market']
        cov = cov_df.loc['share', 'market']
        self.unlevered_beta = cov / market_var


class YahooPriceHistory(PriceHistory):

    def __init__(self, session_model : SessionModelDCF, share_params : dict):
        super().__init__(session_model, share_params)

        self.market_chg = self.session_model.rate_info.market_history_chg
        

    def _retrieve_price_history(self):

        yahoo_hist = self.yahoo_tk.history(period="5y", interval="1mo")[['Close']]

        # y_price_path = os.path.join(self.session_model.output_folder,
        #                                                  f"{self.symbol}_y_price.pckl")
        # try :
        #     yahoo_hist = self.yahoo_tk.history(period="5y", interval="1mo")[['Close']]
        #     yahoo_hist.to_pickle(y_price_path)
        # except RetryError as e:
        #     self.logger.warning(f'{self.name} : failed to update price history from yahoo {e}')
        #     yahoo_hist = pd.read_pickle(y_price_path)

        yahoo_hist = yahoo_hist.rename(columns={'Close': 'close'})
        self.history = yahoo_hist.tz_localize(None)

        self.current_price = yahoo_hist['close'].iloc[-1]


# class DegiroPriceHistory(PriceHistory):

#     def __init__(self, session_model : SessionModelDCF, share_params : dict):
#         super().__init__(session_model, share_params)

#         if (self.session_model.rate_info.degiro_market_history_chg is None):
#             self.session_model.rate_info.degiro_market_history_chg = self._get_price_history("price:issueid:260011946")['close'].pct_change()
#         self.market_chg = self.session_model.rate_info.degiro_market_history_chg

#     def retrieve_intra_day_price(self):
#         """
#         retrieve current price from charts
#         """

#         chart_request = ChartRequest(
#             culture="fr-FR",
#             # culture = "en-US",
        
#             period=Interval.P1D,
#             requestid="1",
#             resolution=Interval.PT5M,        
#             series=[
#                 self.price_series_str,
#             ],
#             tz="Europe/Paris",
#             )
#         chart = self.session_model.degiro_chart_fetcher.get_chart(
#             chart_request=chart_request,
#             raw=False,
#         )
#         try :
#             self.current_price = chart.series[0].data[-1][-1]
#         except IndexError :
#             print(f'{self.name} : warning, not enought data to retrieve intra day price')

#     def _get_price_history(self, serie_str: str) -> pd.DataFrame:
    
#         chart_request = ChartRequest(
#                             culture="fr-FR",
#                             # culture = "en-US",
                        
#                             period=Interval.P5Y,
#                             requestid="1",
#                             resolution=Interval.P1M,
                
#                             series=[
#                                 # "issueid:360148977"+ self.vwd_id ,
#                                 # f"issueid:260011946",
#                                 # self.price_series_str,
#                                 serie_str
#                             ],
#                             tz="Europe/Paris",
#                             )
                
#         chart = self.session_model.degiro_chart_fetcher.get_chart(
#             chart_request=chart_request,
#             raw=False,
#         )

#         if chart is None:
#             raise PriceRetrieveError('error while fetching price history chart')

#         history = pd.DataFrame(data=chart.series[0].data, columns = ['time', 'close'])

#         if self.session_model.use_last_intraday_price:
#             self.retrieve_intra_day_price()

#         last_day_current_month = last_day_of_month(chart.series[0].expires)
#         history['date'] = [last_day_current_month - relativedelta(months= 60 - i) for i in history['time']]

#         return history.set_index('date').tz_localize(None)


#     def _retrieve_price_history(self):
    
#         self.history = self._get_price_history(self.price_series_str)
#         self.current_price = self.history['close'].iloc[-1]


        

