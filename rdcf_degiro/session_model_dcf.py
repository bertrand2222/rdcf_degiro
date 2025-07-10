import os, json, time
from dataclasses import dataclass
import requests
from lxml import html
import pandas as pd
import yahooquery as yq
from dateutil.relativedelta import relativedelta
from degiro_connector.trading.api import API
from degiro_connector.quotecast.tools.chart_fetcher import ChartFetcher
from degiro_connector.trading.models.credentials import build_credentials
from curl_cffi import requests


ERROR_SYMBOL = 1
MARKET_CURRENCY = "USD"
NO_HISTORY_YEAR = 5 # nb year history

BROWSER_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0'}

### get fed fund rate (debt cost)
URL_FED_FUND_RATE = "https://ycharts.com/indicators/effective_federal_funds_rate"
XPATH_FED_FUND_RATE = "/html/body/main/div/div[4]/div/div/div/div/div[2]/div[1]/div[3]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]"


@dataclass
class RateInfos():
    debt_cost : float = None
    free_risk_rate : float = None
    market_rate : float = None
    
    def retrieve(self) :
        """
        Querry market info from web sources and yahoo finance api
        """
        print("querry market info from degiro and yahoo finance api")
        r = requests.get(URL_FED_FUND_RATE, verify= False, headers= BROWSER_HEADERS, timeout= 20)
        text_fed_fund_rate = html.fromstring(r.content).xpath(XPATH_FED_FUND_RATE)[0].text

        ### get debt cost
        self.debt_cost = float(text_fed_fund_rate.strip("%")) / 100
        print(f"debt cost = {self.debt_cost*100}%")

        ### get free risk rate
        #us treasury ten years yield
        self.free_risk_rate = yq.Ticker("^TNX", asynchronous=True).history(period = '1y', ).loc["^TNX"]["close"].iloc[-1]/100
        print(f"free risk rate = {self.free_risk_rate*100:.2f}%")

        ### eval market rate
        sptr_6y = yq.Ticker("^SP500TR", asynchronous=True).history(period = '6y', interval= "1mo").loc["^SP500TR"]
        sptr_6y_rm = sptr_6y.rolling(2).mean()
        # sptr = yq.Ticker("^SP500TR").history(period = '5y', interval= "1mo").loc["^SP500TR"]

        last_date = sptr_6y.index[-2]
        time_delta = NO_HISTORY_YEAR
        past_date = last_date - relativedelta(years = time_delta)
        # sptr = sptr_6y.loc[past_date:]
        sptr_rm = sptr_6y_rm.loc[past_date:]

        # delta = relativedelta(last_date, past_date)
        # timeDelta = delta.years + delta.months / 12 + delta.days / 365
        self.market_rate =  (sptr_rm.loc[last_date]['close'] / sptr_rm.loc[past_date]['close'])**(1/time_delta) - 1 # S&P 500 mean year rate over 5 years
        print(f"market rate = {self.market_rate*100:.2f}%")

    
    def save(self,path:str):
        with open(os.path.join(path),
                        "w", 
                        encoding= "utf8") as outfile:
            json.dump(self.__dict__, outfile, indent = 4)

    def read(self,path: str):
        with open(os.path.join(path),
                        "r", 
                        encoding= "utf8") as readfile:
            self.__dict__.update(json.load( readfile))


class SessionModelDCF(API):

    """
    Object containing all global data
    """
    credential_file_path : str = None
    use_beta = False
    use_multiple = True
    history_avg_nb_year : int = 3
    nb_year_dcf : int = 10
    use_last_intraday_price : bool = False
    terminal_price_to_fcf_bounds = [1, 100]
    output_folder = os.getenv("TEMP")
    taxe_rate = 0.25
    output_name = "rdcf"
    yahoo_symbol_cor = None
    retrieve_shares_from_favorites = True
    retrieve_shares_from_portfolio = True
    update_market_rate = False
    update_statements = False
    rate_history_dic = {}
    rate_current_dic = {}
    chart_fetcher : ChartFetcher = None
    nb_days_update : int = 30
    current_timestamp = time.time()
    
    def __init__(self, config_dict : dict):

        self.__dict__.update(config_dict)
        self.config_dict = self.__dict__.copy()

       # Connexion
        print("connect to degiro trading API")

        if "credential_file_path" not in config_dict :
            raise KeyError("Missing credential_file_path definition in input")
        with open(self.credential_file_path, encoding= "utf8") as config_file:
            config_dict = json.load(config_file)

        user_token = config_dict.get("user_token")
        self.chart_fetcher = ChartFetcher(user_token=user_token)

        credentials = build_credentials(location=self.credential_file_path )
        super().__init__(credentials = credentials )
        
        self.rate_info = RateInfos()
        market_info_path = os.path.join(self.output_folder,"market_info.json")
        if self.update_market_rate_need(market_info_path):
            self.rate_info.retrieve()
            self.rate_info.save(market_info_path)
        else:
            self.rate_info.read(market_info_path)

        if not os.path.isdir(self.output_folder):
            raise FileNotFoundError(f"The specified output folder does not exist {self.output_folder}")
        
        self.connect()
    
    def update_rate_dic(self, currency_1, currency_2, ):
        """
        Record change rate hystory and curent value for one currency pair
        """
        if currency_1 == currency_2 :
            return
        rate_symb = currency_1 + currency_2 + "=X"
        if rate_symb not in self.rate_history_dic :
            rate_path = os.path.join(self.output_folder,f'{rate_symb}_history.pckl')
            if self.update_statements or( not os.path.isfile(rate_path)):
                try :
                    
                    currency_history = yq.Ticker(rate_symb, asynchronous=True, session=self.request_session).history(period= '6y',
                                                                    interval= "1mo", 
                                                                    ).loc[rate_symb]
                except KeyError as e:
                    raise KeyError(f'rate symbol {rate_symb} not found in yahoofinance database') from e
                
                currency_history.to_pickle(rate_path)
            else :
                currency_history = pd.read_pickle(rate_path)

            self.rate_history_dic[rate_symb] = currency_history[["close"]].iloc[:-1].rename(
                columns = {'close' : 'change_rate'}).reindex(
                    pd.to_datetime(currency_history.index[:-1]))
            self.rate_current_dic[rate_symb] = currency_history["close"].iloc[-1]

    def update_statements_need(self, path: str) -> bool :
        """
        Checks if statements file need to be updated
        Args:
            path (str): path of file.
        Returns:
            bool: answer
        """
        if self.update_statements:
            return True
        return self.update_file_date_need(path)
    
    def update_market_rate_need(self, path : str) -> bool:
        """
        Checks if market date file need to be updated.

        Args:
            path (str): path of file.
        Returns:
            bool: answer
        """
        if self.update_market_rate:
            return True
        return self.update_file_date_need(path)
    
    def update_file_date_need(self, path : str) -> bool :
        """
        Checks if file is old enough to be updated
        Args:
            path (str): path of file.
        Returns:
            bool: answer
        """
        if not os.path.isfile(path):
            return True
        
        days_old =  (self.current_timestamp - os.path.getmtime(path)) / (3600 * 24)
        if days_old > self.nb_days_update:
            return True
        return False