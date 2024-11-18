import os, json
import requests
from lxml import html
import pandas as pd
import yahooquery as yq
from dateutil.relativedelta import relativedelta
from degiro_connector.trading.api import API
from degiro_connector.quotecast.tools.chart_fetcher import ChartFetcher
from degiro_connector.trading.models.credentials import build_credentials

ERROR_SYMBOL = 1
MARKET_CURRENCY = "USD"
HISTORY_TIME = 5 # nb year history

BROWSER_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0'}

### get fed fund rate (debt cost)
URL_FED_FUND_RATE = "https://ycharts.com/indicators/effective_federal_funds_rate"
XPATH_FED_FUND_RATE = "/html/body/main/div/div[4]/div/div/div/div/div[2]/div[1]/div[3]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]"

class MarketInfos():
    """
    Global market informations
    """
    
    def __init__(self) -> None:

        self.debt_cost : float = None
        self.free_risk_rate : float = None
        self.market_rate : float = None
        self.month_change_rate : pd.DataFrame = None
        self.var_rm : float = None

        self.rate_history_dic = {}
        self.rate_current_dic = {}

        self.update()
    
    def update(self) :
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
        self.free_risk_rate = yq.Ticker("^TNX").history(period = '1y', ).loc["^TNX"]["close"].iloc[-1]/100
        print(f"free risk rate = {self.free_risk_rate*100:.2f}%")

        ### eval market rate
        sptr_6y = yq.Ticker("^SP500TR").history(period = '6y', interval= "1mo").loc["^SP500TR"]
        sptr_6y_rm = sptr_6y.rolling(2).mean()
        # sptr = yq.Ticker("^SP500TR").history(period = '5y', interval= "1mo").loc["^SP500TR"]

        last_date = sptr_6y.index[-2]
        time_delta = HISTORY_TIME
        past_date = last_date - relativedelta(years = time_delta)
        sptr = sptr_6y.loc[past_date:]
        sptr_rm = sptr_6y_rm.loc[past_date:]

        # delta = relativedelta(last_date, past_date)
        # timeDelta = delta.years + delta.months / 12 + delta.days / 365
        self.market_rate =  (sptr_rm.loc[last_date]['close'] / sptr_rm.loc[past_date]['close'])**(1/time_delta) - 1 # S&P 500 mean year rate over 5 years
        print(f"market rate = {self.market_rate*100:.2f}%")

        self.month_change_rate = sptr["adjclose"][:-1].pct_change(periods = 1).rename('rm')
        self.var_rm = self.month_change_rate.var()


    def update_rate_dic(self, currency_1, currency_2, ):
        """
        Record change rate hystory and curent value for one currency pair
        """
        if currency_1 == currency_2 :
            return
        rate_symb = currency_1 + currency_2 + "=X"
        if rate_symb not in self.rate_history_dic :
            try :
                currency_history = yq.Ticker(rate_symb).history(period= '5y',
                                                                interval= "1mo", 
                                                                ).loc[rate_symb]
            except KeyError as e:
                raise KeyError(f'rate symbol {rate_symb} not found in yahoofinance database') from e
            
            self.rate_history_dic[rate_symb] = currency_history[["close"]].iloc[:-1].rename(
                columns = {'close' : 'change_rate'}).reindex(
                    pd.to_datetime(currency_history.index[:-1]))
            self.rate_current_dic[rate_symb] = currency_history["close"].iloc[-1]


class SessionModelDCF():

    """
    Oobject containing all global data
    """
    
    credential_file_path : str = None
    capital_cost_equal_market = False
    use_multiple = True
    price_to_fcf_avg_method = 'harmonic'
    history_avg_nb_year : int = 3
    nb_year_dcf : int = 10
    output_value_files : bool = False
    use_last_intraday_price : bool = False
    terminal_price_to_fcf_bounds = [1, 100]
    output_folder = os.getenv("TEMP")
    taxe_rate = 0.25
    output_name = "rdcf"
    
    def __init__(self, config_dict : dict):

        self.__dict__.update(config_dict)
        self.config_dict = self.__dict__.copy()

        self.market_infos : MarketInfos = None
        self.chart_fetcher : ChartFetcher = None
        self.trading_api : API = None

        if not os.path.isdir(self.output_folder):
            raise FileNotFoundError(f"The specified output folder does not exist {self.output_folder}")
        
        if "credential_file_path" not in config_dict :
            raise KeyError("Missing credential_file_path definition in input")
        

        self.connect()

    def connect(self) :
        """
        Connexion
        """
        print("connect to degiro trading API")

        with open(self.credential_file_path, encoding= "utf8") as config_file:
            config_dict = json.load(config_file)

        user_token = config_dict.get("user_token")
        self.chart_fetcher = ChartFetcher(user_token=user_token)

        credentials = build_credentials(location=self.credential_file_path )
        self.trading_api = API(credentials = credentials )
        self.trading_api.connect()

