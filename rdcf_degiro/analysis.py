from string import ascii_uppercase
import subprocess
from typing import List
import warnings
from curl_cffi import CurlError
import urllib3
import pandas as pd
import pickle
# from importlib import reload
# from colorama import Fore
from degiro_connector.trading.models.account import UpdateOption, UpdateRequest
from rdcf_degiro.forcast import InconsistentStatementsError
from rdcf_degiro.share import MarketCapError, OptiError, Share
from rdcf_degiro.session_model_dcf import SessionModelDCF
from rdcf_degiro.financial_statements import MissingDataError, YahooRetrieveError
from rdcf_degiro.price_history import PriceRetrieveError
warnings.simplefilter(action='ignore', category=FutureWarning)

urllib3.disable_warnings()

from requests.exceptions import RetryError


# import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from googleapiclient.http import MediaFileUpload
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow

# SCOPES =  ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

PKL_NAME = "df_save.pkl"
IS = 0.25

class RDCFSummary():    

    def __init__(self, df: pd.DataFrame, session_model: SessionModelDCF, sector_info: pd.Series = None, name : str = None):
        
        self.df = df
        self.config = pd.DataFrame.from_dict({**session_model.config_dict, 
                                              'market_rate' : session_model.rate_info.market_rate,
                                              'risk_free_rate' : session_model.rate_info.risk_free_rate}, 
                                                    orient= 'index').iloc[:,0]
        self.name = name
        self.sector_info = sector_info

    def save(self, path):
        with open(path, "wb") as f:
            try:
                pickle.dump(self, f)
            except OSError as e:
                raise OSError(f"Error while recording summary {e}  ") from e

    @staticmethod
    def load(path : str):
        """
        Loads previously saved analysis dataframe
        """
        with open(path, "rb") as fichier:
            obj : RDCFSummary = pickle.load(fichier)
        return obj
    
    def to_excel(self, writer:  pd.ExcelWriter, sheet_name :str ):

        """
        Export in Excel format the analysis dataframe 
        """


        letters = list(ascii_uppercase) + list(['A' + l for l in ascii_uppercase]) + list(['B' + l for l in ascii_uppercase])
        if self.df is None:
            return
        df = self.df
        col_letter = {c : letters[i+1] for i, c in enumerate(df.columns)}
        df.to_excel(writer, sheet_name= sheet_name)
        wb = writer.book
        number = wb.add_format({'num_format': '0.00'})
        price = wb.add_format({'num_format': '###,##0.00'})
        percent = wb.add_format({'num_format': '0.00%'})
        bold_percent = wb.add_format({'bold': True, 'num_format': '0.00%'})
        l_align =  wb.add_format()
        l_align.set_align('left')
        # Add a format. Light red fill with dark red text.
        # format1 = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        # Add a format. Green fill with dark green text.
        # format2 = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        # Add a format. Light red fill .
        format3 = wb.add_format({"bg_color": "#F8696B",})

        worksheet = writer.sheets[sheet_name]

        format_dict = {
            "current_price" : {"size" : 13, "format" : price},
            "unlevered_beta" : {"size" : 0, "format" : number},
            "levered_beta" : {"size" : 0, "format" : number},
            "market_cap_usd" : {"size" : 11, "format" : price},
            "value_to_ebitda_adj" : {"size" : 0, "format" : number},
            "wacc" : {"size" : 11, "format" : percent},
            "assumed_g" : {"size" : 11, "format" : bold_percent},
            "history_growth" : {"size" : 0, "format" : percent},
            "forcasted_ebitda_growth" : {"size" : 11, "format" : percent},
            "target_market_price_multiple" : {"size" : 11, "format" : price},
            "potential_price_evolution" : {"size" : 11, "format" : bold_percent},
            # "target_market_price_perpetual" : {"size" : 11, "format" : number},
            "price_to_sales" : {"size" : 11, "format" : number},
            "debt_to_equity" : {"size" : 0, "format" : number},

                }
        for k, f in format_dict.items():
            worksheet.set_column(f"{col_letter[k]}:{col_letter[k]}",
                                            f['size'], 
                                            f['format'])
        # add hyperlink
        for i, s in enumerate(df.index):
            yahoo_symb = df.loc[s, 'yahoo_symb']
            if isinstance(yahoo_symb, str) and yahoo_symb != "":
                link = fr'https://finance.yahoo.com/quote/{yahoo_symb}/'
            else :
                link = fr'https://www.tradingview.com/symbols/{s}/'
            worksheet.write_url(f'B{i+2}', 
                                link, 
                                string= df.loc[s, 'name'] )

        worksheet.add_table(0,0,len(df.index),len(df.columns) ,
                            {"columns" : [{'header' : 'symbol'}]
                                + [{'header' : col} for col in df.columns],
                            'style' : 'Table Style Light 8'})
        worksheet.set_column('B:B', 30, )
        worksheet.set_column(f"{col_letter['forcasted_wacc_multiple']}:{col_letter['forcasted_wacc_perpetual']}",
                                                        0, #11, 
                                                        percent)
        worksheet.set_column(f"{col_letter['expected_return_multiple']}:{col_letter['beta_ajusted_expected_return_multiple']}",
                                11, bold_percent)
        worksheet.set_column(f"{col_letter['total_payout_ratio']}:{col_letter['total_payout_ratio']}",
                                11, percent )
        worksheet.set_column(f"{col_letter['roic']}:{col_letter['roic']}", 0, percent )
        # worksheet.set_column(f"{col_letter['mean_g_fcf']}:{col_letter['diff_g']}", 13, percent )

        def format_max_min_green_red(ws, col_s : str, col_e : str = None, 
                                        max_type : str = 'max', 
                                        mid_type = 'percentile',
                                        mid_value = 0,
                                        **kwargs):
            if col_e is None:
                col_e = col_s
            format_dic = {"type": "3_color_scale", 'min_type': 'min',
                'max_type': max_type, 'mid_type' : mid_type,
                'min_color' : '#F8696B', "max_color" : '#63BE7B', 
                "mid_color" : "#FFFFFF", "mid_value" : mid_value,
                **kwargs}
            ws.conditional_format(
                f"{col_letter[col_s]}2:{col_letter[col_e]}{len(df.index)+1}", format_dic
                )

    
        format_max_min_green_red(worksheet, 'history_growth',) 
        format_max_min_green_red(worksheet, 'forcasted_ebitda_growth')
        # format_max_min_green_red(worksheet, 'diff_g_forcasted_assumed')
        # format_max_min_green_red(worksheet, 'forcasted_wacc_multiple' , max_type='num', max_value= 1)
        # format_max_min_green_red(worksheet, 'forcasted_wacc_perpetual' , max_type='num', max_value= 1)
        format_max_min_green_red(worksheet, 'expected_return_multiple', max_type='num', max_value= 1, mid_type= "num", mid_value= self.config.loc['market_rate'])
        format_max_min_green_red(worksheet, 'expected_return_perpetual', max_type='num', max_value= 1, mid_type= "num", mid_value= self.config.loc['market_rate'])
        format_max_min_green_red(worksheet, 'potential_price_evolution', max_type='num',  max_value = 2, mid_type= "num", mid_value= 0, min_type='num', min_value= -0.1)
        format_max_min_green_red(worksheet, 'beta_ajusted_expected_return_multiple', max_type='num', max_value= 1, mid_type= "num", mid_value= 0)
        format_max_min_green_red(worksheet, 'roic', max_type='num', max_value= 1, mid_type= "num", mid_value= 0.08, min_type='num', min_value= 0)


        # Format debt to equity
        worksheet.conditional_format(
            f"{col_letter['debt_to_equity']}2:{col_letter['debt_to_equity']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num',
                'max_type': 'num', 'mid_type' : 'num',
                'min_value' : 0, 'mid_value' : 1, "max_value" : 2, 
                'min_color' : '#63BE7B', "max_color" : '#F8696B', 
                "mid_color" : "#FFFFFF"})
        # format PER
        worksheet.conditional_format(
            f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}",
            {"type": "cell", "criteria": "<", "value": 0, "format": format3})
        worksheet.conditional_format(
            f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num','max_type': 'num',
                'mid_type' : 'percentile',
                'min_value' : 3, 'mid_value' : 50, "max_value" : 50,
                'min_color' : '#63BE7B', "max_color" : '#F8696B', 
                "mid_color" : "#FFFFFF"})
        worksheet.conditional_format(
                    f"{col_letter['price_to_sales']}2:{col_letter['price_to_sales']}{len(df.index)+1}",
                    {"type": "3_color_scale", 'min_type': 'num','max_type': 'num',
                        'mid_type' : 'percentile',
                        'min_value' : 3, 'mid_value' : 50, "max_value" : 50,
                        'min_color' : '#63BE7B', "max_color" : '#F8696B', 
                        "mid_color" : "#FFFFFF"})
        # format ROE
        # worksheet.conditional_format(f"{col_letter['roe']}2:{col_letter['roe']}{len(df.index)+1}",
        #                                 {"type": "cell", "criteria": "<",
        #                                 "value": 0, "format": format3})
        # worksheet.conditional_format(f"{col_letter['roe']}2:{col_letter['roe']}{len(df.index)+1}",
        #                             {"type": "3_color_scale", 'min_type': 'num','max_type': 'max',
        #                                 'mid_type' : 'percentile',
        #                             'min_value' : 0, 'mid_value' : 50, "max_value" : 0.15,
        #                             "min_color" : '#F8696B', 'max_color' : '#63BE7B' ,
        #                             "mid_color" : "#FFFFFF"})
   

  
                
        ##### save config
        
        
        self.config.to_excel(writer, sheet_name= sheet_name+"_config")
        worksheet = writer.sheets[sheet_name+"_config"]
        worksheet.set_column('A:A', 30, l_align)
        
        self.sector_info.to_excel(writer, sheet_name= "sector_info")
        worksheet = writer.sheets["sector_info"]
        worksheet.set_column('A:A', 20, )
        worksheet.set_column(f"B:C",
                                20, 
                                number)

        
class RDCFAnal():
    """
    object containing a reverse dcf analysis and all its context
    """

    def __init__(self,
                 config_dict : dict = None,
             ) -> None:

        self.df : pd.DataFrame = None
        self.share_list : List[Share] = []
        # self.ids : List[int] = None
        self.__dict__.update(config_dict)

        self.session_model = SessionModelDCF(config_dict)
        self.logger = self.session_model.logger
        self.degiro_api = None


    def connect_degiro_api(self):

        if self.degiro_api is None:

            self.session_model.connect_degiro_api()
            self.degiro_api = self.session_model.degiro_api
            self.get_client_details_table = self.session_model.degiro_api.get_client_details

    def retrieve_shares_from_favorites(self):
        """
        Retrieve stock and fund ids recorded in Degiro account favorite list
        """
        favorite_batch = self.degiro_api.get_favorite(raw=False)
        ids = favorite_batch.data[0].product_ids

        # FETCH PRODUCT INFO
        product_info = self.degiro_api.get_products_info(
            product_list= ids,
            raw=False,
        )

        self.share_list = []
        for s_id, s_dict in product_info.data.items():
            if s_dict.product_type =='STOCK':
                try :
                    self.share_list.append(Share( s_dict.__dict__ ,
                                 session_model = self.session_model,
                                 )
                                 )
                except YahooRetrieveError as e:
                    self.logger.warning(f'{s_dict.name} : can not retrieve yahoo info, {e}     ')
        
        
    # def retrieve_shares_from_portfolio(self):
    #     """
    #     Retrieve stock and fund ids recorded in Degiro account portfolio
    #     """
        
    #     account_update = self.degiro_api.get_update(
    #     request_list=[
    #         UpdateRequest(
    #             option=UpdateOption.PORTFOLIO,
    #             last_updated=0,
    #         ),
    #     ],
    #     raw=False,
    # )
    #     ids = []


    #     for p in account_update.portfolio['value']:
    #         if is_int(p['id']) :
    #             # select only portfolio value wich position size != 0
    #             for value in p['value']:
    #                 if value['name']  == "size" and value['value'] != 0:
    #                     ids.append(int(p['id']))
    #                     break
        
    #     # FETCH PRODUCT INFO
    #     product_info = self.degiro_api.get_products_info(
    #         product_list= ids,
    #         raw=False,
    #     )

    #     self.share_list += [Share( s_dict.__dict__ ,
    #                              session_model = self.session_model,
    #                              )
    #                              for s_id, s_dict in product_info.data.items() if s_dict.product_type =='STOCK'
    #                              ]

    def retrieve_data(self ) :

        """
        Generate an analysis summary dataframe
        """
        self.connect_degiro_api()

        if self.session_model.retrieve_shares_from_favorites:
            self.retrieve_shares_from_favorites()
        # if self.session_model.retrieve_shares_from_portfolio:
        #     self.retrieve_shares_from_portfolio()

        if not self.share_list:
            self.logger.error('No product to process, retrieve products before')
            return
        
        for s in self.share_list :
            if not s.valid_init:
                continue
            try :
                for retrieve_from in ['degiro', 'yahoo']:
                    s.retrieve_from = retrieve_from
                    try :
                        s.retrieve_share_data()
                        break
                    except MarketCapError as e:
                        self.logger.warning(f"{e} can not retrieve data from {retrieve_from}")

            except (PriceRetrieveError, YahooRetrieveError, 
                    MissingDataError,
                    KeyError,
                    CurlError,
                    RetryError,
                    IndexError,
                    ) as e:
                s.error_msg += f"Error while retrieving data {type(e).__name__} : {e}   "
                self.logger.error(f"{s.name} : {s.error_msg}")
                continue


    def process(self) :

        valid_share_list = [s for s in self.share_list if s.valid_retrieve]
        if not valid_share_list :
            print('no share whith valid data, run retrieve_data() first')
            return

        # Compute unlevered beta and levered beta for each share, based on sector median unlevered beta
        info_df = pd.DataFrame.from_dict({s.symbol : {
                                        'unlevered_beta' : s.price_history.unlevered_beta,
                                        # 'market_cap_usd' : s.market_cap_usd,
                                        'value_to_ebitda' : s.value_to_ebitda,
                                        'sector' : s.sector,
                                        } for s in valid_share_list},
                                         orient= 'index')

        sector_info = info_df.groupby('sector').median()
        # sector_info = info_df.groupby('sector').apply(lambda x: x['unlevered_beta'].median())


        for s in valid_share_list :
            s.sector_unlevered_beta = sector_info.loc[s.sector, 'unlevered_beta']
            sector_value_to_ebitda = sector_info.loc[s.sector, 'value_to_ebitda']
            s.levered_beta = s.sector_unlevered_beta * (1 + (1 - self.session_model.taxe_rate) * s.debt_to_equity)

            # Ajust value to ebitda with sector
            self_weight = 0 if (s.value_to_ebitda < 0) else s.share_params.get("multiple_self_weight",2/3)
            s.value_to_ebitda_adj =  1 / (self_weight/s.value_to_ebitda + (1-self_weight)/sector_value_to_ebitda)

            try :
                s.compute_share_dcf()
            except (KeyError,
                     TypeError, 
                    IndexError,
                    InconsistentStatementsError
                    ) as e:
                s.error_msg += f"Error while computing dcf {type(e).__name__} : {e}   "
                self.logger.error(f"{s.name} : {s.error_msg}")
                continue

        self.logger.info("generate summary table")
        
        attributes = [
                    'symbol',
                    'name', 
                      'current_price', 
                      'price_currency', 'unlevered_beta','levered_beta', 
                      'value_to_ebitda_adj',
                      'wacc', 'assumed_g', 'history_growth', 'forcasted_ebitda_growth', 
                      'forcasted_wacc_multiple', 
                      'forcasted_wacc_perpetual', 
                      'expected_return_multiple', 
                      'expected_return_perpetual', 
                      'expected_return_multiple' , 
                      'expected_return_perpetual', 
                      'beta_ajusted_expected_return_multiple',
                      'target_market_price_multiple',
                      'potential_price_evolution',
                      'per', 'price_to_sales', 
                    #   'roe', 
                      'roic', 'debt_to_equity', 
                      'total_payout_ratio', 
                      'market_cap_usd', 
                      "yahoo_symb",
                      "sector",
                      'error_msg']
        
        data = [ {attr : getattr(s, attr, 
                            getattr(s.statements, attr, # get attr from statements if not in share
                                getattr(s.price_history, attr, # get attr from price_history if not in statements
                                    getattr(s.forcast, attr,None)) # get attr from forcast if not in statements
                                                             ) 
                                 ) for attr in attributes} for s in self.share_list 
                                #  if s.valid_init
                                 ]
        df = pd.DataFrame.from_records(index = "symbol", data= data)
        
        df.sort_values(by = ['potential_price_evolution',]  , inplace= True, ascending= False)

        self.degiro_api.logout()


        summary = RDCFSummary(df, self.session_model, 
                              sector_info= sector_info, 
                              name= self.session_model.config_dict.get('name', None))
        return summary



def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False