import os
import sys
import re
from string import ascii_uppercase
import subprocess
from typing import List
import warnings
import urllib3
import pandas as pd
# from colorama import Fore
from degiro_connector.trading.api import API
from degiro_connector.trading.models.account import UpdateOption, UpdateRequest
from share import Share
from session_model_dcf import SessionModelDCF, MarketInfos
from importlib import reload

warnings.simplefilter(action='ignore', category=FutureWarning)

urllib3.disable_warnings()


# import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from googleapiclient.http import MediaFileUpload
# from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow

# import pandas as pd
# SCOPES =  ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

PKL_PATH = "df_save.pkl"
IS = 0.25
NB_YEAR_DCF = 10


class RDCFAnal():
    """
    object containing a reverse dcf analysis and all its context
    """
    
    def __init__(self,
                 config_dict : dict = None,
             ) -> None:
        
        reload(pd)
        self.df : pd.DataFrame = None
        self.share_list : List[Share] = []
        self.trading_api : API = None
        # self.ids : List[int] = None
        self.__dict__.update(config_dict)

        self.session_model = SessionModelDCF(config_dict)

        self.trading_api = self.session_model.trading_api
        self.get_client_details_table = self.trading_api.get_client_details
       
     
    def retrieve_shares_from_favorites(self):
        """
        Retrieve stock and fund ids recorded in Degiro account favorite list
        """
        favorite_batch = self.trading_api.get_favorite(raw=False)
        ids = favorite_batch.data[0].product_ids

        # FETCH PRODUCT INFO
        product_info = self.trading_api.get_products_info(
            product_list= ids,
            raw=False,
        )
        
        self.share_list += [Share( s_dict.__dict__ ,
                                 session_model = self.session_model,
                                 )
                                 for s_id, s_dict in product_info.data.items() if s_dict.product_type =='STOCK'
                                 ]
        
    def retrieve_shares_from_portfolio(self):
        """
        Retrieve stock and fund ids recorded in Degiro account portfolio
        """
        
        account_update = self.trading_api.get_update(
        request_list=[
            UpdateRequest(
                option=UpdateOption.PORTFOLIO,
                last_updated=0,
            ),
        ],
        raw=False,
    )
        ids = []


        for p in account_update.portfolio['value']:
            if is_int(p['id']) :
                # select only portfolio value wich position size != 0
                for value in p['value']:
                    if value['name']  == "size" and value['value'] != 0:
                        ids.append(int(p['id']))
                        break
        
        # FETCH PRODUCT INFO
        product_info = self.trading_api.get_products_info(
            product_list= ids,
            raw=False,
        )

        self.share_list += [Share( s_dict.__dict__ ,
                                 session_model = self.session_model,
                                 )
                                 for s_id, s_dict in product_info.data.items() if s_dict.product_type =='STOCK'
                                 ]

    def process(self ) :

        """
        Generate an analysis summary dataframe
        """
        if not self.share_list:
            print('No product to process, retrieve products before')
            return
        
        self.session_model.market_infos = MarketInfos()

        valid_share_list : List[Share] = []
        for s in self.share_list :
            
            print(f'{s.identity.name} : retrieves all values                    ', flush= True, end = "\r")
            try:
                s.retrieves_all_values()
            except (KeyError, TypeError) as e:
                print(f"{s.identity.name} : error while retrieving values {e}           ")
                continue
            
            print(f'{s.identity.name} : compute complementary values            ', flush= True, end='\r')
            try:
                s.compute_complementary_values()
            except (KeyError, TypeError) as e:
                print(f"{s.identity.name} : error while computing complementary values \n   {e}           ")
                continue

            print(f'{s.identity.name} : eval assumed growth                     ', )
            s.eval_g()

            valid_share_list.append(s)

        print("generate summary table")


        df = pd.DataFrame.from_records(index = [s.identity.symbol for s in valid_share_list],
                          data= [
                              {
                                'short_name' :          s.identity.name  ,
                                'current_price' :       s.values.current_price ,
                                'currency' :            s.currency ,
                                'beta' :                s.values.beta ,
                                'price_to_fcf' :        s.values.price_to_fcf,
                                'capital_cost' :        s.values.capital_cost,
                                'cmpc' :                s.values.cmpc ,
                                'assumed_g' :           s.values.g ,  
                                'assumed_g_ttm' :      s.values.g_from_ttm,  
                                'per' :                 s.values.per,
                                'roic' :                s.values.roic , 
                                'debt_to_equity' :      s.values.debt_to_equity,
                                'price_to_book' :       s.values.price_to_book ,
                                'total_payout_ratio' :  s.values.total_payout_ratio,
                                # 'mean_g_fcf':     s.mean_g_fcf ,
                                # 'mean_g_tr' :     s.mean_g_tr,
                                # 'mean_g_inc' :    s.mean_g_netinc 
                                    } for s in valid_share_list])


        self.trading_api.logout()

        df.sort_values(by = ['assumed_g', 'debt_to_equity']  , inplace= True, ascending= True)

        self.df = df
        df.to_pickle(PKL_PATH)

    def load_df(self):
        """
        Loads previously saved analysis dataframe
        """
        self.df = pd.read_pickle(PKL_PATH)
        self.share_list = self.df.index

    def to_excel(self, xl_outfile : str = None):

        """
        Export in Excel format the analysis dataframe 
        """

        letters = list(ascii_uppercase)
        while True:
            try :
                writer = pd.ExcelWriter(xl_outfile,  engine="xlsxwriter")
                break
            except PermissionError:
                xl_outfile = re.sub(".xlsx$","_1.xlsx", xl_outfile)

        df = self.df
        col_letter = {c : letters[i+1] for i, c in enumerate(df.columns)}
        df.to_excel(writer, sheet_name= "rdcf")
        wb = writer.book
        number = wb.add_format({'num_format': '0.00'})
        percent = wb.add_format({'num_format': '0.00%'})
        # Add a format. Light red fill with dark red text.
        # format1 = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        # Add a format. Green fill with dark green text.
        # format2 = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        # Add a format. Light red fill .
        format3 = wb.add_format({"bg_color": "#F8696B",})

        worksheet = writer.sheets['rdcf']

        # add hyperlink
        for i, s in enumerate(df.index):
            worksheet.write_url(f'B{i+2}', fr'https://www.tradingview.com/symbols/{s}/', string= df.loc[s, 'short_name'] )

        worksheet.add_table(0,0,len(df.index),len(df.columns) ,
                            {"columns" : [{'header' : 'symbol'}]
                                + [{'header' : col} for col in df.columns],
                            'style' : 'Table Style Light 8'})
        worksheet.set_column('B:B', 30, )
        worksheet.set_column(
            f"{col_letter['current_price']}:{col_letter['current_price']}", 13, number)
        worksheet.set_column(
            f"{col_letter['beta']}:{col_letter['price_to_fcf']}", 8, number)
        worksheet.set_column(f"{col_letter['capital_cost']}:{col_letter['assumed_g_ttm']}", 11, percent)
        worksheet.set_column(f"{col_letter['per']}:{col_letter['price_to_book']}", 13, number)
        worksheet.set_column(f"{col_letter['total_payout_ratio']}:{col_letter['total_payout_ratio']}", 11, percent )
        worksheet.set_column(f"{col_letter['roic']}:{col_letter['roic']}", 13, percent )
        # worksheet.set_column(f"{col_letter['mean_g_fcf']}:{col_letter['diff_g']}", 13, percent )

        # format assumed g
        worksheet.conditional_format(
            f"{col_letter['assumed_g']}2:{col_letter['assumed_g_ttm']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num',
            'max_type': 'max', 'mid_type' : 'percentile',
            'min_value' : -0.2, 'mid_value' : 50,  
            'min_color' : '#63BE7B', "max_color" : '#F8696B', 
            "mid_color" : "#FFFFFF"})
        # format debt ratio

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
        # format ROIC
        worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}",
                                     {"type": "cell", "criteria": "<",
                                      "value": 0, "format": format3})
        worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}",
                                    {"type": "3_color_scale", 'min_type': 'num','max_type': 'num',
                                     'mid_type' : 'percentile',
                                    'min_value' : 0, 'mid_value' : 50, "max_value" : 0.15,
                                    "min_color" : '#F8696B', 'max_color' : '#63BE7B' ,
                                    "mid_color" : "#FFFFFF"})

        # format Price to Book
        worksheet.conditional_format(
            f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}",
                                     {"type": "cell", "criteria": "<", 
                                      "value": 0, "format": format3})
        worksheet.conditional_format(
            f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}",
                                    {"type": "3_color_scale", 'min_type': 'num',
                                     'max_type': 'num', 'mid_type' : 'percentile',
                                    'min_value' : 1, 'mid_value' : 50, "max_value" : 10, 
                                    'min_color' : '#63BE7B', "max_color" : '#F8696B',
                                    "mid_color" : "#FFFFFF"})
        
        #
        # worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format1})
        # worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": ">", "value": 0, "format": format2})
        writer.close()

        if sys.platform == "linux" :
            subprocess.call(["open", xl_outfile])
        else :
            os.startfile(xl_outfile)

 

# def upload_file(outfile):
#     """
#     Upload a new file in google drive
#     Returns : Id's of the file uploaded

#     """
#     creds = None
#     # The file token.json stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json", SCOPES)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 "credentials.json", SCOPES
#             )
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open("token.json", "w", encoding= "utf-8") as token:
#             token.write(creds.to_json())

#     try:
#         # create drive api client
#         service = build("drive", "v3", credentials=creds)

#         name = "rdcf_"+ str(date.today())
#         file_metadata = {"name": name}
#         media = MediaFileUpload(outfile)
#         # pylint: disable=maybe-no-member

#         print(f"upload {name}.xlsx to google drive")
#         file = (
#             service.files()
#             .create(body=file_metadata, media_body=media, fields="id")
#             .execute()
#         )
#         print(f'File ID: {file.get("id")}')

#     except HttpError as error:
#         print(f"An error occurred: {error}")
#         file = None

#     return file.get("id")

def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False