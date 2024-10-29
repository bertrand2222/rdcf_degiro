import os
import sys
from string import ascii_uppercase
import json
from datetime import date
import subprocess
import warnings
import numpy as np
from tabulate import tabulate
import urllib3
import pandas as pd
# from colorama import Fore
import requests
from lxml import html
import urllib3
from degiro_connector.trading.api import API

from typing import List
from share import Share
from session_model_dcf import SessionModelDCF, ERROR_SYMBOL
import yahooquery as yq
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

YEAR_G = 0





SHARE_PROFILE_FILE = 'share_profile.json'


class DCFAnal():
    """
    object containing a reverse dcf analysis and all its context
    """

    def __init__(self,
                 config_dict : dict = None,
             ) -> None:
        
        reload(pd)
        self.__dict__.update(config_dict)
        self.df : pd.DataFrame = None
        self.share_list : List[Share] = None
        self.trading_api : API = None
        self.ids : List[int] = None

        self.session_model = SessionModelDCF(config_dict)

        self.connect()

    def connect(self) :
        """
        Connexion
        """
        self.session_model.connect()
        self.trading_api = self.session_model.trading_api
     
    def retrieve_shares_from_favorites(self):
        """
        Retrieve stock and fund ids recorded in Degiro account favorite list
        """
        favorite_batch = self.trading_api.get_favorite(raw=False)
        self.ids = favorite_batch.data[0].product_ids

        # FETCH PRODUCT INFO
        product_info = self.trading_api.get_products_info(
            product_list= self.ids,
            raw=False,
        )
        
        self.share_list = [Share( s_dict.__dict__ ,
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
        
        # self.session_model.market_infos.update()

        share_list = self.share_list
        for s in share_list :

            flg = s.querry_financial_info()
            if flg == ERROR_SYMBOL :
                continue
            flg = s.compute_financial_info()
            if flg == ERROR_SYMBOL :
                continue



            s.eval_g(use_multiple= self.use_multiple)

 
        df = pd.DataFrame(index= self.ids,
                          data= {
                                'short_name' : [s.short_name for s in share_list] ,
                                'current_price' : [s.close_price for s in share_list],
                                'currency' : [s.financial_currency for s in share_list],
                                'beta' : [s.beta for s in share_list],
                                'price_to_fcf' : [s.price_to_fcf for s in share_list],
                                'capital_cost' : [s.capital_cost for s in share_list],
                                'cmpc' : [s.cmpc for s in share_list],
                                'assumed_g' : [s.g for s in share_list],  
                                'assumed_g_last' : [s.g_last for s in share_list],  
                                'per' :  [s.per for s in share_list ],
                                'roic' : [s.roic for s in share_list], 
                                'debt_to_equity' : [s.debt_to_equity for s in share_list],
                                'price_to_book' : [s.price_to_book for s in share_list] ,
                                'total_payout_ratio' : [s.total_payout_ratio for s in share_list] ,
                                # 'mean_g_fcf': [s.mean_g_fcf for s in share_list] ,
                                # 'mean_g_tr' : [s.mean_g_tr for s in share_list],
                                # 'mean_g_inc' : [s.mean_g_netinc for s in share_list]
                                    })


        # df["diff_g"] = df['mean_g_tr'] - df['assumed_g']
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
        # self.xl_outfile = xl_outfile
        writer = pd.ExcelWriter(xl_outfile,  engine="xlsxwriter")
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
            worksheet.write_url(f'B{i+2}', fr'https://finance.yahoo.com/quote/{s}/', string= df.loc[s, 'short_name'] )

        worksheet.add_table(0,0,len(df.index),len(df.columns) ,
                            {"columns" : [{'header' : 'symbol'}]
                                + [{'header' : col} for col in df.columns],
                            'style' : 'Table Style Light 8'})
        worksheet.set_column('B:B', 30, )
        worksheet.set_column(
            f"{col_letter['current_price']}:{col_letter['current_price']}", 13, number)
        worksheet.set_column(
            f"{col_letter['beta']}:{col_letter['price_to_fcf']}", 8, number)
        worksheet.set_column(f"{col_letter['capital_cost']}:{col_letter['assumed_g_last']}", 11, percent)
        worksheet.set_column(f"{col_letter['per']}:{col_letter['price_to_book']}", 13, number)
        worksheet.set_column(f"{col_letter['total_payout_ratio']}:{col_letter['total_payout_ratio']}", 11, percent )
        worksheet.set_column(f"{col_letter['roic']}:{col_letter['roic']}", 13, percent )
        # worksheet.set_column(f"{col_letter['mean_g_fcf']}:{col_letter['diff_g']}", 13, percent )

        # format assumed g
        worksheet.conditional_format(
            f"{col_letter['assumed_g']}2:{col_letter['assumed_g_last']}{len(df.index)+1}",
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
