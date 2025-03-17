import sys, os 

sys.path.append(r"C:\Users\SAFCOB009150\rdcf\rdcf_degiro")
import pandas as pd

import os
import json
from degiro_connector.trading.api import API , Credentials
from degiro_connector.trading.models.credentials import build_credentials
from degiro_connector.trading.models.account import UpdateOption, UpdateRequest
# from degiro_connector.quotecast.tools.ticker_fetcher import TickerFetcher
import yahooquery as yq
import polars as pl
from importlib import reload
from rdcf_degiro.financial_statements import TYPES
from rdcf_degiro.dcf_degiro import RDCFAnal

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()

yahoo_symbol_cor = {

    'RIGD' : 'RIGD.IL',
    'MAU' : 'MAU.PA',
}

config_dict = {
    'credential_file_path'          : os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json"),
    'use_beta'     : False,
    'use_multiple'                  : True,
    'terminal_price_to_fcf_bounds'  : [1, 50],
    'history_avg_nb_year'           : 3,
    'use_last_intraday_price'       : True,
    'output_folder'                 : r'C:\Users\SAFCOB009150\OneDrive - Saipem\Documents\rdcf_degiro_out',
    'taxe_rate'                     : 0.25,
    'output_name'                   : 'rdcf',
    'yahoo_symbol_cor'              : yahoo_symbol_cor,
    "update_market_rate"            : False,
    'update_statements'             : False,
}
# outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":

    rdcf_anal = RDCFAnal(config_dict)

    # rdcf_anal.share_list = [ s for s in rdcf_anal.share_list if s.symbol in [ "ALSTI"] ]
    
    # rdcf_anal.load_df()
    rdcf_anal.process()

    rdcf_anal.to_excel()
  # upload_file(outfile)
