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

from dcf_degiro import RDCFAnal

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()


config_dict = {
    'credential_file_path'          : os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json"),
    'capital_cost_equal_market'     : True,
    'use_multiple'                  : True,
    'fcf_history_multiple_method'   : 'mean', # mean or median
    'terminal_price_to_fcf_bounds' : [1, 70],
    'history_avg_nb_year'           : 3,
    'save_data'                     : False,
    'use_last_price_intraday'       : True,
}
outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":
    

    rdcf_anal = RDCFAnal(config_dict)

    
    rdcf_anal.retrieve_shares_from_favorites()
    rdcf_anal.retrieve_shares_from_portfolio()
    rdcf_anal.process()

    # for s in dcf_anal.share_list:
    #     # if s.identity.symbol == 'TM':
    #     s.compute_financial_info()
            # s.retrieve_history()
        # break
#   # dcf_anal.load_df()

    rdcf_anal.to_excel(outfile)
  # upload_file(outfile)
