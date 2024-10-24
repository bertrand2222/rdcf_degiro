import os
import json
from degiro_connector.trading.api import API , Credentials
from degiro_connector.trading.models.credentials import build_credentials
from degiro_connector.trading.models.product_search import LookupRequest, StocksRequest
# from degiro_connector.quotecast.tools.ticker_fetcher import TickerFetcher
import yahooquery as yq
import polars as pl
from dcf_degiro import DCFAnal


CASH_FLOW = 2

EURONEXT_ID = '710'
NASDAQ_ID = '663'

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()


# financial statement

# company_profile = trading_api.get_company_profile(
#     product_isin='FR0000131906',
# )
# company_ratios = trading_api.get_company_ratios(
#     product_isin='FR0000131906',
# )

# financial_st = trading_api.get_financial_statements(
#     product_isin='FR0000131906',
#     raw= True
# )
# with open("financial_statement.json", "w", encoding= "utf8") as outfile: 
#     json.dump(financial_st, outfile, indent = 4)
# num = 1
# print(financial_st['data']['interim'][num]['fiscalYear'])
# print(financial_st['data']['interim'][num]['statements'][CASH_FLOW])
# trading_api.logout()



# FETCH PRODUCTS

# favorite_batch = trading_api.get_favorite(raw=False)


 # SETUP REQUEST
# request = ProductsInfo.Request()
# request.products.extend([96008, 1153605, 5462588])
# underl = StocksRequest(search_text = 'US0378331005', limit=4,
#         offset=0,
#          )

# product_batch = trading_api.product_search(product_request=underl, raw=True)
# pr = product_batch['products'][0]
# for p in product_batch['products']:
#     if p['exchangeId'] == EURONEXT_ID:
#         pr = p
#         break




    
#     if 'products' in product_batch :
#         requested_shares[s] = {'name' : product_batch['products'][0]['name'], "isin" :product_batch['products'][0]['isin'] }
#         print(product_batch['products'][0]['name'], product_batch['products'][0]['isin'] )
#     else :
#         requested_shares[s] = {'name' : "not found", "isin" : "not found" }
#         print(f'{s} not found in degiro data' )

# with open("requested_shares.json", "w", encoding= "utf8") as outfile: 
#     json.dump(requested_shares, outfile, indent = 4)


if __name__ == "__main__":
    

    dcf_anal = DCFAnal(  capital_cost_equal_market = True)

    # client_details_table = dcf_anal.trading_api.get_client_details()

    dcf_anal.retrieve_shares_from_favorites()
    dcf_anal.process()
    # for s in dcf_anal.share_list:
    #     s.get_char()
    #     break
#   # dcf_anal.load_df()

#   dcf_anal.to_excel(outfile)
  # upload_file(outfile)




  # share = Share("ACA.PA")
  # share.eval_g(2.6e9, True)
  # share = Share("TTE.PA")
  # share.eval_g( pr= True, )
  # share.get_dcf( pr= True)
