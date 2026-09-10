import re
import sys
import os

import pandas as pd 

sys.path.append(os.path.join(os.getenv('USERPROFILE'),"rdcf", "rdcf_degiro"))

from rdcf_degiro.analysis import RDCFAnal, RDCFSummary

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()

yahoo_symbol_cor = {

    'RIGD' : 'RIGD.IL',
    'MAU' : 'MAU.PA',
    'BY6' : 'BYDDY',
    'TKY' : '8035.T',
    '3CP' : 'XIACY',
    'MBI' : '8058.T',
    'UBSG' : 'UBS',
    'SMSN' : '005930.KS',
    'HY9H' : '000660.KS',
    'HHPD' : '2317.TW'
}

config_dict = {
    'credential_file_path'          : os.path.join(os.getenv('USERPROFILE'), "OneDrive - Saipem", ".degiro", "credentials.json"),
    'use_beta'                      : False,
    'use_multiple'                  : True,
    'terminal_value_to_ebitda_bounds'  : [0.1, 20],
    'history_avg_nb_year'           : 3,
    'use_last_intraday_price'       : True,
    'output_folder'                 : r'C:\Users\SAFCOB009150\OneDrive - Saipem\Documents\rdcf_degiro_out',
    'taxe_rate'                     : 0.25,
    'yahoo_symbol_cor'              : yahoo_symbol_cor,
    "update_market_rate"            : False,
    'update_statements'             : False,
    'nb_year_dcf'                   : 10
}
# outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":

    rdcf_fcff_anal = RDCFAnal(config_dict)

    # rdcf_fcff_anal.share_list = [ s for s in rdcf_fcff_anal.share_list if s.symbol in [ "MBI"] ]
    
    rdcf_fcff_anal.retrieve_data()
    summary = rdcf_fcff_anal.process()
    summary.save(os.path.join(config_dict['output_folder'],'FCFF.pkl'))

    # summary = RDCFSummary.load(os.path.join(config_dict['output_folder'],'FCFF.pkl'))

    xl_outfile = os.path.join(rdcf_fcff_anal.session_model.output_folder,  "rdcf.xlsx")
    while True:
        try :
            writer = pd.ExcelWriter(xl_outfile,  engine="xlsxwriter")
            break
        except PermissionError:
            xl_outfile = re.sub(".xlsx$","_1.xlsx", xl_outfile)

    summary.to_excel(writer, 'FCFF')
    
    writer.close()

    if sys.platform == "linux" :
        subprocess.call(["open", xl_outfile])
    else :
        os.startfile(xl_outfile)
