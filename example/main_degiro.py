import re
import sys
import os

import pandas as pd 
import subprocess

sys.path.append("/home/bertrand/Documents/rdcf_degiro")

from rdcf_degiro.analysis import RDCFAnal, RDCFSummary

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()

symbol_params = {

    'RIGD' : {"yahoo_tk" : 'RIGD.IL'},
    'MAU' : {"yahoo_tk" : 'MAU.PA'},
    'BY6' : {"yahoo_tk" : 'BYDDY'},
    'TKY' : {"yahoo_tk" : '8035.T'},
    '3CP' : {"yahoo_tk" : 'XIACY'},
    'MBI' : {"yahoo_tk" : '8058.T'},
    'UBSG' : {"yahoo_tk" : 'UBS'},
    'SMSN' : {"yahoo_tk" : '005930.KS'},
    'SMSD' : {"yahoo_tk" : '005935.KS'},
    'HY9H' : {"yahoo_tk" : '000660.KS'},
    'HHPD' : {"yahoo_tk" : '2317.TW'}
}

config_dict = {
    'credential_file_path'          : "/home/bertrand/.degiro/credentials.json",
    'use_beta'                      : False,
    'use_multiple'                  : True,
    'terminal_value_to_ebitda_bounds'  : [0.1, 20],
    'history_avg_nb_year'           : 3,
    'use_last_intraday_price'       : True,
    'output_folder'                 : "/home/bertrand/Documents/rdcf_degiro_out",
    'taxe_rate'                     : 0.25,
    'symbol_params'              : symbol_params,
    "update_market_rate"            : False,
    'update_statements'             : False,
    'nb_year_dcf'                   : 10
}
# outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":

    rdcf_fcff_anal = RDCFAnal(config_dict)

    # rdcf_fcff_anal.share_list = [ s for s in rdcf_fcff_anal.share_list if s.symbol in [ "SU"] ]
    
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
