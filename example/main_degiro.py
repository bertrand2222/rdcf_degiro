import re
import sys
import os
import shutil

import pandas as pd 
import subprocess

sys.path.append(os.environ["HOME"] + "/Documents/rdcf_degiro")

from rdcf_degiro.analysis import RDCFAnal, RDCFSummary

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()

symbol_params = {
    'EMEIS' : {"value_to_ebitda" : 10,},
    'RIGD' : {"yahoo_symb" : 'RIGD.IL'},
    'MAU' : {"yahoo_symb" : 'MAU.PA'},
    'BY6' : {"yahoo_symb" : 'BYDDY'},
    'TKY' : {"yahoo_symb" : '8035.T'},
    '3CP' : {"yahoo_symb" : 'XIACY'},
    'UBSG' : {"yahoo_symb" : 'UBS'},
    'SMSN' : {"yahoo_symb" : '005930.KS'},
    'SMSD' : {"yahoo_symb" : '005935.KS',
              "price_from" : "yahoo"},
    'HY9H' : {"yahoo_symb" : '000660.KS',
            "price_from" : "yahoo"},
    'HHPD' : {"yahoo_symb" : '2317.TW'},
    'SONY' : {"yahoo_symb" : '6758.T',
              "price_from" : "yahoo"},
    'TM' : {"yahoo_symb" : '7203.T',   # Toyota
            "price_from" : "yahoo"},
    'MBI'  : {"yahoo_symb" : '8058.T',  # Mitsubishi Corp
            "price_from" : "yahoo"},
    'TKY'  : {"yahoo_symb" : '8035.T', # Tokyo Electron
            "price_from" : "yahoo"},
    'VAN'  : {"yahoo_symb" : '6857.T', # Advantest
                "price_from" : "yahoo"},
}

config_dict = {
    'credential_file_path'          : os.environ['HOME'] + "/.degiro/credentials.json",
    'use_beta'                      : False,
    'use_multiple'                  : True,
    'terminal_value_to_ebitda_bounds'  : [0.1, 20],
    'history_avg_nb_year'           : 3,
    'use_last_intraday_price'       : True,
    'output_folder'                 : os.environ['HOME'] + "/Documents/rdcf_degiro_out",
    'taxe_rate'                     : 0.25,
    'symbol_params'              : symbol_params,
    "update_market_rate"            : False,
    'update_statements'             : False,
    'nb_year_dcf'                   : 10
}
# outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":

    rdcf_fcff_anal = RDCFAnal(config_dict)

    # rdcf_fcff_anal.share_list = [ s for s in rdcf_fcff_anal.share_list if s.symbol in [ "SNDK"] ]

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
        if shutil.which("xdg-open") is not None:
            subprocess.call(["xdg-open", xl_outfile])
        else:
            subprocess.call(["open", xl_outfile])
    else :
        os.startfile(xl_outfile)
