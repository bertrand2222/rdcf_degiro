import re
import sys
import os
import shutil

import pandas as pd 
import subprocess
import json

sys.path.append(os.environ["HOME"] + "/Documents/rdcf_degiro")

from rdcf_degiro.analysis import RDCFAnal, RDCFSummary

# credentials_path = os.path.join(os.getenv('USERPROFILE'), ".degiro", "credentials.json")
# credentials = build_credentials(location=credentials_path )

# trading_api = API(credentials = credentials )

# trading_api.connect()
# client_details_table = trading_api.get_client_details()

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Définition des constantes pour api google drive
SCOPES = ['https://www.googleapis.com/auth/drive']
GDRIVE_CRED_FILE = os.environ['HOME'] + "/Documents/rdcf_config/rdcf_google_drive_credential.json"
GDRIVE_TOKEN_FILE = os.environ['HOME'] + "/Documents/rdcf_config/token.json"

usr_config_path = os.environ['HOME'] + "/Documents/rdcf_config/user.json"
with open(usr_config_path, "r") as f:
    d = json.load(f)

# ID du fichier Google Drive de destination (trouvable dans l'URL du dossier)
GDRIVE_FILE_ID = d['GDRIVE_FILE_ID']

def upload_excel_to_drive(file_path : str):

    creds = None
    
    # Le fichier token.json stocke les jetons d'accès et d'actualisation de l'utilisateur.
    if os.path.exists(GDRIVE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(GDRIVE_TOKEN_FILE, SCOPES)
        
    # S'il n'y a pas de jetons valides, on demande à l'utilisateur de se connecter via le navigateur.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                GDRIVE_CRED_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Sauvegarder les identifiants pour les prochaines exécutions
        with open(GDRIVE_TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    # Construction du service Drive avec votre propre compte
    service = build('drive', 'v3', credentials=creds)

    
    media = MediaFileUpload(
        file_path, 
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Envoi et conversion
    file = service.files().update(
        # body=file_metadata,
        fileId=GDRIVE_FILE_ID,
        media_body=media,
        fields='id'
    ).execute()

    file_url = f"https://docs.google.com/spreadsheets/d/{file.get('id')}/edit"

    print(f"Fichier transféré et converti avec succès sur votre Drive ! : {file_url}")

symbol_params = {
    'EMEIS' : {"value_to_ebitda" : 8 ,
               "yahoo_symb": "EMEIS.PA"},
    'CLARI' : {"yahoo_symb": "CLARI.PA"},
    'BKD' : {"sector" : "Healthcare",},
    'RIGD' : {"yahoo_symb" : 'RIGD.IL'},
    'MAU' : {"yahoo_symb" : 'MAU.PA'},
    'BY6' : {"yahoo_symb" : 'BYDDY'},
    'TKY' : {"yahoo_symb" : '8035.T'},
    '3CP' : {"yahoo_symb" : 'XIACY'},
    'UBSG' : {"yahoo_symb" : 'UBS'},
    'SMSN' : {"yahoo_symb" : '005930.KS'
              },
    'SMSD' : {"yahoo_symb" : '005935.KS',
            "sector" : "DataCenter",
              "price_from" : "yahoo"},
    'HY9H' : {"yahoo_symb" : '000660.KS',
            "sector" : "DataCenter",
            "price_from" : "yahoo"},
    'HHPD' : {"yahoo_symb" : '2317.TW'},
    'SONY' : {"yahoo_symb" : '6758.T',
              "price_from" : "yahoo"},
    'TM' : {"yahoo_symb" : '7203.T',   # Toyota
            "price_from" : "yahoo"},
    'MBI'  : {"yahoo_symb" : '8058.T',  # Mitsubishi Corp
            "price_from" : "yahoo"},
    'TKY'  : {"yahoo_symb" : '8035.T', # Tokyo Electron
            "price_from" : "yahoo",
            "sector" : "DataCenter",
            },
    'VAN'  : {"yahoo_symb" : '6857.T', # Advantest
            "sector" : "DataCenter",
                "price_from" : "yahoo"},
    "CLS" : {"yahoo_symb" : 'CLS', # Celestica,
            "sector" : "DataCenter",
               },
    "TSM" : {"yahoo_symb" : '2330.TW', # Taiwan Semiconductor Manufacturing Company
            "sector" : "DataCenter",
              "price_from" : "yahoo"},
    "INFY" : {"yahoo_symb" : "INFY.NS", 
              'sector' : "Technology",
              "price_from" : "yahoo"},
    "CRWV" : {"sector" : "DataCenter",
              "yahoo_symb" : "CRWV",
                "price_from" : "yahoo"},
    "ORCL" : {"sector" : "DataCenter",
              "yahoo_symb" : "ORCL",
                "price_from" : "yahoo"},
    "BFSA" : {"sector" : "Chemicals",
                  "yahoo_symb" : "BFSA.DE",
                    "price_from" : "yahoo"},
    "P911" : {"sector" : "Automotive",
                      "yahoo_symb" : "P911.DE",},
    "MSFT" : {"sector" : "Technology",},
    "NVDA" : {"sector" : "DataCenter",},
    "AAPL" : {"sector" : "Technology",},
    "AMD" : {"sector" : "DataCenter",},
    "INTC" : {"sector" : "DataCenter",
              "yahoo_symb" : "INTC",
              "price_from" : "yahoo"},
    "TE" : {"sector" : "Energy",
            "yahoo_symb" : "TE.PA"},  # Technip Energy
    "TTE" : {"sector" : "Energy",}, # Total
    "XOM" : {"sector" : "Energy",}, # Exxon mobil
    "SPM" : {"sector" : "Energy",}, # Saipem
    "OXY" : {"sector" : "Energy",}, # Oxyendal petroleum
    "AM" : {"yahoo_symb" : "AM.PA",
            "sector" : "AerospaceAndDefense",}, # Dassault Aviation
    "RHM" : {"yahoo_symb" : "RHM.DE",
                    "sector" : "AerospaceAndDefense",}, # Rheinmetal
    "HO" : {"yahoo_symb" : "HO.PA",
            "sector" : "AerospaceAndDefense",}, # Thales

    "CS" : {"yahoo_symb" : "CS.PA",
                "sector" : "Inssurance",}, # Axa
    "DSY" : {"yahoo_symb" : "DSY.PA",
            "sector" : "Technology",}, # Dassault Systeme
    "ENR" : {"yahoo_symb" : "ENR.DE"}, # Siemens Energy
    "ALV" : {"yahoo_symb" : "ALV.DE"}, #Allianz
    "MC" : {"yahoo_symb" : "MC.PA"}, #LVMH
    "OR" : {"yahoo_symb" : "OR.PA"}, #Allianz
    "AIR" : {"yahoo_symb" : "AIR.PA"}, #Airbus
    "AI" : {"yahoo_symb" : "AI.PA"}, #Ar liquide
    "LI" : {"yahoo_symb" : "LI.PA"}, #Klepierre
    "SU" : {"yahoo_symb" : "SU.PA"}, #Schneider Electric
    "RMS" : {"yahoo_symb" : "RMS.PA"}, 
    "KER" : {"yahoo_symb" : "KER.PA"}, 
    "ENI" : {"yahoo_symb" : "ENI.MI"}, 
    "SPM" : {"yahoo_symb" : "SPM.MI"}, 
    "VK" : {"yahoo_symb" : "VK.PA"}, 
    "SHELL" : {"yahoo_symb" : "SHELL.AS"}, 
    "8TRA" : {"yahoo_symb" : "8TRA.DE"},
    "URW" : {"yahoo_symb" : "URW.PA"},
    "STLAP" : {"yahoo_symb" : "STLAP.PA"},
    "CAP" : {"yahoo_symb" : "CAP.PA"},
    "AMUN" : {"yahoo_symb" : "AMUN.PA"},
    "SAF" : {"yahoo_symb" : "SAF.PA"},
    "SOI" : {"yahoo_symb" : "SOI.PA"},
    "FGR" : {"yahoo_symb" : "FGR.PA"},
    "STMPA" : {"yahoo_symb" : "STMPA.PA"},
    "ERA" : {"yahoo_symb" : "ERA.PA"},
    "SGO" : {"yahoo_symb" : "SGO.PA"},
    "COFA" : {"yahoo_symb" : "COFA.PA"},
    "SIE" : {"yahoo_symb" : "SIE.DE"},
    "ENGI" : {"yahoo_symb" : "ENGI.PA"},
    "SCR" : {"yahoo_symb" : "SCR.PA"},
    "GTT" : {"yahoo_symb" : "GTT.PA"},
    "EN" : {"yahoo_symb" : "EN.PA"},
    "OVH" : {"yahoo_symb" : "OVH.PA"},
    "VU" : {"yahoo_symb" : "VU.PA"},
    "BAS" : {"yahoo_symb" : "BAS.DE"},
    "LDO" : {"yahoo_symb" : "LDO.MI"},
    "GLE" : {"yahoo_symb" : "GLE.PA"},
    "DG" : {"yahoo_symb" : "DG.PA"},
    "SUBC" : {"yahoo_symb" : "SUBC.OL"},
    "DBG" : {"yahoo_symb" : "DBG.PA"},
    "IBE" : {"yahoo_symb" : "IBE.MC"},
    "GLEN" : {"yahoo_symb" : "GLEN.L"},
    "ALRIB" : {"yahoo_symb" : "ALRIB.PA"},
    "LNA" : {"yahoo_symb" : "LNA.PA"},
    "EC" : {"yahoo_symb" : "EC.PA"},
    "VOW3" : {"yahoo_symb" : "VOW3.DE"},
    "RIGD" : {"yahoo_symb" : "RELIANCE.NS"},
    "RR." : {"yahoo_symb" : "RR.L"},
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
    "update_market_rate"            : True,
    'update_statements'             : False,
    'nb_year_dcf'                   : 8
}
# outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

if __name__ == "__main__":

    rdcf_fcff_anal = RDCFAnal(config_dict)


    rdcf_fcff_anal.retrieve_data()
    # rdcf_fcff_anal.share_list = [ s for s in rdcf_fcff_anal.share_list if s.symbol in [ "INTC", 'CRWV', 'ORCL', 'RR.'] ]
    summary = rdcf_fcff_anal.process()
    summary.save(os.path.join(config_dict['output_folder'],'FCFF.pkl'))

    # exit()

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

    upload_excel_to_drive(xl_outfile)
