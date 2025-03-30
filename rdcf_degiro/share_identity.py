# import dataclasses

# @dataclasses.dataclass
from rdcf_degiro.session_model_dcf import SessionModelDCF


class ShareIdentity():
    """
    object containing constant identity attributes of a share
    """
    name : str = None
    isin :str = None
    vwd_id : str = None
    vwd_id_secondary : str = None
    symbol : str = None
    share_currency : str = None
    vwd_identifier_type : str = None
    vwd_identifier_type_secondary : str = None
    session_model : SessionModelDCF = None

