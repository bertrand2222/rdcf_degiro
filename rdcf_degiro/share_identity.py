import dataclasses

@dataclasses.dataclass
class ShareIdentity():
    """
    object containing constant identity attributes of a share
    """
    name : str = None
    isin :str = None
    vwd_id : str = None
    vwd_id_secondary : str = None
    symbol : str = None
    currency : str = None
    vwd_identifier_type : str = None
    vwd_identifier_type_secondary : str = None

    def __init__(self, s_dict : dict):

        self.__dict__.update(s_dict)