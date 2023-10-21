import json
import requests
import numpy as np
import os

_default_host = 'http://localhost:6060/'
_headers = {'Accept': 'application/json', "Content-Type": "application/json", "Charset": "UTF-8"}


# Copied here from engine, unclear where it belongs be
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


""" Function: Get Gateway Properties

    Description: Extract node information from a parameters object
    
    Parameters:
        - params (dict): an Aequitas params object
    
    Returns:
        - a dict of gateway params or nothing
"""
def _get_gateway_element(params: dict, key: str, version: str) -> dict:
    _e = {"key": None, "version": "0"} | (params["element"] if "element" in params else {}) | {k: v for (k, v) in {
        "key": key,
        "version": version
    }.items() if v is not None}

    if _e['key'] is None:
        raise ValueError("No element key defined in parameters or element_key= parameter")

    return _e


""" Class: Gateway

    Description:
        The Gateway provides persistence for the Aequitas parameters object and synchronization 
        with a Context Engine.
        If host is left blank, gateway will instead persist to files.

    Parameters:
        - project_key (string): A project key under which the information will be stored
        - (optional) gw_name: The name of file-system persistence, default 'aq_gw' No effect on remote host.
        - (optional) host: A remote context engine.

    Returns:
        - A Gateway object

"""
class Gateway:
    def __init__(self, project_key, gw_name=None, host=None):
        self.pid = project_key
        self._host = host
        self._gw_name = 'aq_gw' if gw_name is None else gw_name

    """ Function: Load Parameters

        Description:
            Initialize an Aequitas object using a parameters dict 

        Parameters:
            - dataset (pd.Dataframe): The dataset.
            - parameters (dict): A dictionary containing the additional information
            - filesystem: Force filesystem even if a host is set
        Returns:
            - a parameters object
    """
    def load_element(self, element_key: str, version: str = "0", filesystem: bool = False):

        if self._host is None or filesystem is True:
            return self._load_element_from_filesystem(element_key, version)
        else:
            return self._load_element_from_remote(element_key, version)

    def save_element(self, parameters: dict, element_key: str = None, version: str = None, filesystem: bool = False):
        _e = _get_gateway_element(parameters, element_key, version)

        # modify the parameters object to include element props
        _p = parameters | {"element": _e}

        if self._host is None or filesystem is True:
            return self._save_element_to_filesystem(_e['key'], _e['version'], _p)
        else:
            return self._save_element_to_remote(_e['key'], _e['version'], _p)

    # ------------------- Private Functions -------------------

    """ Function: Compose a storage path for the element on fs 
        Description: n/a
        TODO: let useer set a working directory / name for this storage.
    """
    def _get_fs_path(self, element_key=None, data_version=None):
        path = f'{self._gw_name}/{self.pid}'
        if element_key is not None:
            path += f'/{element_key}'
            if data_version is not None:
                path += f'__{data_version}'
            else:
                path += f'__0'
            path += '.json'
        return path

    """ Function: Ensure file structure 
        Description: Sets up folders for local persistence.
    """
    def _ensure_filestructure(self):
        path = self._get_fs_path()
        if not os.path.exists(path):
            os.makedirs(path)

    """ Function: Load from file system

        Description: Loads a parameter file from the filesystem.
            The filename will follow this structure:
                [gwname]/[project_name]/[element_key]/[version].json

        Params:
            - element_key (string): the data key
            - version (string): the version key

        Returns:
            - the parameters object

    """
    def _load_element_from_filesystem(self, element_key, version):
        with open(self._get_fs_path(element_key, version)) as f:
            return json.loads(f.read())

    """ Function: Save to file system

        Description: Loads a parameter file from the filesystem.
            The filename will follow this structure:
                [gwname]/[project_name]/[element_key]/[version].json
        Params:
            - element_key (string): the data key
            - version (string): the version key        
            - parameters (object): data to save      
    """
    def _save_element_to_filesystem(self, element_key, version, parameters):
        self._ensure_filestructure()
        with open(self._get_fs_path(element_key, version), 'w') as f:
            f.write(json.dumps(parameters, cls=NpEncoder))

    """ Function: Load from remote service

        Description: Loads a parameter file from the remote service.

        Params:
            - element_key (string): the data key
            - version (string): the version key

        Returns:
            - the parameters object

    """
    def _load_element_from_remote(self, element_key, version):

        if version is None:
            r = requests.get(f'{self._host}project/{self.pid}/data/{element_key}',
                             headers=_headers)
        else:
            r = requests.get(f'{self._host}project/{self.pid}/data/{element_key}/{version}',
                             headers=_headers)

        if r.status_code != 200:
            print("Error: ", r.status_code)

        return r.json()

    """ Function: Save to remote service

        Description: Loads a parameter file from the remote service.

        Params:
            - element_key (string): the data key
            - version (string): the version key        
            - parameters (object): data to save
    """
    def _save_element_to_remote(self, element_key, version, parameters):

        if version is None:
            r = requests.put(f'{self._host}project/{self.pid}/data/{element_key}/0',
                             data=json.dumps(parameters, cls=NpEncoder),
                             headers=_headers)
        else:
            r = requests.put(f'{self._host}project/{self.pid}/data/{element_key}/{version}',
                             data=json.dumps(parameters, cls=NpEncoder),
                             headers=_headers)

        if r.status_code != 200:
            print("Error: ", r.status_code)
