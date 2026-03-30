import os
import getpass

_user = getpass.getuser()

_paths = {
    "chang.2590": {
        "DIR":       r"D:\Dropbox\project\firm_scope",
        "DATA_DIR":  r"D:\Dropbox\project\firm_scope\data",
        "RAW_DIR":   r"D:\Dropbox\project\firm_scope\data\raw",
        "TEMP_DIR":  r"D:\Dropbox\project\firm_scope\data\temp",
        "PROC_DIR":  r"D:\Dropbox\project\firm_scope\data\processed",
        "OUT_DIR":   r"D:\Dropbox\project\firm_scope\outputs",
        "PLOTS_DIR": r"D:\Dropbox\project\firm_scope\outputs\plots",
        "TABLES_DIR":r"D:\Dropbox\project\firm_scope\outputs\tables",
    },
    "User": {
        "DIR":       r"F:\Dropbox\project\firm_scope",
        "DATA_DIR":  r"F:\Dropbox (Personal)\project\firm_scope\data",
        "RAW_DIR":   r"F:\Dropbox (Personal)\project\firm_scope\data\raw",
        "TEMP_DIR":  r"F:\Dropbox (Personal)\project\firm_scope\data\temp",
        "PROC_DIR":  r"F:\Dropbox (Personal)\project\firm_scope\data\processed",
        "OUT_DIR":   r"F:\Dropbox (Personal)\project\firm_scope\outputs",
        "PLOTS_DIR": r"F:\Dropbox (Personal)\project\firm_scope\outputs\plots",
        "TABLES_DIR":r"F:\Dropbox (Personal)\project\firm_scope\outputs\tables",
    },
}

if _user not in _paths:
    raise ValueError(f"Unknown user '{_user}'. Add paths for this user in utils/set_paths.py.")

DIR        = _paths[_user]["DIR"]
DATA_DIR   = _paths[_user]["DATA_DIR"]
RAW_DIR    = _paths[_user]["RAW_DIR"]
TEMP_DIR   = _paths[_user]["TEMP_DIR"]
PROC_DIR   = _paths[_user]["PROC_DIR"]
OUT_DIR    = _paths[_user]["OUT_DIR"]
PLOTS_DIR  = _paths[_user]["PLOTS_DIR"]
TABLES_DIR = _paths[_user]["TABLES_DIR"]
