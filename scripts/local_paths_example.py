"""Template for machine-specific locations.

Copy this file to ``local_paths.py`` (same folder) and fill in the values:

    cp scripts/local_paths_example.py scripts/local_paths.py

``local_paths.py`` is gitignored. The scripts import it after adding ``scripts/`` to
``sys.path``, so run them from the repository root:

    sys.path.append('scripts/')
    from local_paths import BASE_MODEL_PATH
"""
BASE_MODEL_PATH = ""   # trained models,         e.g. "/home/me/MODELS/generative_models"
BASE_DATA_PATH = ""    # datasets,               e.g. "/home/me/DATA/generative_data"
BASE_ROOT = ""         # simulation ROOT files
RF_PARAMS_CONFIG_FILE = ""  # RF-chain parameter JSON
LFMAP_DIR = ""         # galactic noise maps LFmap*.npy
