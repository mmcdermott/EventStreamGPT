
import os
import rootutils
import sys
root = rootutils.setup_root(os.path.abspath(""), dotenv=True, pythonpath=True, cwd=False)
sys.path.append(os.environ["EVENT_STREAM_PATH"])
import os
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime, timedelta
from humanize import naturalsize, naturaldelta
from pathlib import Path
from sparklines import sparklines
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Callable

from EventStream.data.dataset_polars import Dataset
from EventStream.data.config import PytorchDatasetConfig
from EventStream.data.types import PytorchBatch
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.tasks.profile import add_tasks_from




COHORT_NAME = "ESD_09-01-23-1"
TASK_NAME = "readmission_30d_all"
PROJECT_DIR = Path(os.environ["PROJECT_DIR"])
dataset_dir = f"/storage/shared/mgh-hf-dataset/processed/{COHORT_NAME}" # PROJECT_DIR / "data" / COHORT_NAME

pyd_config = PytorchDatasetConfig(
    save_dir=dataset_dir,
    max_seq_len=256,
    train_subset_size=0.001,
    train_subset_seed=79163,
    do_include_start_time_min=True,
    #cache_for_epochs=1,
)

pyd = PytorchDataset(config=pyd_config, split='train')

VOCAB_OBS_FREQ = 1 # cant find this in the config ??

codes = []
vocab = Dataset.load(Path(dataset_dir)).unified_vocabulary_idxmap
for key, cfg in pyd.measurement_configs.items(): 
    has_value = 'regression' in cfg.modality
    ofoc = cfg.observation_rate_over_cases
    ofpc = cfg.observation_rate_per_case
    for code_name, code_idx in vocab[key].items(): 
        if code_name=="UNK": continue 
        if '__EQ_' in code_name: has_value = False
        obs_freq = ofoc * ofpc * VOCAB_OBS_FREQ
        codes.append( (code_name, code_idx, has_value, obs_freq) ) 
    
# todo: 
# figure out where to get the VOCAB_OBS_FREQ
# code sampling strategy based on the observation frequency 
# place in pyd class 

print(f"Dataset has {len(pyd)} rows")
inputs, query, freq = pyd[0]
print('patient context',inputs.keys())
print('query',query)
print('frequency',freq)