
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
# print(f"Dataset has {len(pyd)} rows")

inputs, query, rate = pyd[0]
print('patient context',inputs.keys())
print('query',query)
print('poisson rate',rate)

'''
Sample Output 

# printing in the dataloader getitem: 
data start at 1991-10-10 00:00:00
data cuts off inputs at 2014-02-24 07:00:00
data actually ends at 2014-08-25 00:01:00
answer to the query is calculated from 271 to end of events 724
query duration 532.4472222222222 days
query start time 2009-05-20 10:25:00
query start idx time 2009-05-20 12:00:00 at idx 411
query end time 2010-11-03 21:09:00
query end idx time 2010-10-25 00:00:00 at idx 438
inputs go from 15 to 271

# printing from this script: 
patient context dict_keys(['static_measurement_indices', 'static_indices', 'start_time', 'dynamic_measurement_indices', 'dynamic_indices', 'dynamic_values', 'time_delta'])
query {'start_delta_from_input_end': 14255425.0, 'end_delta_from_input_end': 15022149.0, 'duration': 766724.0, 'code': 29198}
poisson rate 1.3042502908478148e-06
'''