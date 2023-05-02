import sys
sys.path.append('../..')

from EventStream.EventStreamData.event_stream_dataset import EventStreamDataset
from EventStream.EventStreamData.config import EventStreamPytorchDatasetConfig

from EventStream.EventStreamData.event_stream_pytorch_dataset import EventStreamPytorchDataset
from EventStream.EventStreamTransformer.config import (
    StructuredEventStreamTransformerConfig, EventStreamOptimizationConfig
)
from EventStream.EventStreamTransformer.model import (
    StructuredEventStreamTransformerForGenerativeSequenceModeling
)
from EventStream.EventStreamTransformer.generative_sequence_modelling_lightning import (
    fit_generative_sequence_model
)

from collections import Counter
from pathlib import Path
import argparse, math, time, torch, pandas as pd

from typing import Dict, Optional, Tuple, Union

def main():
    parser = argparse.ArgumentParser(
        prog = 'pretrain.py',
        description = 'Runs an eventstream Generative Medical sequence model',
    )

    parser.add_argument(
        '--dataset_save_dir', type=str, help="Which dataset should be used?"
    )
    parser.add_argument(
        '--model_save_dir', type=str, help="Where to store the model?"
    )

    # Data config.
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length.')

    # Model config
    parser.add_argument('--hidden_size', type=int, default=100, help='Model hidden size.')
    parser.add_argument('--head_dim', type=int, default=None, help='Model hidden size per head.')
    parser.add_argument('--num_attention_heads', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--resid_dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--input_dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout')

    parser.add_argument('--structured_event_processing_mode', type=str, default='conditionally_independent')
    parser.add_argument('--seq_attention_types', type=str, default='local,global')
    parser.add_argument('--dep_graph_attention_types', type=str, default='global')
    parser.add_argument('--seq_window_size', type=int, default=8)
    parser.add_argument('--dep_graph_window_size', type=int, default=2)
    parser.add_argument(
        '--do_add_temporal_position_embeddings_to_data_embeddings', action=argparse.BooleanOptionalAction,
        default=False, help='do add temporal position embeddings to data embeddings.'
    )
    parser.add_argument(
        '--do_full_block_in_seq_attention', action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        '--do_full_block_in_dep_graph_attention', action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument('--TTE_generation_layer_type', type=str, default='log_normal_mixture')
    parser.add_argument('--TTE_lognormal_generation_num_components', type=int, default=4)

    # Data input config.
    parser.add_argument('--static_embedding_mode', type=str, default='sum_all')
    parser.add_argument(
        '--do_normalize_by_measurement_index', action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument('--do_split_embeddings', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--categorical_embedding_dim', type=int, default=32)
    parser.add_argument('--numerical_embedding_dim', type=int, default=32)
    parser.add_argument('--static_weight', type=float, default=0.5)
    parser.add_argument('--categorical_weight', type=float, default=0.5)

    # Dependency Graph
    # Specialize to your dataset!
    # parser.add_argument(
    #     '--do_event_type_alone_after_time', action=argparse.BooleanOptionalAction, default=False
    # )
    # parser.add_argument(
    #     '--do_split_itemid_categorical_and_numerical', action=argparse.BooleanOptionalAction,
    #     default=False
    # )
    # parser.add_argument(
    #     '--do_icd_codes_after_discharge_location', action=argparse.BooleanOptionalAction, default=False
    # )

    # Optimization config
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial LR.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size.')
    parser.add_argument('--max_epochs', type=int, default=400, help='Max Epochs.')
    parser.add_argument('--lr_frac_warmup_steps', type=float, default=1e-2)
    parser.add_argument('--lr_decay_power', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--end_lr_frac_of_init_lr', type=float, default=5e-3)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--num_dataloader_workers', type=int, default=19, help='# of dataloader workers.')
    parser.add_argument('--model_name', type=str, default=None, help='model name / save directory')

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = time.asctime()

    if args.head_dim is not None:
        args.hidden_size = args.head_dim * args.num_attention_heads

    model_dir = Path(args.model_save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # As we're using pre-cached data, do_normalize_log_inter_event_times is always true.
    data_config = EventStreamPytorchDatasetConfig(max_seq_len = args.max_seq_len)

    seq_attention_types_list = args.seq_attention_types.split(',')
    n_repeats_list = args.num_hidden_layers // len(seq_attention_types_list)
    seq_attention_types = [[
        seq_attention_types_list, args.num_hidden_layers // len(seq_attention_types_list)
    ]]

    if n_repeats_list * len(seq_attention_types_list) != args.num_hidden_layers:
        seq_attention_types.append([[seq_attention_types_list[0]], 1])

    dep_graph_attention_types_list = args.dep_graph_attention_types.split(',')
    dep_graph_attention_types = [[
        dep_graph_attention_types_list, args.num_hidden_layers // len(dep_graph_attention_types_list)
    ]]

    # measurements_per_dep_graph_level = [
    #     ['age', 'time_of_day'],
    # ]

    # if args.do_event_type_alone_after_time:
    #     measurements_per_dep_graph_level.append(['event_type'])
    #     measurements_per_dep_graph_level.append([
    #         'admission_type', 'admission_location', 'discharge_location', 'first_careunit',
    #         'race', 'language', 'marital_status', 'insurance', 'medication',
    #     ])
    # else:
    #     measurements_per_dep_graph_level.append([
    #         'event_type', 'admission_type', 'admission_location', 'discharge_location', 'first_careunit',
    #         'race', 'language', 'marital_status', 'insurance', 'medication',
    #     ])

    # if args.do_split_itemid_categorical_and_numerical:
    #     if args.do_icd_codes_after_discharge_location:
    #         measurements_per_dep_graph_level[-1].append(('itemid', 'categorical_only'))
    #         measurements_per_dep_graph_level.append([('itemid', 'numerical_only'), 'icd_codes'])
    #     else:
    #         measurements_per_dep_graph_level[-1].extend([('itemid', 'categorical_only'), 'icd_codes'])
    #         measurements_per_dep_graph_level.append([('itemid', 'numerical_only')])
    # else:
    #     measurements_per_dep_graph_level[-1].append('itemid')
    #     if args.do_icd_codes_after_discharge_location:
    #         measurements_per_dep_graph_level.append(['icd_codes'])
    #     else:
    #         measurements_per_dep_graph_level[-1].append('icd_codes')

    dep_graph_kwargs = {
        'do_full_block_in_seq_attention': args.do_full_block_in_seq_attention,
        'do_full_block_in_dep_graph_attention': args.do_full_block_in_dep_graph_attention,
        'do_add_temporal_position_embeddings_to_data_embeddings': args.do_add_temporal_position_embeddings_to_data_embeddings,
        'dep_graph_attention_types': dep_graph_attention_types,
        'dep_graph_window_size': len(measurements_per_dep_graph_level),
        'measurements_per_dep_graph_level': measurements_per_dep_graph_level
    }
    if args.structured_event_processing_mode == 'conditionally_independent':
        dep_graph_kwargs = {k: None for k, v in dep_graph_kwargs.items()}

    input_kwargs = {
        'static_embedding_mode': args.static_embedding_mode,
        'do_normalize_by_measurement_index': args.do_normalize_by_measurement_index,
    }

    if args.static_embedding_mode != 'drop':
        assert 0 <= args.static_weight <= 1
        input_kwargs['static_weight'] = args.static_weight
        input_kwargs['dynamic_weight'] = 1 - args.static_weight
    if args.do_split_embeddings:
        input_kwargs['categorical_embedding_dim'] = args.categorical_embedding_dim
        input_kwargs['numerical_embedding_dim'] = args.numerical_embedding_dim

        assert 0 <= args.categorical_weight <= 1
        input_kwargs['categorical_weight'] = args.categorical_weight
        input_kwargs['numerical_weight'] = 1 - args.categorical_weight
    else:
        input_kwargs['categorical_embedding_dim'] = None
        input_kwargs['numerical_embedding_dim'] = None

    TTE_kwargs = {
        'TTE_generation_layer_type': args.TTE_generation_layer_type,
    }
    if args.TTE_generation_layer_type == 'log_normal_mixture':
        TTE_kwargs['TTE_lognormal_generation_num_components'] = args.TTE_lognormal_generation_num_components

    config = StructuredEventStreamTransformerConfig(
        hidden_size = args.hidden_size,
        head_dim = args.head_dim,
        num_attention_heads = args.num_attention_heads,
        num_hidden_layers = args.num_hidden_layers,
        resid_dropout = args.resid_dropout,
        input_dropout = args.input_dropout,
        attention_dropout = args.attention_dropout,
        seq_attention_types = seq_attention_types,
        seq_window_size = args.seq_window_size,
        structured_event_processing_mode = args.structured_event_processing_mode,
        **TTE_kwargs,
        **input_kwargs,
        **dep_graph_kwargs,
    )

    end_lr = args.init_lr * args.end_lr_frac_of_init_lr
    optimization_config = EventStreamOptimizationConfig(
        max_epochs = args.max_epochs,
        init_lr = args.init_lr,
        batch_size = args.batch_size,
        lr_frac_warmup_steps = args.lr_frac_warmup_steps,
        lr_decay_power = args.lr_decay_power,
        weight_decay = args.weight_decay,
        end_lr = end_lr,
        patience = args.patience,
    )

    out = fit_generative_sequence_model(
        wandb_name = args.model_name,
        save_dir = model_dir,
        config = config,
        optimization_config = optimization_config,
        data_config = data_config,
        dataset_dir = Path(args.dataset_dir),
        num_dataloader_workers = args.num_dataloader_workers,
        do_detect_anomaly = False,
        log_every_n_steps = 1,
        do_skip_all_metrics_in_train = True,
        do_final_validation_on_metrics = True,
        do_save_pretrained = True,
    )

if __name__ == "__main__": main()
