import torch

from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.model_output import get_event_types
from EventStream.transformer.zero_shot_labeler import Labeler


def masked_idx_in_set(
    indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
) -> torch.BoolTensor:
    return torch.where(mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False)


class TaskLabeler(Labeler):
    def __call__(self, batch: PytorchBatch, input_seq_len: int) -> tuple[torch.LongTensor, torch.BoolTensor]:
        gen_mask = batch.event_mask[:, input_seq_len:]
        gen_measurements = batch.dynamic_measurement_indices[:, input_seq_len:, :]
        gen_indices = batch.dynamic_indices[:, input_seq_len:, :]

        gen_event_types = get_event_types(
            gen_measurements,
            gen_indices,
            self.config.measurements_idxmap["event_type"],
            self.config.vocab_offsets_by_measurement["event_type"],
        )

        # gen_event_types is of shape [batch_size, sequence_length]

        discharge_indices = {
            i for et, i in self.config.event_types_idxmap.items() if ("DISCHARGE" in et.split("&"))
        }
        death_indices = {i for et, i in self.config.event_types_idxmap.items() if ("DEATH" in et.split("&"))}

        is_discharge = masked_idx_in_set(gen_event_types, discharge_indices, gen_mask)
        is_death = masked_idx_in_set(gen_event_types, death_indices, gen_mask)

        no_discharge = (~is_discharge).all(dim=1)
        first_discharge = torch.argmax(is_discharge.float(), 1)
        first_discharge = torch.where(no_discharge, batch.sequence_length + 1, first_discharge)

        no_death = (~is_death).all(dim=1)
        first_death = torch.argmax(is_death.float(), 1)
        first_death = torch.where(no_death, batch.sequence_length + 1, first_death)

        pred_discharge = torch.where(
            (~no_discharge) & (first_discharge < first_death),
            torch.ones_like(first_discharge),
            torch.zeros_like(first_discharge),
        )
        pred_death = torch.where(
            (~no_death) & (first_death <= first_discharge),
            torch.ones_like(first_death),
            torch.zeros_like(first_discharge),
        )

        # MAKE SURE THIS ORDER MATCHES THE EXPECTED LABEL VOCAB
        # Accessible in self.config.label2id
        pred_labels = torch.stack([pred_discharge, pred_death], 1)
        unknown_pred = (pred_discharge == 0) & (pred_death == 0)

        return pred_labels, unknown_pred
