import torch
from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.zero_shot_labeler import Labeler


class TaskLabeler(Labeler):
    def __call__(
        self, batch: PytorchBatch, input_seq_len: int
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:

        N = len(self.config.label2id)

        unk_preds = (torch.rand((batch.batch_size,)) > 0.5).bool()
        val_preds = (torch.rand((batch.batch_size, N)) > 0.5).float()
        val_preds = torch.where(unk_preds.unsqueeze(-1).expand_as(val_preds), 0, val_preds)

        return val_preds, unk_preds
