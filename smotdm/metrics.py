import torch
from torchmetrics import Metric


class MeanStdMetric(Metric):
    def __init__(self, nfeats=166):
        super().__init__()
        self.nfeats = nfeats
        self.add_state("sums", default=torch.zeros(self.nfeats), dist_reduce_fx="sum")
        self.add_state(
            "sum_of_squares", default=torch.zeros(self.nfeats), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, feature_tensors: torch.Tensor, num_frames: int) -> None:
        if feature_tensors.shape[-1] != self.nfeats:
            raise ValueError("Feature dim does not match!")

        self.count += num_frames
        self.sums += feature_tensors.sum(dim=0)
        self.sum_of_squares += (feature_tensors**2).sum(dim=0)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sums / self.count
        variance = (self.sum_of_squares / self.count) - (mean**2)
        std = torch.sqrt(variance)
        return mean, std
