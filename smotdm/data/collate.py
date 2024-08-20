from typing import Dict, List, Optional
import torch

from torch.utils.data import default_collate


def collate_tensor_with_padding(batch: List[torch.Tensor]) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def length_to_mask(length, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    assert type(length) == torch.Tensor
    max_len = torch.max(length).item()
    mask = torch.arange(max_len, device=device).expand(
        len(length), int(max_len)
    ) < length.unsqueeze(1)
    return mask


def collate_x_dict(data: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in data])
    if device is not None:
        x = x.to(device)
    length = [len(d) for d in data]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)  # type: ignore
    return batch
