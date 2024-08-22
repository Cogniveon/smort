import h5py
import torch

class TextLoader:
    def __init__(self, dataset_file: str, nfeats=None):
        self.dataset_file = dataset_file
        self.nfeats = nfeats

    def __call__(self, scene_id: str, i: int):
        assert i < 3, "There are only 3 texts in InterX"
        with h5py.File(self.dataset_file, "r") as f:
            texts = f[str(f"texts/{scene_id}")]
            assert type(texts) == h5py.Dataset
            text = torch.from_numpy(texts[i]).to(torch.float)
            x_dict = {"x": text, "length": len(text)}
            return x_dict
        

class MotionLoader:
    def __init__(self, dataset_file: str, fps, normalizer=None, nfeats=None):
        self.fps = fps
        self.dataset_file = dataset_file
        self.motions = {}
        self.normalizer = normalizer
        self.nfeats = nfeats

    def __call__(self, scene_id: str, i: int, start: float, end: float):
        begin = int(start * self.fps)
        end = int(end * self.fps)
        if scene_id not in self.motions:
            with h5py.File(self.dataset_file, "r") as f:
                motions = f[str(f"motions/{scene_id}")]
                assert type(motions) == h5py.Dataset
                motion = torch.from_numpy(motions[i]).to(torch.float)
                if self.normalizer is not None:
                    motion = self.normalizer(motions[i])
                self.motions[f"{scene_id}_{i}"] = motion

        motion = self.motions[f"{scene_id}_{i}"][begin:end]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict

