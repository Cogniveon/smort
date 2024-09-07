import math
import os
import random
from typing import Literal, Optional

import numpy as np
import torch
from omegaconf import ListConfig
from smplx import SMPLX
from tqdm.auto import tqdm

from smort.data.data_module import InterXDataModule
from smort.models.smort import SMORT
from smort.rifke import feats_to_joints


def get_smplx_model(
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    model_path: str = "deps/smplx/SMPLX_NEUTRAL.npz",
) -> SMPLX:
    """
    Returns an instance of the SMPLX model, configured with the given batch size and device.

    Args:
        batch_size (Optional[int]): The batch size for the model. Defaults to 1 if not specified.
        device (Optional[torch.device]): The device to run the model on (CPU or GPU). Defaults to CPU.
        model_path (str): The path to the SMPLX model file. Defaults to 'deps/smplx/SMPLX_NEUTRAL.npz'.

    Returns:
        SMPLX: The configured SMPLX model instance.
    """
    # Initialize the SMPLX model
    smplx_model = SMPLX(
        model_path=model_path,
        num_betas=10,
        use_pca=False,
        use_face_contour=True,
        batch_size=batch_size if batch_size is not None else 1,
    )

    # Move the model to the specified device
    smplx_model.to(device)

    return smplx_model


@torch.no_grad()
def compute_joints(
    smplx_model: SMPLX,
    smplx_params: dict,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    smplx_model.to(device)

    output = smplx_model(**smplx_params)

    transform_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    transformed_joints = torch.matmul(output.joints, transform_matrix)

    return transformed_joints[:, : smplx_model.NUM_BODY_JOINTS + 3, :]


def loop_interx(
    base_dir: str,
    include_only: Literal["all"] | ListConfig | list[str] | int = "all",
    exclude: list[str] = [],
    device=torch.device("cpu"),
    fps: float = 20.0,
    min_seconds: float = 2.0,
    max_seconds: float = 30.0,
):
    if include_only == "all":
        motions = os.listdir(f"{base_dir}/motions")
    elif type(include_only) == int:
        motions = os.listdir(f"{base_dir}/motions")
        random.shuffle(motions)
        motions = motions[:include_only]
    elif type(include_only) == ListConfig:
        motions = [
            x
            for x in os.listdir(f"{base_dir}/motions")
            if x in include_only and x not in exclude
        ]
    pbar = tqdm(motions, total=len(motions))
    for scene_id in pbar:
        with open(f"{base_dir}/texts/{str(scene_id)}.txt", "r") as annotation:
            texts = []
            for line in annotation.readlines():
                texts.append(line.rstrip("\n"))

            motions = []
            end: Optional[int] = None
            for file in [
                f"{base_dir}/motions/{scene_id}/P1.npz",
                f"{base_dir}/motions/{scene_id}/P2.npz",
            ]:
                data = np.load(file)
                num_frames = data["pose_body"].shape[0]

                if num_frames < math.floor(min_seconds * fps):
                    continue
                if end is None:
                    end = math.floor(max_seconds * fps)

                motions.append(
                    {
                        "body_pose": torch.tensor(
                            data["pose_body"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "left_hand_pose": torch.tensor(
                            data["pose_lhand"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "right_hand_pose": torch.tensor(
                            data["pose_rhand"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "transl": torch.tensor(
                            data["trans"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "global_orient": torch.tensor(
                            data["root_orient"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                    }
                )

            if end is None:
                continue
            pbar.set_postfix(dict(scene_id=scene_id))
            yield scene_id, motions, end, texts


def get_random_sample_from_dataset(
    dataset_path: str = "deps/interx/processed/dataset.h5",
    scene_id: Optional[str | int] = None,
):
    # text_motion_dataset = TextMotionDataset(
    #     "deps/interx/processed/dataset.h5",
    # )
    # train_dataloader = DataLoader(
    #     text_motion_dataset,
    #     batch_size=1,
    #     collate_fn=text_motion_dataset.collate_fn,
    #     shuffle=True,
    #     # num_workers=7,
    #     # persistent_workers=True,
    # )
    # mean, std = text_motion_dataset.get_mean_std()
    # assert type(mean) == torch.Tensor and type(std) == torch.Tensor
    # model = SMORT(mean, std)

    # trainer = Trainer(
    #     accelerator="cpu", max_epochs=10, fast_dev_run=False, num_sanity_val_steps=0
    # )

    # trainer.fit(model, data_module)

    data_module = InterXDataModule(
        dataset_path,
        batch_size=1,
        num_workers=1,
        use_tiny=1.0,
        return_scene=True,
        return_scene_text=True,
    )
    data_module.setup("train")

    if scene_id:
        sample = data_module.dataset.collate_fn([data_module.get_sample(scene_id)])
    else:
        scene_idx = random.randint(0, len(data_module.dataset))
        print(f"Random Scene: {scene_idx}")
        sample = data_module.dataset.collate_fn([data_module.get_sample(scene_idx)])

    return sample, data_module.dataset


def get_joints_from_ckpt(
    model_ckpt: str = "model-xwis7v3p:v0",
    scene_id: Optional[str | int] = None,
):
    if not os.path.exists(f"artifacts/{model_ckpt}/model.ckpt"):
        import wandb

        run = wandb.init()
        artifact = run.use_artifact(f"rohit-k-kesavan/smort/{model_ckpt}", type="model")
        artifact_dir = artifact.download()
        print(f"Checkpoint downloaded: {artifact_dir}")

    sample, dataset = get_random_sample_from_dataset(scene_id=scene_id)
    mean, std = dataset.get_mean_std()

    model = SMORT.load_from_checkpoint(
        f"artifacts/{model_ckpt}/model.ckpt",
        data_mean=mean,
        data_std=std,
    )

    encoded = model.text_encoder(sample["text_x_dict"])

    dists = encoded.unbind(1)
    mu, logvar = dists
    latent_vectors = mu
    motion = dataset.reverse_norm(
        model.motion_decoder(
            {
                "z": latent_vectors,
                "mask": sample["reactor_x_dict"]["mask"],
            },
            sample["actor_x_dict"],
        ).squeeze(dim=0)
    )

    return (
        motion,
        sample,
        dataset,
    )
