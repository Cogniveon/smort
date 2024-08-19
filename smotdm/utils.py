import os
import torch
import numpy as np
from tqdm.auto import tqdm


def loop_interx(base_dir: str, exclude: list[str] = []):
    pbar = tqdm(os.listdir(f'{base_dir}/motions'))
    for scene_id in pbar:
        with open(
            f'{base_dir}/texts/{str(scene_id)}.txt',
            'r'
        ) as annotation:
            texts = []
            for line in annotation.readlines():
                texts.append(line.rstrip('\n'))
            
            num_frames = 0
            motions = []
            for file in [
                f'{base_dir}/motions/{scene_id}/P1.npz',
                f'{base_dir}/motions/{scene_id}/P2.npz',
            ]:
                data = np.load(file)
                num_frames += data['pose_body'].shape[0]
                motions.append({
                    'body_pose': torch.tensor(data['pose_body'], dtype=torch.float32),
                    'left_hand_pose': torch.tensor(data['pose_lhand'], dtype=torch.float32),
                    'right_hand_pose': torch.tensor(data['pose_rhand'], dtype=torch.float32),
                    'transl': torch.tensor(data['trans'], dtype=torch.float32),
                    'global_orient': torch.tensor(data['root_orient'], dtype=torch.float32),
                })
            
            num_frames = num_frames // len(motions)
            pbar.set_postfix(dict(scene_id=scene_id), num_frames=num_frames)
            yield scene_id, num_frames, motions, texts
