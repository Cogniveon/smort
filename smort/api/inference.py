import math
import random
from typing import Optional

import h5py
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, callback, dcc, html

from smort.data.data_module import InterXDataModule
from smort.data.text_motion_dataset import TextMotionDataset
from smort.rifke import feats_to_joints


class App:
    dataset_file: str
    app: Dash

    def __init__(self, dataset_file: str) -> None:
        self.dataset_file = dataset_file
        self.app = Dash(
            __name__, external_scripts=[{"src": "https://cdn.tailwindcss.com"}]
        )
        self.init_layout()

    def get_scene_ids(self):
        with h5py.File(self.dataset_file, "r") as hdf5_file:
            motions_dataset = hdf5_file["motions"]
            assert type(motions_dataset) == h5py.Group
            for key in motions_dataset.keys():
                yield str(key)

    def init_layout(self):
        layout = html.Div(
            className="flex flex-col min-h-screen gap-2 px-2",
            children=[
                html.H1(
                    children="SMORT Inference",
                    className="text-center text-4xl text-slate-700 py-5",
                ),
                dcc.Store(id="dataset-file", data=self.dataset_file),
                dcc.Store(id="scene", data=None),
                dcc.Dropdown(
                    list(self.get_scene_ids()),
                    None,
                    id="selected-scene",
                    className="w-full",
                ),
                dcc.Textarea(
                    value="", disabled=True, id="scene-text", className="w-full"
                ),
                dcc.Graph(id="gt"),
                dcc.Slider(
                    0,
                    599,
                    step=1,
                    marks=None,
                    value=0,
                    updatemode="drag",
                    id="frame-slider",
                ),
            ],
        )

        self.app.layout = layout

    def run(self, **kwargs):
        self.app.run(**kwargs)


def plot_floor(fig: go.Figure, joints: list[np.ndarray], minz: float = 0.0):
    all_joints = np.concatenate(joints, axis=1)
    minx, miny, _ = np.min(all_joints, axis=(0, 1))
    maxx, maxy, _ = np.max(all_joints, axis=(0, 1))

    # Vertices for the solid floor
    verts_solid = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz],
    ]

    # Vertices for the translucent floor
    radius = max((maxx - minx), (maxy - miny))
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius
    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts_translucent = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz],
    ]

    # Solid floor
    fig.add_trace(
        go.Mesh3d(
            x=[v[0] for v in verts_solid],
            y=[v[1] for v in verts_solid],
            z=[v[2] for v in verts_solid],
            color="gray",
            opacity=0.9,
            name="floor_1",
            showlegend=False,
            # flatshading=True,
            # alphahull=0
        )
    )

    # Translucent floor
    fig.add_trace(
        go.Mesh3d(
            x=[v[0] for v in verts_translucent],
            y=[v[1] for v in verts_translucent],
            z=[v[2] for v in verts_translucent],
            color="gray",
            opacity=0.5,
            name="floor_2",
            showlegend=False,
            # flatshading=True,
            # alphahull=0
        )
    )

    return [minx_all, maxx_all], [miny_all, maxy_all], [minz - 0.5, minz + 3]


def plot_joints(
    fig: go.Figure,
    joints: np.ndarray,
    joint_idx: list[int],
    frame_idx: int,
    color="red",
):
    joints_pos = joints[frame_idx, joint_idx, :]

    fig.add_trace(
        go.Scatter3d(
            x=joints_pos[:, 0],
            y=joints_pos[:, 1],
            z=joints_pos[:, 2],
            mode="lines+markers",
            line=dict(width=7, color="black"),
            marker=dict(size=3, color=color),
            showlegend=False,
        )
    )


def plot_trajectory(fig: go.Figure, joints: np.ndarray, minz: float = 0.0, color="red"):
    trajectory = joints[:, 0, [0, 1]]

    fig.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=np.zeros_like(trajectory[:, 0]) + minz,
            mode="lines",
            line=dict(width=4, color=color),
            showlegend=False,
        )
    )
    # left = max(frame_idx - draw_offset, 0)
    # right = min(frame_idx + draw_offset, trajectories[pid].shape[0])

    # spline_lines[pid].set_xdata(trajectories[pid][left:right, 0])
    # spline_lines[pid].set_ydata(trajectories[pid][left:right, 1])
    # spline_lines[pid].set_3d_properties(  # type: ignore
    #     np.zeros_like(trajectories[pid][left:right, 0])
    # )


@callback(
    [
        Output("scene", "data"),
        Output("scene-text", "value"),
    ],
    [
        Input("dataset-file", "data"),
        Input("selected-scene", "value"),
    ],
)
def update_scene(dataset_file: str, value: str | None):
    if value is None:
        return None, ""

    data_module = InterXDataModule(
        dataset_file, batch_size=1, return_scene=True, return_scene_text=True
    )
    data_module.setup("predict")
    sample = data_module.get_sample(value)
    assert type(sample["text"]) == list
    return sample, random.choice(sample["text"])


@callback(
    Output("gt", "figure"),
    [
        Input("scene", "data"),
        # Input("frame-slider", "value"),
        Input("frame-slider", "value"),
        Input("dataset-file", "data"),
    ],
)
def update_gt_graph(sample: dict | None, frame_idx: int, dataset_file: str):
    fig = go.Figure()

    if sample is None:
        return fig

    dataset = TextMotionDataset(dataset_file)

    reactor_feats = dataset.reverse_norm(np.array(sample["reactor_x_dict"]["x"]))
    actor_feats = dataset.reverse_norm(np.array(sample["actor_x_dict"]["x"]))
    # import pdb; pdb.set_trace()

    motions = [
        feats_to_joints(torch.from_numpy(reactor_feats)).detach().cpu().numpy(),
        feats_to_joints(torch.from_numpy(actor_feats)).detach().cpu().numpy(),
    ]

    num_frames = motions[0].shape[0]
    frame_idx = min(math.floor(num_frames * frame_idx / 600), num_frames - 1)

    xrange, yrange, zrange = plot_floor(fig, motions, minz=0.0)
    height_offsets = [np.min(joints[:, :, 2]) for joints in motions]
    for joints, offset in zip(motions, height_offsets):
        joints[:, :, 2] -= offset

    for j, color in zip(motions, ["red", "blue"]):

        plot_trajectory(fig, j, 0.0, color)
        # print(j[frame_idx, 10, :])
        for chain in [
            [0, 9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
        ]:
            plot_joints(fig, j, chain, frame_idx, color)

    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.1, y=1.1, z=0.1),
            ),
            xaxis=dict(showgrid=False, showline=False, range=xrange),
            yaxis=dict(showgrid=False, showline=False, range=yrange),
            zaxis=dict(showgrid=False, showline=False, range=zrange),
            aspectratio=dict(x=1, y=1, z=1),
        ),
    )

    return fig
