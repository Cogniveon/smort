import math
import random
import hydra
import numpy as np
import torch
import h5py
from hydra.utils import instantiate
from omegaconf import DictConfig
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from smort.data.collate import collate_x_dict, length_to_mask
from smort.data.data_module import InterXDataModule
from smort.models.smort import SMORT
from smort.models.text_encoder import TextToEmb
from smort.rifke import feats_to_joints
from smort.utils import plot_floor, plot_joints, plot_trajectory


device: torch.device
text_model: TextToEmb
data_module: InterXDataModule
model: SMORT
app: Dash
sample: dict
motion: np.ndarray | None

@hydra.main(config_path="configs", config_name="app", version_base="1.3")
def run_server(cfg: DictConfig):
    global device, text_model, data_module, model, app
    
    device = torch.device(cfg.device)
    text_model = instantiate(cfg.text_encoder, device=device)
    
    data_module = instantiate(cfg.data, return_scene_text=True)
    data_module.setup("fit")
    mean, std = data_module.dataset.get_mean_std()
    if not cfg.ckpt:
        model = instantiate(cfg.model, data_mean=mean, data_std=std)
    else:
        model = SMORT.load_from_checkpoint(
            cfg.ckpt, data_mean=mean, data_std=std
        )
    
    app = Dash(
        __name__, external_scripts=[{"src": "https://cdn.tailwindcss.com"}]
    )
    
    def get_scene_ids():
        with h5py.File(cfg.data.dataset_file, "r") as hdf5_file:
            motions_dataset = hdf5_file["motions"]
            assert type(motions_dataset) == h5py.Group
            for key in motions_dataset.keys():
                yield str(key)
    
    app.layout = html.Div(
        className="flex flex-col min-h-screen gap-2 px-2",
        children=[
            html.H1(
                children="SMORT Inference",
                className="text-center text-4xl text-slate-700 py-5",
            ),
            dcc.Dropdown(
                list(get_scene_ids()),
                None,
                id="selected-scene",
                className="w-full",
            ),
            html.Div(
                children=[
                    dcc.Textarea(
                        value="",
                        id="scene-text",
                        className="flex-1 rounded border border-gray-300 p-2",
                    ),
                    html.Button(
                        children="Run Inference",
                        id="infer-btn",
                        className="bg-green-500 text-white rounded px-3 py-2",
                    ),
                ],
                className="flex items-center justify-center gap-2",
            ),
            html.Div(
                children=[
                    dcc.Graph(id="gt"),
                    dcc.Graph(id="pred"),
                ],
                className="flex items-center justify-center gap-2",
            ),
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
    app.run(debug=True)


@callback(
    Output("scene-text", "value"),
    Input("selected-scene", "value"),
)
def update_scene(value: str | None):
    global sample
    if value is None:
        return ""
    sample = data_module.get_sample(value)
    assert type(sample["text"]) == list
    return random.choice(sample["text"])


@callback(
    Output("gt", "figure"),
    Input("frame-slider", "value"),
    prevent_initial_call=True,
)
def update_gt(frame_idx: int):
    fig = go.Figure()

    reactor_feats = data_module.dataset.reverse_norm(sample["reactor_x_dict"]["x"])
    actor_feats = data_module.dataset.reverse_norm(sample["actor_x_dict"]["x"])
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

    plot_trajectory(fig, motions[0], 0.0, "red")
    plot_joints(fig, motions[0], frame_idx, "reactor", "red")
    plot_trajectory(fig, motions[1], 0.0, "blue")
    plot_joints(fig, motions[1], frame_idx, "actor", "blue")

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


@callback(
    Output("pred", "figure"),
    Input("frame-slider", "value"),
    prevent_initial_call=True,
)
def update_pred(frame_idx: int):
    fig = go.Figure()

    try:
        if motion is None:
            return fig
    except:
        raise PreventUpdate
    
    actor_feats = data_module.dataset.reverse_norm(sample["actor_x_dict"]["x"])
    # import pdb; pdb.set_trace()

    motions = [
        feats_to_joints(torch.from_numpy(motion)).detach().cpu().numpy(),
        feats_to_joints(torch.from_numpy(actor_feats)).detach().cpu().numpy(),
    ]

    num_frames = motions[0].shape[0]
    frame_idx = min(math.floor(num_frames * frame_idx / 600), num_frames - 1)

    xrange, yrange, zrange = plot_floor(fig, motions, minz=0.0)
    height_offsets = [np.min(joints[:, :, 2]) for joints in motions]
    for joints, offset in zip(motions, height_offsets):
        joints[:, :, 2] -= offset

    plot_trajectory(fig, motions[0], 0.0, "red")
    plot_joints(fig, motions[0], frame_idx, "pred", "red")
    plot_trajectory(fig, motions[1], 0.0, "blue")
    plot_joints(fig, motions[1], frame_idx, "actor", "blue")

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



# @callback(
#     [
#         Input("pred-motion", "data"),
#         Input("scene", "data"),
#         Input("frame-slider", "value"),
#         Input("dataset-file", "data"),
#     ],
# )
# def update_pred(pred: dict | None, sample: dict | None, frame_idx: int, dataset_file: str):
#     if pred is None or sample is None:
#         raise PreventUpdate
    


@callback(
    Input("scene-text", "value"),
    Input("infer-btn", "n_clicks"),
    running=[
        (Output("scene-text", "disabled"), True, False),
        (Output("infer-btn", "disabled"), True, False),
    ]
)
def run_infer(text: str, n_clicks: int | None):
    global motion
    if n_clicks is None or sample is None or text is None:
        raise PreventUpdate
    
    text_embeddings = text_model.forward(text, device=device) # type: ignore
    encoded = model.text_encoder(collate_x_dict([text_embeddings]))
    dists = encoded.unbind(1)
    mu, logvar = dists
    latent_vectors = mu
    
    motion = data_module.dataset.reverse_norm(
        model.motion_decoder(
            {
                "z": latent_vectors,
                "mask": length_to_mask(torch.tensor([sample["reactor_x_dict"]['length']])),
            },
            collate_x_dict([sample["actor_x_dict"]]),
        ).squeeze(dim=0)
    )


if __name__ == "__main__":
    run_server()
