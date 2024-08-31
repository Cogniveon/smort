import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, List
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import torch
from smort.rifke import canonicalize_rotation

logger = logging.getLogger("matplotlib.animation")
logger.setLevel(logging.ERROR)

KINEMATIC_TREES = {
    "smplxjoints": [
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
    ],
    "smpljoints": [
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
    ],
    "guoh3djoints": [  # no hands
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
    ],
}

@dataclass
class SceneRenderer:
    jointstype: str = "smplxjoints"
    fps: float = 20.0
    colors: List[Iterable[str]] = field(
        default_factory=lambda: [
            ("green", "green", "green", "green", "green"),
            ("red", "red", "red", "red", "red"),
            ("blue", "blue", "blue", "blue", "blue"),
            ("magenta", "magenta", "magenta", "magenta", "magenta"),
            ("black", "black", "black", "black", "black")
        ]
    )
    figsize: Tuple[int, int] = (4, 4)
    fontsize: int = 15
    canonicalize: bool = False

    def render_animation(
        self,
        joints_input: list[np.ndarray | torch.Tensor],
        highlights_list: Optional[List[np.ndarray]] = None,
        title: str = "",
        output: str = "notebook",
        jointstype: str = "smpljoints",
        agg: bool = True,
    ):
        if agg:
            import matplotlib
            matplotlib.use("Agg")

        jointstype = jointstype if jointstype is not None else self.jointstype
        assert jointstype in KINEMATIC_TREES
        kinematic_tree = KINEMATIC_TREES[jointstype]

        x, y, z = 0, 1, 2

        joints_list = []
        for joints in joints_input:
            if not isinstance(joints, np.ndarray):
                joints = joints.detach().cpu().numpy()
            joints_list.append(joints)
        if self.canonicalize:
            joints_list = [canonicalize_rotation(joints, jointstype=jointstype) for joints in joints_list]

        fig = plt.figure(figsize=self.figsize)
        ax = self.init_axis(fig, title)

        trajectories = [joints[:, 0, [x, y]] for joints in joints_list]
        avg_segment_length = (
            np.mean(np.linalg.norm(np.diff(trajectories[0], axis=0), axis=1)) + 1e-3
        )
        draw_offset = int(25 / avg_segment_length)

        spline_lines = [ax.plot(*trajectory.T, zorder=10)[0] for trajectory in trajectories]

        all_joints = np.concatenate(joints_list, axis=1)
        minx, miny, _ = np.min(all_joints, axis=(0, 1))
        maxx, maxy, _ = np.max(all_joints, axis=(0, 1))
        self.plot_floor(ax, minx, maxx, miny, maxy, 0)

        height_offsets = [np.min(joints[:, :, z]) for joints in joints_list]
        for joints, offset in zip(joints_list, height_offsets):
            joints[:, :, z] -= offset

        lines_list = [[] for _ in range(len(joints_list))]
        initialized = False

        def update(frame):
            nonlocal initialized

            for idx, (joints, lines, colors) in enumerate(zip(joints_list, lines_list, self.colors)):
                skeleton = joints[frame]

                root = skeleton[0]
                self.update_camera(ax, root)

                hcolors = colors
                if highlights_list and highlights_list[idx] is not None and highlights_list[idx][frame]:
                    hcolors = ["yellow"] * 5

                for i, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(list(hcolors)))):
                    if not initialized:
                        lines.append(
                            ax.plot(
                                skeleton[chain, x],
                                skeleton[chain, y],
                                skeleton[chain, z],
                                linewidth=6.0,
                                color=color,
                                zorder=20,
                                path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                            )[0]
                        )
                    else:
                        lines[i].set_xdata(skeleton[chain, x])
                        lines[i].set_ydata(skeleton[chain, y])
                        lines[i].set_3d_properties(skeleton[chain, z])
                        lines[i].set_color(color)

                left = max(frame - draw_offset, 0)
                right = min(frame + draw_offset, trajectories[idx].shape[0])

                spline_lines[idx].set_xdata(trajectories[idx][left:right, 0])
                spline_lines[idx].set_ydata(trajectories[idx][left:right, 1])
                spline_lines[idx].set_3d_properties(np.zeros_like(trajectories[idx][left:right, 0]))

            initialized = True
            return []

        frames = min(joints.shape[0] for joints in joints_list)
        anim = FuncAnimation(fig, update, frames=frames, interval=1000 / self.fps, repeat=False)

        if output == "notebook":
            from IPython.display import HTML, display
            display(HTML(anim.to_jshtml()))
        else:
            anim.save(output, fps=int(self.fps))

        plt.close()

    @staticmethod
    def init_axis(fig, title, radius=1.5):
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.view_init(elev=20.0, azim=-60)

        fact = 2
        ax.set_xlim3d([-radius / fact, radius / fact])
        ax.set_ylim3d([-radius / fact, radius / fact])
        ax.set_zlim3d([0, radius])

        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_axis_off()
        ax.grid(b=False)

        ax.set_title(title, loc="center", wrap=True)
        return ax

    @staticmethod
    def plot_floor(ax, minx, maxx, miny, maxy, minz):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        verts = [
            [minx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts], zorder=1)
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
        ax.add_collection3d(xz_plane)

        radius = max((maxx - minx), (maxy - miny))

        minx_all = (maxx + minx) / 2 - radius
        maxx_all = (maxx + minx) / 2 + radius

        miny_all = (maxy + miny) / 2 - radius
        maxy_all = (maxy + miny) / 2 + radius

        verts = [
            [minx_all, miny_all, minz],
            [minx_all, maxy_all, minz],
            [maxx_all, maxy_all, minz],
            [maxx_all, miny_all, minz],
        ]
        xz_plane = Poly3DCollection([verts], zorder=1)
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        return ax

    @staticmethod
    def update_camera(ax, root, radius=1.5):
        fact = 2
        ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
        ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])
