import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger("matplotlib.animation")
logger.setLevel(logging.ERROR)


@dataclass
class SceneRenderer:
    fps: float = 20.0
    colors: List[str] = field(
        default_factory=lambda: ["red", "green", "blue", "yellow", "black"]
    )
    figsize: Tuple[int, int] = (4, 4)
    fontsize: int = 15

    def render_animation(
        self,
        motions: List[np.ndarray | torch.Tensor],
        title: str = "",
        output: str = "notebook",
        agg: bool = True,
    ):
        if agg:
            import matplotlib

            matplotlib.use("Agg")

        kinematic_tree = [
            [0, 3, 6, 9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
        ]
        x, y, z = 0, 1, 2

        motions = [
            (
                motion.detach().cpu().numpy()
                if isinstance(motion, torch.Tensor)
                else motion
            )
            for motion in motions
        ]
        assert type(motions[0]) == np.ndarray

        fig = plt.figure(figsize=self.figsize)
        ax = self.init_axis(fig, title)

        trajectories = [joints[:, 0, [x, y]] for joints in motions]
        avg_segment_length = (
            np.mean(np.linalg.norm(np.diff(trajectories[0], axis=0), axis=1)) + 1e-3
        )
        draw_offset = int(25 / avg_segment_length)

        spline_lines = [
            ax.plot(*trajectory.T, zorder=10, color=self.colors[i])[0]
            for i, trajectory in enumerate(trajectories)
        ]
        all_motions = np.concatenate(motions, axis=1)
        minx, miny, _ = np.min(all_motions, axis=(0, 1))
        maxx, maxy, _ = np.max(all_motions, axis=(0, 1))
        self.plot_floor(ax, minx, maxx, miny, maxy, 0)

        height_offsets = [np.min(joints[:, :, z]) for joints in motions]
        for joints, offset in zip(motions, height_offsets):
            joints[:, :, z] -= offset

        skel_list: list[list[Line2D]] = [[] for _ in range(len(motions))]
        initialized = False

        def update(frame):
            nonlocal initialized
            mean_root = np.zeros_like(motions[0][0, 0])

            for idx, (motion, skel, color) in enumerate(
                zip(motions, skel_list, self.colors)
            ):
                joints = motion[frame]

                mean_root += joints[0]

                for i, chain in enumerate(reversed(kinematic_tree)):
                    if not initialized:
                        skel.append(
                            ax.plot(
                                joints[chain, x],
                                joints[chain, y],
                                joints[chain, z],
                                linewidth=6.0,
                                color=color,
                                zorder=20,
                                path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                            )[0]
                        )
                    else:
                        skel[i].set_xdata(joints[chain, x])
                        skel[i].set_ydata(joints[chain, y])
                        skel[i].set_3d_properties(joints[chain, z])  # type: ignore
                        skel[i].set_color(color)

                left = max(frame - draw_offset, 0)
                right = min(frame + draw_offset, trajectories[idx].shape[0])

                spline_lines[idx].set_xdata(trajectories[idx][left:right, 0])
                spline_lines[idx].set_ydata(trajectories[idx][left:right, 1])
                spline_lines[idx].set_3d_properties(  # type: ignore
                    np.zeros_like(trajectories[idx][left:right, 0])
                )

            mean_root /= len(motions)

            self.update_camera(ax, mean_root)
            initialized = True
            return []

        frames = min(joints.shape[0] for joints in motions)
        anim = FuncAnimation(
            fig, update, frames=frames, interval=1000 / self.fps, repeat=False
        )

        if output == "notebook":
            from IPython.display import HTML, display

            display(HTML(anim.to_jshtml()))
        else:
            anim.save(output, fps=int(self.fps))

        plt.close()


    def render_image(
        self,
        motions: List[np.ndarray | torch.Tensor],
        frame_skip: int = 20,
        title: str = "",
        output: str = "notebook",
        agg: bool = True,
    ):
        if agg:
            import matplotlib

            matplotlib.use("Agg")
        kinematic_tree = [
            [0, 3, 6, 9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
        ]
        x, y, z = 0, 1, 2

        motions = [
            (
                motion.detach().cpu().numpy()
                if isinstance(motion, torch.Tensor)
                else motion
            )
            for motion in motions
        ]
        assert type(motions[0]) == np.ndarray

        fig = plt.figure(figsize=self.figsize)
        plt.ion()
        ax = self.init_axis(fig, title)

        trajectories = [joints[:, 0, [x, y]] for joints in motions]

        spline_lines = [
            ax.plot(*trajectory.T, zorder=10, color=self.colors[i])[0]
            for i, trajectory in enumerate(trajectories)
        ]
        all_motions = np.concatenate(motions, axis=1)
        minx, miny, _ = np.min(all_motions, axis=(0, 1))
        maxx, maxy, _ = np.max(all_motions, axis=(0, 1))
        mean_root = np.mean(all_motions, axis=(0))[0]
        self.plot_floor(ax, minx, maxx, miny, maxy, 0)
        self.update_camera(ax, mean_root)

        height_offsets = [np.min(joints[:, :, z]) for joints in motions]
        for joints, offset in zip(motions, height_offsets):
            joints[:, :, z] -= offset
        
        # num_frames = motions[0].shape[0]
        # for frame_idx in reversed(range(0, num_frames, num_frames // 10)):
        #     print(frame_idx)
        

        # Display or save the image
        if output == "notebook":
            from IPython.display import display

            plt.show()
        else:
            fig.savefig(output)

        plt.close()


    @staticmethod
    def init_axis(fig: Figure, title, radius=1.5):
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        assert type(ax) == Axes3D
        ax.view_init(elev=20.0, azim=-60)

        fact = 2
        ax.set_xlim3d([-radius / fact, radius / fact])
        ax.set_ylim3d([-radius / fact, radius / fact])
        ax.set_zlim3d([0, radius])

        ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])  # type: ignore

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
