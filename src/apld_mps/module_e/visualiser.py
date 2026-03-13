"""Detail 6.1 — Real-time visualisation of E-field and polariton density.

Renders |ψ(r,t)|² and E-field as colour-coded 2D/3D animations during
simulation, allowing the user to watch light flow and wave interference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """Single animation frame.

    Attributes:
        time_ps: Simulation time [ps].
        field: 2D array of field values.
        label: Description (e.g. "Ez", "|ψ|²").
    """

    time_ps: float
    field: np.ndarray
    label: str = ""


class FieldVisualiser:
    """Accumulate simulation snapshots for animated 2D field visualisation.

    Stores a sequence of FrameData objects that can be rendered with
    matplotlib or exported for external rendering.
    """

    def __init__(self, title: str = "APLD-MPS Simulation") -> None:
        self.title = title
        self.frames: list[FrameData] = []

    def capture_frame(
        self,
        field: np.ndarray,
        time_ps: float,
        label: str = "",
    ) -> None:
        self.frames.append(FrameData(time_ps=time_ps, field=field.copy(), label=label))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def get_intensity_range(self) -> tuple[float, float]:
        """Global min/max across all frames for consistent colour scaling."""
        if not self.frames:
            return (0.0, 1.0)
        vmin = min(float(np.min(f.field)) for f in self.frames)
        vmax = max(float(np.max(f.field)) for f in self.frames)
        return (vmin, vmax)

    def render_matplotlib(self, output_path: str | None = None) -> None:
        """Render frames as a matplotlib animation.

        If *output_path* is given (e.g. ``"sim.gif"``), saves to file.
        Otherwise displays interactively.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not self.frames:
            raise ValueError("No frames captured.")

        vmin, vmax = self.get_intensity_range()

        fig, ax = plt.subplots()
        im = ax.imshow(
            self.frames[0].field,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
        )
        cb = fig.colorbar(im, ax=ax)
        time_text = ax.set_title(f"{self.title}  t = {self.frames[0].time_ps:.2f} ps")

        def update(frame_idx: int):
            frame = self.frames[frame_idx]
            im.set_data(frame.field)
            ax.set_title(
                f"{self.title}  t = {frame.time_ps:.2f} ps  {frame.label}"
            )
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames), blit=True)

        if output_path:
            anim.save(output_path, writer="pillow", fps=30)
        else:
            plt.show()

        plt.close(fig)
