"""Sektion 6 — RL Optimizer Status panel (Popup/Sidebar).

Live display of the reinforcement-learning layout optimiser's training
progress, reward evolution, and best-layout statistics from Module D's
AutoMapper3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_d.mapper_3d import AutoMapper3D, VolumetricLayout
from ..module_d.gate_library import GateType, GateLibrary
from ..module_d.mapper import AutoMapper


class OptimiserState(Enum):
    """Training state of the RL optimiser."""
    IDLE = auto()
    RUNNING = auto()
    CONVERGED = auto()
    PAUSED = auto()


@dataclass
class TrainingLogEntry:
    """Single entry in the RL training log.

    Attributes:
        episode: Episode number.
        reward: Reward value from _compute_reward.
        wire_length_nm: Total wire length for this episode's layout.
        n_layers: Number of z-layers used.
    """

    episode: int
    reward: float
    wire_length_nm: float
    n_layers: int


@dataclass
class TrainingProgress:
    """RL optimiser training progress summary.

    Attributes:
        state: Current training state.
        current_episode: Current episode number.
        total_episodes: Total episodes to run.
        best_reward: Best reward seen so far.
        best_episode: Episode that achieved best reward.
        reward_history: Full reward history for plotting.
        best_wire_length_nm: Wire length of best layout.
        best_layer_count: Layer count of best layout.
        convergence_rate: Moving average reward slope.
    """

    state: OptimiserState
    current_episode: int
    total_episodes: int
    best_reward: float
    best_episode: int
    reward_history: list[float]
    best_wire_length_nm: float
    best_layer_count: int
    convergence_rate: float


class RLOptimizerPanel:
    """RL layout optimiser status panel.

    Wraps AutoMapper3D with progress tracking and training log
    for the dashboard sidebar/popup.
    """

    def __init__(
        self,
        library: GateLibrary | None = None,
        volume_nm: tuple[float, float, float] = (
            400_000_000.0, 400_000_000.0, 400_000_000.0
        ),
    ) -> None:
        self._library = library or GateLibrary()
        self._volume_nm = volume_nm
        self._mapper = AutoMapper3D(self._library, volume_nm)
        self._2d_mapper = AutoMapper(self._library)

        self._state = OptimiserState.IDLE
        self._log: list[TrainingLogEntry] = []
        self._reward_history: list[float] = []
        self._best_reward = -np.inf
        self._best_episode = 0
        self._best_layout: VolumetricLayout | None = None
        self._current_episode = 0
        self._total_episodes = 0

    @property
    def state(self) -> OptimiserState:
        return self._state

    @property
    def training_log(self) -> list[TrainingLogEntry]:
        return list(self._log)

    @property
    def best_layout(self) -> VolumetricLayout | None:
        return self._best_layout

    def run_optimisation(
        self,
        netlist: list[tuple[str, GateType, list[str]]],
        n_episodes: int = 500,
        layout_name: str = "rl-optimised",
        seed: int = 42,
    ) -> VolumetricLayout:
        """Run RL optimisation with episode-by-episode tracking.

        Records reward history and training log for dashboard display.
        """
        self._state = OptimiserState.RUNNING
        self._total_episodes = n_episodes
        self._log.clear()
        self._reward_history.clear()
        self._best_reward = -np.inf
        self._best_episode = 0

        rng = np.random.default_rng(seed)
        from ..module_d.mapper_3d import LayoutRLAgent, _euclidean_3d, _distance_to_boundary

        agent = LayoutRLAgent(self._volume_nm, grid_divisions=8)
        gate_ids = [gid for gid, _, _ in netlist]

        best_positions: dict[str, tuple[float, float, float]] = {}

        for ep in range(n_episodes):
            self._current_episode = ep + 1
            positions: dict[str, tuple[float, float, float]] = {}
            actions: list[tuple[int, int]] = []

            for gi, gid in enumerate(gate_ids):
                pi = agent.select_position(gi, rng)
                positions[gid] = agent.candidate_positions[pi]
                actions.append((gi, pi))

            reward = self._mapper._compute_reward(positions, netlist)

            # Track best
            if reward > self._best_reward:
                self._best_reward = reward
                self._best_episode = ep + 1
                best_positions = dict(positions)

            # Update Q-values
            for gi, pi in actions:
                agent.update(gi, pi, reward / len(gate_ids), 0.0)

            # Compute wire length for this episode
            total_wire = 0.0
            for gid, _, sources in netlist:
                if gid in positions:
                    for src in sources:
                        if src in positions:
                            total_wire += _euclidean_3d(positions[gid], positions[src])

            n_layers = len({pos[2] for pos in positions.values()})

            entry = TrainingLogEntry(
                episode=ep + 1,
                reward=reward,
                wire_length_nm=total_wire,
                n_layers=n_layers,
            )
            self._log.append(entry)
            self._reward_history.append(reward)

        # Build final layout from best positions
        self._best_layout = self._mapper.place_and_route_3d(
            netlist, layout_name, n_episodes=1
        )
        # Override positions with the best ones we found
        for pg in self._best_layout.placed_gates:
            if pg.gate_id in best_positions:
                pg.origin_nm = best_positions[pg.gate_id]

        self._state = OptimiserState.CONVERGED
        return self._best_layout

    def run_half_adder(self, n_episodes: int = 500) -> VolumetricLayout:
        """Convenience: optimise a half-adder layout."""
        netlist = self._2d_mapper.decompose_half_adder()
        return self.run_optimisation(netlist, n_episodes, "half-adder-rl")

    def run_full_adder(self, n_episodes: int = 500) -> VolumetricLayout:
        """Convenience: optimise a full-adder layout."""
        netlist = self._2d_mapper.decompose_full_adder()
        return self.run_optimisation(netlist, n_episodes, "full-adder-rl")

    def get_progress(self) -> TrainingProgress:
        """Get current training progress for dashboard display."""
        # Compute convergence rate from last 50 episodes
        convergence = 0.0
        if len(self._reward_history) >= 10:
            recent = self._reward_history[-50:]
            if len(recent) > 1:
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent, 1)[0]
                convergence = float(slope)

        best_wire = 0.0
        best_layers = 0
        if self._log:
            best_entry = max(self._log, key=lambda e: e.reward)
            best_wire = best_entry.wire_length_nm
            best_layers = best_entry.n_layers

        return TrainingProgress(
            state=self._state,
            current_episode=self._current_episode,
            total_episodes=self._total_episodes,
            best_reward=self._best_reward if self._best_reward > -np.inf else 0.0,
            best_episode=self._best_episode,
            reward_history=list(self._reward_history),
            best_wire_length_nm=best_wire,
            best_layer_count=best_layers,
            convergence_rate=convergence,
        )

    def get_last_log_entries(self, n: int = 10) -> list[TrainingLogEntry]:
        """Get the most recent training log entries."""
        return self._log[-n:]
