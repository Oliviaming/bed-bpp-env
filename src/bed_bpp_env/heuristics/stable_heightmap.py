"""
A stability-first heuristic for the online 3D bin packing tasks O3DBP-k-s, designed for the evaluation of BED-BPP.

The evaluation counts an item only if the whole pile stays stable in the Blender rigid body simulation and the
bottom of the item is not above 2 m. A single falling item makes the whole order worthless. Hence, this heuristic

1. only considers placements that are physically safe (enough direct support, supported center of mass), and never
   starts an item above `max_base_height_mm`; if no safe placement exists, it stops the order,
2. among the safe placements, it packs densely by preferring low placements that waste little space below the item,
   touch neighbors or walls, and keep the surface flat, and
3. optionally looks one step ahead: it simulates the best placements and prefers those after which the items that
   become selectable next still have good safe placements.

All computations use a coarse grid (default 5 mm) of the height map. Item positions are multiples of the grid size
and item lengths/widths are rounded up to the grid. Then, the height of a grid cell equals the height of its first
millimeter, so the planned bottom height of an item is exactly the one the environment computes.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bed_bpp_env.data_model.item import Item
from bed_bpp_env.heuristics.load_stability import PileModel

logger = logging.getLogger(__name__)

N_ORIENTATIONS = 2
CHUNK_SIZE = 256
"""Number of candidate positions that are evaluated at once (limits memory)."""


@dataclass
class Candidate:
    """A safe placement on the planning grid and its score (lower is better)."""

    score: float
    item: Item
    orientation: int
    i: int
    """Index of the first cell in x-direction."""
    j: int
    """Index of the first cell in y-direction."""
    d_x: int
    """Number of cells in x-direction."""
    d_y: int
    """Number of cells in y-direction."""
    z_mm: int
    """Height of the item's bottom."""


class StableHeightmapHeuristic:
    """
    Stability-first height map heuristic. See the module documentation.

    The default values of the parameters were tuned for the task O3DBP-3-2 on 20 orders of BED-BPP with the Blender
    stability check; see `experiments/algorithm_comparison.ipynb`.

    Parameters.
    -----------
    grid_mm: int
        The resolution of the height map that is used for planning. Positions are multiples of it.
    min_support: float
        Minimum ratio of the item's bottom area that rests directly on the surface below (within
        `support_tolerance_mm`), in `[0, 1]`.
    support_tolerance_mm: int
        Cells whose height is at most this value below the item's bottom count as support.
    require_com_support: bool
        Require support in each quadrant of the item's bottom, which implies that the center of mass lies within the
        convex hull of the supporting area.
    max_base_height_mm: Optional[int]
        Maximum height of the bottom of an item. Items above 2 m never count in the evaluation.
    w_base, w_waste, w_position, w_rise, w_contact, w_flush: float
        Weights of the score components (lower score is better):
        `w_base` * bottom height [m], `w_waste` * mean gap below the item [dm], `w_position` * normalized distance
        to the origin corner, `w_rise` * increase of the pile's maximum height [m], `- w_contact` * share of the
        item's perimeter that touches walls or higher neighbors, `- w_flush` * share of the perimeter where the
        neighbors' height equals the item's top.
    w_area: float
        Bonus for large items: `- w_area` * item bottom area / target area. Large items that are postponed often
        find no safe placement later.
    w_age: float
        Bonus for items that have been selectable for a long time: `- w_age` * (newest visible sequence number -
        item's sequence number) / 10. An item that is never chosen blocks a slot of the selection.
    lookahead_top_n: int
        Number of best placements that are re-ranked with a one-step lookahead (`0` disables the lookahead).
    lookahead_weight: float
        Weight of the best next score in the lookahead.
    dead_end_penalty: float
        Added in the lookahead if none of the next items has a safe placement.
    load_check: bool
        Check the static stability of the whole pile with `load_stability.PileModel`: the resultant of each box and
        the loads on it must lie inside its support hull. This detects leaning towers whose boxes are stable on
        their own. The heuristic keeps a model of the boxes it placed in the current order; if the model does not
        match the observed height map, the check is switched off for the rest of the order.
    stability_margin_mm: float
        Minimum distance of a resultant to the boundary of the support hull.
    stability_margin_per_m: float
        Additional margin (mm) per meter of height of a box's bottom: tall stacks sway in the simulation.
    max_column_slenderness: Optional[float]
        Maximum height-to-width ratio of a column of boxes that each rest on exactly one box below (see
        `PileModel.column_slenderness`); `None` disables the check. Requires `load_check`.
    w_slenderness: float
        Soft version of the column limit: the score of a placement increases by `w_slenderness` * (column
        slenderness - `free_slenderness`), if positive. Requires `load_check`.
    free_slenderness: float
        Column slenderness without penalty.
    load_check_top_n: int
        Number of best placements that are checked with the load model.
    load_check_max_candidates: int
        If none of the best `load_check_top_n` placements is stable, further placements are checked in the order of
        their scores, up to this number, before the order is stopped.
    """

    def __init__(
        self,
        grid_mm: int = 5,
        min_support: float = 0.3,
        support_tolerance_mm: int = 5,
        require_com_support: bool = True,
        max_base_height_mm: Optional[int] = 2000,
        w_base: float = 1.0,
        w_waste: float = 1.0,
        w_position: float = 0.1,
        w_rise: float = 0.5,
        w_contact: float = 1.0,
        w_flush: float = 0.2,
        w_area: float = 1.0,
        w_age: float = 0.0,
        lookahead_top_n: int = 8,
        lookahead_weight: float = 1.0,
        dead_end_penalty: float = 2.0,
        load_check: bool = True,
        stability_margin_mm: float = 30.0,
        stability_margin_per_m: float = 0.0,
        max_column_slenderness: Optional[float] = None,
        w_slenderness: float = 0.0,
        free_slenderness: float = 2.0,
        load_check_top_n: int = 64,
        load_check_max_candidates: int = 2048,
    ) -> None:
        self._grid = grid_mm
        self._min_support = min_support
        self._support_tolerance = support_tolerance_mm
        self._require_com_support = require_com_support
        self._max_base_height = max_base_height_mm
        self._weights = {
            "base": w_base,
            "waste": w_waste,
            "position": w_position,
            "rise": w_rise,
            "contact": w_contact,
            "flush": w_flush,
            "area": w_area,
            "age": w_age,
        }
        self._lookahead_top_n = lookahead_top_n
        self._lookahead_weight = lookahead_weight
        self._dead_end_penalty = dead_end_penalty
        self._load_check = load_check
        self._stability_margin = stability_margin_mm
        self._margin_per_m = stability_margin_per_m
        self._max_slenderness = max_column_slenderness
        self._w_slenderness = w_slenderness
        self._free_slenderness = free_slenderness
        self._load_check_top_n = load_check_top_n
        self._load_check_max_candidates = load_check_max_candidates
        logger.info(f"StableHeightmapHeuristic: {self.__dict__}")

        self._pile: Optional[PileModel] = None
        """Model of the boxes placed in the current order (only used with `load_check`)."""
        self._pile_valid = False
        """Indicates whether the pile model matches the observed height map."""

    def getAction(self, observation: np.ndarray, info: dict) -> tuple[dict, bool]:
        """
        Returns the action with the best score among all safe placements of the selectable items.

        Returns.
        --------
        action: dict
            `{"x": int, "y": int, "orientation": int, "item": Item}`, or values `None` if no safe placement exists.
        successful: bool
            `False` if no safe placement exists; the order should be stopped then.
        """
        heights = self._coarse_heights(observation)
        selection = [item for item in info.get("next_items_selection", []) if item is not None]
        preview = [item for item in info.get("next_items_preview", []) if item is not None]
        use_load_check = self._load_check and self._sync_pile(observation, heights)

        n_best = max(1, self._lookahead_top_n)
        if use_load_check:
            candidates = self._stable_candidates(heights, selection, n_best)
        else:
            candidates = self._best_candidates(heights, selection, n_best)
        if not candidates:
            logger.info("no safe placement found -> stop the order")
            return {"x": None, "y": None, "orientation": None, "item": None}, False

        best = candidates[0]
        if self._lookahead_top_n > 0 and len(candidates) > 1:
            best = min(candidates, key=lambda c: self._lookahead_score(heights, c, selection, preview))

        if use_load_check:
            # the environment places the item at exactly the planned position (see module documentation)
            self._pile.add(*self._box_geometry(best))

        action = {"x": best.i * self._grid, "y": best.j * self._grid, "orientation": best.orientation, "item": best.item}
        logger.info(f"selected action: {action} (z={best.z_mm}, score={best.score:.4f})")
        return action, True

    def best_candidate(self, observation: np.ndarray, items: list[Optional[Item]]) -> Optional[Candidate]:
        """Returns the best safe placement (without lookahead) of any of the given items, or `None`."""
        candidates = self._best_candidates(self._coarse_heights(observation), [i for i in items if i is not None], 1)
        return candidates[0] if candidates else None

    # helpers

    def _box_geometry(self, candidate: Candidate) -> tuple[tuple[float, float, float, float], float, float, float]:
        """The exact bottom rectangle, bottom and top height, and mass of the candidate's item."""
        item = candidate.item
        length, width = (
            (item.length_mm, item.width_mm) if candidate.orientation == 0 else (item.width_mm, item.length_mm)
        )
        x0, y0 = candidate.i * self._grid, candidate.j * self._grid
        return (x0, y0, x0 + length, y0 + width), candidate.z_mm, candidate.z_mm + item.height_mm, item.weight_kg

    def _stable_candidates(self, heights: np.ndarray, selection: list[Item], n_best: int) -> list[Candidate]:
        """
        Returns up to `n_best` candidates that pass the load check, in the order of their scores. The best
        `load_check_top_n` candidates are checked first; only if none of them passes, the search continues down the
        ranking (up to `load_check_max_candidates`) until the first stable candidate is found.
        """
        ranked = self._best_candidates(heights, selection, self._load_check_max_candidates)
        stable = []
        for rank, candidate in enumerate(ranked):
            if rank >= self._load_check_top_n and stable:
                break
            if self._is_pile_stable(candidate):
                stable.append(candidate)
                if self._w_slenderness == 0 and len(stable) >= n_best:
                    break

        if self._w_slenderness > 0:
            # soft preference for interlocking placements over slender columns
            for candidate in stable:
                slenderness = self._pile.column_slenderness(*self._box_geometry(candidate)[:3])
                candidate.score += self._w_slenderness * max(0.0, slenderness - self._free_slenderness)
            stable.sort(key=lambda c: c.score)
        return stable[:n_best]

    def _is_pile_stable(self, candidate: Candidate) -> bool:
        rect, z0, z1, mass = self._box_geometry(candidate)
        if self._max_slenderness is not None and self._pile.column_slenderness(rect, z0, z1) > self._max_slenderness:
            return False
        margin = self._pile.min_stability_margin(rect, z0, z1, mass, margin_per_m=self._margin_per_m)
        return margin >= self._stability_margin

    def _sync_pile(self, observation: np.ndarray, heights: np.ndarray) -> bool:
        """Starts a new pile model for an empty target and checks that the model matches the observed heights."""
        if not observation.any():
            self._pile = PileModel(contact_tolerance_mm=self._support_tolerance)
            self._pile_valid = True
            return True
        if not self._pile_valid or self._pile is None:
            return False

        rendered = np.zeros_like(heights)
        for box in self._pile.boxes:
            x0, y0, x1, y1 = box.rect
            rendered[
                int(y0) // self._grid : math.ceil(y1 / self._grid), int(x0) // self._grid : math.ceil(x1 / self._grid)
            ] = box.z1
        if not np.array_equal(rendered, heights):
            logger.warning("pile model does not match the observed heights -> no load check for this order")
            self._pile_valid = False
        return self._pile_valid

    def _lookahead_score(
        self, heights: np.ndarray, candidate: Candidate, selection: list[Item], preview: list[Item]
    ) -> float:
        """The candidate's score plus the weighted best score of the items that are selectable after placing it."""
        next_heights = heights.copy()
        j, i = candidate.j, candidate.i
        next_heights[j : j + candidate.d_y, i : i + candidate.d_x] = candidate.z_mm + candidate.item.height_mm

        remaining = list(selection)
        remaining.remove(candidate.item)
        next_selectable = remaining + preview[:1]
        if not next_selectable:
            return candidate.score

        next_best = self._best_candidates(next_heights, next_selectable, 1)
        if not next_best:
            return candidate.score + self._dead_end_penalty
        return candidate.score + self._lookahead_weight * next_best[0].score

    def _best_candidates(self, heights: np.ndarray, items: list[Item], n_best: int) -> list[Candidate]:
        """Returns the `n_best` safe placements of the given items, sorted by score."""
        target_area = heights.size * self._grid**2
        newest_sequence = max(item.sequence for item in items) if items else 0

        candidates = []
        for item in items:
            item_bonus = (
                self._weights["area"] * item.length_mm * item.width_mm / target_area
                + self._weights["age"] * (newest_sequence - item.sequence) / 10.0
            )
            for orientation in range(N_ORIENTATIONS):
                for candidate in self._placements(heights, item, orientation, n_best):
                    candidate.score -= item_bonus
                    candidates.append(candidate)
        candidates.sort(key=lambda c: c.score)
        return candidates[:n_best]

    def _coarse_heights(self, observation: np.ndarray) -> np.ndarray:
        """The height map on the planning grid; a cell's height is the height of its first millimeter."""
        size_y, size_x = observation.shape
        if size_x % self._grid or size_y % self._grid:
            raise ValueError(f"target size {size_x}x{size_y} mm is not a multiple of the grid size {self._grid} mm")
        return observation[:: self._grid, :: self._grid].astype(np.int64)

    def _footprint_cells(self, item: Item, orientation: int) -> tuple[int, int]:
        """The size `(cells_x, cells_y)` of the item's bottom on the planning grid (rounded up)."""
        length, width = (item.length_mm, item.width_mm) if orientation == 0 else (item.width_mm, item.length_mm)
        return math.ceil(length / self._grid), math.ceil(width / self._grid)

    def _placements(self, heights: np.ndarray, item: Item, orientation: int, n_best: int) -> list[Candidate]:
        """Evaluates all candidate positions of the item in the given orientation; returns the `n_best` safe ones."""
        n_y, n_x = heights.shape
        d_x, d_y = self._footprint_cells(item, orientation)
        if d_x > n_x or d_y > n_y:
            return []

        # bottom height of the item for each position (sliding window maximum)
        windows = sliding_window_view(heights, (d_y, d_x))
        base = sliding_window_view(heights, d_x, axis=1).max(axis=-1)
        base = sliding_window_view(base, d_y, axis=0).max(axis=-1)  # shape (n_y - d_y + 1, n_x - d_x + 1)

        positions = self._anchored_positions(heights, d_x, d_y)
        if self._max_base_height is not None:
            positions = positions[base[positions[:, 0], positions[:, 1]] <= self._max_base_height]
        if len(positions) == 0:
            return []

        current_max = int(heights.max())
        size_cells = n_x + n_y
        padded = np.pad(heights, 1, constant_values=-1)  # -1 marks outside of the target (walls)

        all_scores, all_positions, all_z = [], [], []
        for start in range(0, len(positions), CHUNK_SIZE):
            chunk = positions[start : start + CHUNK_SIZE]
            js, is_ = chunk[:, 0], chunk[:, 1]
            z = base[js, is_]  # (n,)
            win = windows[js, is_]  # (n, d_y, d_x)

            supported = win >= (z - self._support_tolerance)[:, None, None]
            feasible = supported.mean(axis=(1, 2)) >= self._min_support - 1e-9
            if self._require_com_support:
                feasible &= self._quadrants_supported(supported)
            if not feasible.any():
                continue
            js, is_, z, win = js[feasible], is_[feasible], z[feasible], win[feasible]

            mean_gap_mm = (z[:, None, None] - win).mean(axis=(1, 2))
            top = z + item.height_mm
            rise = np.maximum(0, top - current_max)
            contact, flush = self._perimeter_features(padded, js, is_, d_x, d_y, z, top)
            position = (is_ + js) / size_cells

            w = self._weights
            score = (
                w["base"] * z / 1000.0
                + w["waste"] * mean_gap_mm / 100.0
                + w["position"] * position
                + w["rise"] * rise / 1000.0
                - w["contact"] * contact
                - w["flush"] * flush
            )
            all_scores.append(score)
            all_positions.append(np.stack([js, is_], axis=1))
            all_z.append(z)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        positions = np.concatenate(all_positions)
        zs = np.concatenate(all_z)
        order = np.argsort(scores, kind="stable")[:n_best]
        return [
            Candidate(
                score=float(scores[k]),
                item=item,
                orientation=orientation,
                i=int(positions[k, 1]),
                j=int(positions[k, 0]),
                d_x=d_x,
                d_y=d_y,
                z_mm=int(zs[k]),
            )
            for k in order
        ]

    @staticmethod
    def _anchored_positions(heights: np.ndarray, d_x: int, d_y: int) -> np.ndarray:
        """
        Positions where a side of the item is aligned with a wall or a step of the height map, both in x- and in
        y-direction. A step counts if it lies along the item's side, e.g., for the left side, the height changes
        between the column left of the item and the item's first column in one of the item's rows. This covers
        items pushed against neighbors and items aligned with the edges of the items below.

        Returns an array of `(j, i)` = `(y, x)` indices of the item's first cell.
        """
        n_y, n_x = heights.shape

        # edges_x[r, c]: height step (or wall) at the boundary left of column c in row r, c in [0, n_x]
        edges_x = np.ones((n_y, n_x + 1), dtype=np.int32)
        edges_x[:, 1:n_x] = heights[:, 1:] != heights[:, :-1]
        # any step within the d_y rows of a position: window sums of a cumulative sum over rows
        cum = np.concatenate([np.zeros((1, n_x + 1), dtype=np.int32), np.cumsum(edges_x, axis=0)])
        steps_x = (cum[d_y:] - cum[:-d_y]) > 0  # (n_y - d_y + 1, n_x + 1)
        anchored_x = steps_x[:, : n_x - d_x + 1] | steps_x[:, d_x:]  # left side | right side

        edges_y = np.ones((n_y + 1, n_x), dtype=np.int32)
        edges_y[1:n_y, :] = heights[1:, :] != heights[:-1, :]
        cum = np.concatenate([np.zeros((n_y + 1, 1), dtype=np.int32), np.cumsum(edges_y, axis=1)], axis=1)
        steps_y = (cum[:, d_x:] - cum[:, :-d_x]) > 0  # (n_y + 1, n_x - d_x + 1)
        anchored_y = steps_y[: n_y - d_y + 1, :] | steps_y[d_y:, :]  # front side | back side

        return np.argwhere(anchored_x & anchored_y)

    @staticmethod
    def _quadrants_supported(supported: np.ndarray) -> np.ndarray:
        """True if each quadrant of the bottom has a supported cell (center of mass within the support hull)."""
        _, d_y, d_x = supported.shape
        h_y, h_x = (d_y + 1) // 2, (d_x + 1) // 2  # the middle row/column belongs to both halves
        low_y, high_y = slice(0, h_y), slice(d_y - h_y, d_y)
        low_x, high_x = slice(0, h_x), slice(d_x - h_x, d_x)
        ok = np.ones(supported.shape[0], dtype=bool)
        for sy in (low_y, high_y):
            for sx in (low_x, high_x):
                ok &= supported[:, sy, sx].any(axis=(1, 2))
        return ok

    def _perimeter_features(
        self,
        padded: np.ndarray,
        js: np.ndarray,
        is_: np.ndarray,
        d_x: int,
        d_y: int,
        z: np.ndarray,
        top: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns, for each candidate, the share of the perimeter cells that (a) are a wall or higher than the item's
        bottom (contact) and (b) have the same height as the item's top (flush surface).
        """
        # in the padded map, the footprint of candidate (j, i) is [j + 1, j + 1 + d_y) x [i + 1, i + 1 + d_x)
        columns = sliding_window_view(padded, (d_y, 1))
        rows = sliding_window_view(padded, (1, d_x))
        left = columns[js + 1, is_][..., 0]
        right = columns[js + 1, is_ + 1 + d_x][..., 0]
        below = rows[js, is_ + 1][:, 0, :]
        above = rows[js + 1 + d_y, is_ + 1][:, 0, :]
        perimeter = np.concatenate([left, right, below, above], axis=1)  # (n, 2 * (d_x + d_y))

        is_wall = perimeter < 0
        contact = (is_wall | (perimeter > z[:, None])).mean(axis=1)
        flush = (np.abs(perimeter - top[:, None]) <= self._support_tolerance).mean(axis=1)
        return contact, flush
