"""
A simple static stability model of a pile of boxes, used by `StableHeightmapHeuristic`.

Each box rests on the floor and/or on other boxes. The weight of a box and the loads on top of it are passed down to
its supporters, split in proportion to the contact areas, and applied at the point of the contact rectangle that is
closest to the resultant above. A box is considered stable if its resultant (own weight plus loads, at their center)
lies inside the convex hull of its contact areas, with a safety margin.

All coordinates are in millimeters; `x` and `y` are the horizontal axes.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FLOOR = -1
"""Index of the floor as supporter."""

Rect = tuple[float, float, float, float]
"""An axis-parallel rectangle `(x0, y0, x1, y1)`."""


@dataclass
class Box:
    """A placed box and its contacts with the boxes (or the floor) below."""

    rect: Rect
    z0: float
    z1: float
    mass: float
    supports: list[tuple[int, Rect]] = field(default_factory=list)
    """Pairs of (index of the supporting box or `FLOOR`, contact rectangle)."""

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.rect
        return (x0 + x1) / 2, (y0 + y1) / 2


def _area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _intersection(a: Rect, b: Rect) -> Optional[Rect]:
    rect = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


def _clamp(point: tuple[float, float], rect: Rect) -> tuple[float, float]:
    return min(max(point[0], rect[0]), rect[2]), min(max(point[1], rect[1]), rect[3])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain; returns the hull in counter-clockwise order."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _depth_inside(point: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    """Distance of the point to the hull's boundary; negative if outside (or if the hull is degenerate)."""
    if len(hull) < 3:
        return -1.0
    depth = np.inf
    for k in range(len(hull)):
        (ax, ay), (bx, by) = hull[k], hull[(k + 1) % len(hull)]
        edge_length = np.hypot(bx - ax, by - ay)
        # signed distance to the edge line, positive on the left side (inside for a counter-clockwise hull)
        distance = ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)) / edge_length
        depth = min(depth, distance)
    return float(depth)


class PileModel:
    """The boxes of one pile and their static stability."""

    def __init__(self, contact_tolerance_mm: float = 5.0) -> None:
        self.boxes: list[Box] = []
        self._tolerance = contact_tolerance_mm

    def contacts(self, rect: Rect, z0: float) -> list[tuple[int, Rect]]:
        """The supporters of a box with bottom area `rect` at height `z0` and the contact rectangles."""
        if z0 <= self._tolerance:
            return [(FLOOR, rect)]
        contacts = []
        for index, box in enumerate(self.boxes):
            if z0 - self._tolerance <= box.z1 <= z0:
                contact = _intersection(rect, box.rect)
                if contact is not None:
                    contacts.append((index, contact))
        return contacts

    def add(self, rect: Rect, z0: float, z1: float, mass: float) -> Box:
        box = Box(rect=rect, z0=z0, z1=z1, mass=mass, supports=self.contacts(rect, z0))
        self.boxes.append(box)
        return box

    def min_stability_margin(
        self, rect: Rect, z0: float, z1: float, mass: float, margin_per_m: float = 0.0
    ) -> float:
        """
        Places a box tentatively and returns the smallest margin (mm) of all affected boxes, i.e., the new box and
        all boxes below it: the distance of a box's resultant to the boundary of its support hull. Negative values
        mean that a box would tip over.

        With `margin_per_m`, a box's margin is reduced by `margin_per_m` * (height of its bottom in m), i.e., boxes
        high up in the pile need a larger margin. Slender, tall stacks sway in the simulation.
        """
        new_box = Box(rect=rect, z0=z0, z1=z1, mass=mass, supports=self.contacts(rect, z0))
        if not new_box.supports:
            return -1.0
        boxes = self.boxes + [new_box]
        affected = self._ancestors(boxes, len(boxes) - 1)
        margins = self._margins(boxes, affected)
        return min(margin - margin_per_m * boxes[k].z0 / 1000.0 for k, margin in margins.items())

    def column_slenderness(self, rect: Rect, z0: float, z1: float) -> float:
        """
        Height-to-width ratio of the column that a new box would complete. A column is a chain of boxes that each
        rest on exactly one box below, i.e., they do not interlock with neighbors. Its height reaches from the
        bottom of the chain's lowest box to the top of the new box, and its width is the smallest side of the contact
        areas in the chain. Returns `0` if the new box rests on the floor or on more than one box.

        Slender columns are statically stable, but they sway and topple in the rigid body simulation.
        """
        supports = self.contacts(rect, z0)
        if len(supports) != 1 or supports[0][0] == FLOOR:
            return 0.0

        index, contact = supports[0]
        width = min(contact[2] - contact[0], contact[3] - contact[1])
        while True:
            box = self.boxes[index]
            if len(box.supports) != 1 or box.supports[0][0] == FLOOR:
                break
            index, contact = box.supports[0]
            width = min(width, contact[2] - contact[0], contact[3] - contact[1])

        return (z1 - self.boxes[index].z0) / width

    @staticmethod
    def _ancestors(boxes: list[Box], index: int) -> set[int]:
        """The box and all boxes that (transitively) support it."""
        result, stack = set(), [index]
        while stack:
            k = stack.pop()
            if k in result or k == FLOOR:
                continue
            result.add(k)
            stack.extend(s for s, _ in boxes[k].supports)
        return result

    @staticmethod
    def _margins(boxes: list[Box], indices: set[int]) -> dict[int, float]:
        """Computes the resultant of every box (top-down) and returns the stability margin of the given boxes."""
        n = len(boxes)
        children: list[list[tuple[int, Rect]]] = [[] for _ in range(n)]
        for k, box in enumerate(boxes):
            for s, contact in box.supports:
                if s != FLOOR:
                    children[s].append((k, contact))

        mass = np.zeros(n)
        point = np.zeros((n, 2))
        for k in sorted(range(n), key=lambda k: -boxes[k].z0):  # boxes above first
            box = boxes[k]
            total = box.mass
            moment = np.array(box.center) * box.mass
            for c, contact in children[k]:
                child = boxes[c]
                share = _area(contact) / sum(_area(r) for _, r in child.supports)
                load = share * mass[c]
                total += load
                moment += load * np.array(_clamp(tuple(point[c]), contact))
            mass[k] = total
            point[k] = moment / total

        margins = {}
        for k in indices:
            if all(s == FLOOR for s, _ in boxes[k].supports):
                # loads act within the footprint, so a box resting on the floor cannot tip over
                margins[k] = np.inf
                continue
            hull_points = [(r[ix], r[iy]) for _, r in boxes[k].supports for ix, iy in ((0, 1), (2, 1), (2, 3), (0, 3))]
            margins[k] = _depth_inside(tuple(point[k]), _convex_hull(hull_points))
        return margins
