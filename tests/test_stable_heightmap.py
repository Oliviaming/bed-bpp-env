"""Tests for `heuristics.stable_heightmap.StableHeightmapHeuristic`."""

from pathlib import Path

import numpy as np
import pytest

from bed_bpp_env.data_model.item import Item
from bed_bpp_env.environment.sim_pal_env import SimPalEnv
from bed_bpp_env.heuristics.stable_heightmap import StableHeightmapHeuristic
from bed_bpp_env.io_utils import load_order_sequence

EXAMPLE_ORDERS_PATH = Path(__file__).parents[1] / "example_data" / "5_bed-bpp.json"
EURO_PALLET_SHAPE = (800, 1200)  # (y, x) in mm


def make_item(length: int, width: int, height: int, sequence: int = 1) -> Item:
    return Item(
        article=f"article-{sequence}",
        id=f"id-{sequence}",
        product_group="test",
        length_mm=length,
        width_mm=width,
        height_mm=height,
        weight_kg=5.0,
        sequence=sequence,
    )


def info_for(*items: Item) -> dict:
    return {"next_items_selection": list(items), "next_items_preview": []}


def support_ratio(heights: np.ndarray, action: dict) -> float:
    item: Item = action["item"]
    d_x, d_y = (item.length_mm, item.width_mm) if action["orientation"] == 0 else (item.width_mm, item.length_mm)
    footprint = heights[action["y"] : action["y"] + d_y, action["x"] : action["x"] + d_x]
    return float((footprint >= footprint.max() - 5).mean())


def test_first_item_goes_to_the_origin_corner_on_the_floor():
    heuristic = StableHeightmapHeuristic()
    action, successful = heuristic.getAction(np.zeros(EURO_PALLET_SHAPE, dtype=int), info_for(make_item(600, 400, 200)))

    assert successful
    assert (action["x"], action["y"]) == (0, 0)


def test_no_item_starts_above_the_maximum_base_height():
    heights = np.full(EURO_PALLET_SHAPE, 2100, dtype=int)
    heuristic = StableHeightmapHeuristic(max_base_height_mm=2000)

    action, successful = heuristic.getAction(heights, info_for(make_item(600, 400, 200)))

    assert not successful
    assert action["item"] is None


def test_placements_respect_the_minimum_support():
    # a single 300 x 300 block of height 500 mm, otherwise empty: the item may only be placed on the floor
    heights = np.zeros(EURO_PALLET_SHAPE, dtype=int)
    heights[0:300, 0:300] = 500
    heuristic = StableHeightmapHeuristic(min_support=0.8)

    action, successful = heuristic.getAction(heights, info_for(make_item(600, 400, 200)))

    assert successful
    assert support_ratio(heights, action) >= 0.8


def test_stops_if_no_safe_placement_exists():
    # narrow pillars everywhere: no placement has enough support
    heights = np.zeros(EURO_PALLET_SHAPE, dtype=int)
    heights[:, ::100] = 300
    heuristic = StableHeightmapHeuristic(min_support=0.5)

    _, successful = heuristic.getAction(heights, info_for(make_item(600, 400, 200)))

    assert not successful


def test_ignores_placeholders_in_the_selection():
    heuristic = StableHeightmapHeuristic()
    item = make_item(600, 400, 200)

    action, successful = heuristic.getAction(np.zeros(EURO_PALLET_SHAPE, dtype=int), info_for(None, item))

    assert successful
    assert action["item"] == item


def test_rejects_targets_that_do_not_fit_the_grid():
    heuristic = StableHeightmapHeuristic(grid_mm=7)
    with pytest.raises(ValueError):
        heuristic.getAction(np.zeros(EURO_PALLET_SHAPE, dtype=int), info_for(make_item(600, 400, 200)))


def test_planned_bottom_heights_match_the_environment():
    """On the planning grid, the bottom height of each placement equals the one the environment computes."""
    order = load_order_sequence(EXAMPLE_ORDERS_PATH)[0]
    env = SimPalEnv()
    env._n_item_preview, env._n_item_selection = 3, 2
    observation, info = env.reset(order_sequence=[order])
    heuristic = StableHeightmapHeuristic(min_support=0.5, lookahead_top_n=4, w_area=1.0)

    done, n_placed = False, 0
    while not done:
        action, successful = heuristic.getAction(observation, info)
        if not successful:
            break
        planned = heuristic.best_candidate(observation, [action["item"]])
        grid = heuristic._coarse_heights(observation)
        d_x, d_y = heuristic._footprint_cells(action["item"], action["orientation"])
        planned_z = grid[action["y"] // 5 : action["y"] // 5 + d_y, action["x"] // 5 : action["x"] // 5 + d_x].max()
        assert planned is not None

        observation, _, done, info = env.step(action)
        n_placed += 1
        placed = env._actions[-1]
        assert placed["flb_coordinates"][2] == planned_z
        assert placed["flb_coordinates"][2] <= 2000

    assert n_placed > 0
