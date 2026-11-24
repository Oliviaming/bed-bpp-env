"""Tests for the port of the O3DBP-3-2 heuristic (and `SimPalEnv`) to the `Order`/`Item` data model.

The regression test replays one order of `example_data/5_bed-bpp.json` and compares the resulting placements with
`tests/fixtures/o3dbp_3_2_reference_00100408.json`, which was produced by the last dict-based implementation
(commit 09ad773) with `run_heuristic_O3DBP_3_2.py`.

`PalletizingEnvironment` is not imported here: its visualization module requires the `--data` argument at import time.
Its ported preview logic is identical to the one of `SimPalEnv`, which is tested below.
"""

import json
from pathlib import Path

import pytest

from bed_bpp_env.data_model.item import Item
from bed_bpp_env.data_model.order import Order, Properties
from bed_bpp_env.environment.sim_pal_env import SimPalEnv
from bed_bpp_env.heuristics.o3dbp_3_2 import O3DBP_3_2
from bed_bpp_env.io_utils import load_order_sequence

PREVIEW, SELECTION = 3, 2

REPO_ROOT = Path(__file__).parents[1]
EXAMPLE_ORDERS_PATH = REPO_ROOT / "example_data" / "5_bed-bpp.json"
REFERENCE_PATH = Path(__file__).parent / "fixtures" / "o3dbp_3_2_reference_00100408.json"


def make_order(order_id: str = "order-1", n_items: int = 4) -> Order:
    items = [
        Item(
            article=f"article-{i}",
            id=f"id-{i}",
            product_group="test",
            length_mm=400,
            width_mm=300,
            height_mm=200,
            weight_kg=5.0,
            sequence=i,
        )
        for i in range(1, n_items + 1)
    ]
    properties = Properties(id=order_id, order_number=order_id, type="test", target="euro-pallet")
    return Order(id=order_id, item_sequence=items, properties=properties)


def make_sim_env(preview: int = PREVIEW, selection: int = SELECTION) -> SimPalEnv:
    env = SimPalEnv()
    env._n_item_preview = preview
    env._n_item_selection = selection
    return env


def place(item: Item) -> dict:
    return {"x": 0, "y": 0, "orientation": 0, "item": item}


def test_sim_env_selection_and_preview_follow_item_sequence():
    order = make_order(n_items=4)
    i1, i2, i3, i4 = order.item_sequence
    env = make_sim_env()

    _, info = env.reset(order_sequence=[order])
    assert info["order_id"] == order.id
    assert info["next_items_selection"] == [i1, i2]
    assert info["next_items_preview"] == [i3]
    assert set(info["corner_points"]) == {i1.article, i2.article}

    # select the second item first
    _, _, done, info = env.step(place(i2))
    assert not done
    assert info["next_items_selection"] == [i1, i3]
    assert info["next_items_preview"] == [i4]

    _, _, done, info = env.step(place(i1))
    assert not done
    assert info["next_items_selection"] == [i3, i4]
    assert info["next_items_preview"] == [None]

    _, _, done, info = env.step(place(i3))
    assert not done
    assert info["next_items_selection"] == [i4, None]
    assert info["next_items_preview"] == [None]
    assert set(info["corner_points"]) == {i4.article}

    _, _, done, info = env.step(place(i4))
    assert done
    assert info["next_items_selection"] == []
    assert env.getNPlacedItems() == 4


def test_sim_env_reset_advances_through_order_sequence():
    first, second = make_order("first", 2), make_order("second", 2)
    env = make_sim_env()

    _, info = env.reset(order_sequence=[first, second])
    assert info["order_id"] == "first"

    _, info = env.reset()
    assert info["order_id"] == "second"
    assert info["all_orders_considered"] is False

    _, info = env.reset()
    assert info["all_orders_considered"] is True
    assert info["next_items_selection"] == []


def test_copy_without_orders_keeps_state_but_drops_orders():
    order = make_order(n_items=4)
    env = make_sim_env()
    env.reset(order_sequence=[order, make_order("other", 2)])
    env.step(place(order.item_sequence[0]))

    copied = env.copy_without_orders()

    assert copied._orders == []
    assert len(env._orders) == 2
    assert copied._current_order == env._current_order
    assert copied._items_selection == env._items_selection
    assert (copied._target_space.getHeights() == env._target_space.getHeights()).all()
    assert copied.getNPlacedItems() == env.getNPlacedItems() == 1


def test_heuristic_reproduces_reference_placements():
    with open(REFERENCE_PATH) as file:
        reference = json.load(file)

    order = next(order for order in load_order_sequence(EXAMPLE_ORDERS_PATH) if order.id == reference["order_id"])
    env, sim_env = make_sim_env(), make_sim_env()
    observation, info = env.reset(order_sequence=[order])
    sim_env.reset(order_sequence=[order])

    heuristic = O3DBP_3_2(preview=PREVIEW, selection=SELECTION)
    heuristic.setSimEnv(sim_env)

    done = False
    while not done:
        action, successful = heuristic.getAction(observation, info)
        assert successful
        observation, _, done, info = env.step(action)

    placements = [
        [action["item"].sequence, *action["flb_coordinates"], action["orientation"]] for action in env._actions
    ]
    assert placements == reference["placements"]


@pytest.mark.parametrize("n_items", [1, 2])
def test_heuristic_handles_orders_shorter_than_preview(n_items):
    order = make_order(n_items=n_items)
    env, sim_env = make_sim_env(), make_sim_env()
    observation, info = env.reset(order_sequence=[order])
    sim_env.reset(order_sequence=[order])

    heuristic = O3DBP_3_2(preview=PREVIEW, selection=SELECTION)
    heuristic.setSimEnv(sim_env)

    for _ in range(n_items):
        action, successful = heuristic.getAction(observation, info)
        assert successful
        observation, _, done, info = env.step(action)

    assert done
    assert sorted(action["item"].sequence for action in env._actions) == list(range(1, n_items + 1))
