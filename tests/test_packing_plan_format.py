"""Checks the format of a packing plan, using the example plan shipped in `example_data/`.

A packing plan maps an order id to a list of actions. Each action places one item of that order:

    {"item": {<item exactly as in the order>}, "flb_coordinates": [x, y, z], "orientation": 0 | 1}

`flb_coordinates` is the item's front-left-bottom corner in millimeters; orientation `1` swaps length and width.
Note that the format does not forbid placements that overhang the target; the example plan contains such placements.
"""

import json
from pathlib import Path

import pytest

from bed_bpp_env.data_model.packing_plan import PackingPlan
from bed_bpp_env.io_utils import load_packing_plan_sequence

EXAMPLE_DATA_DIR = Path(__file__).parents[1] / "example_data"
ORDERS_PATH = EXAMPLE_DATA_DIR / "5_bed-bpp.json"
PACKING_PLAN_PATH = EXAMPLE_DATA_DIR / "packing_plan_5-bed-bpp.json"

ACTION_KEYS = {"item", "flb_coordinates", "orientation"}
ORIENTATIONS = {0, 1}


@pytest.fixture(scope="module")
def orders() -> dict:
    with open(ORDERS_PATH) as file:
        return json.load(file)


@pytest.fixture(scope="module")
def raw_packing_plans() -> dict:
    with open(PACKING_PLAN_PATH) as file:
        return json.load(file)


def test_packing_plans_refer_to_known_orders(raw_packing_plans, orders):
    assert raw_packing_plans, "packing plan file is empty"
    assert set(raw_packing_plans) <= set(orders)


def test_actions_have_expected_schema(raw_packing_plans):
    for order_id, actions in raw_packing_plans.items():
        assert isinstance(actions, list), order_id
        for index, action in enumerate(actions):
            assert set(action) == ACTION_KEYS, (order_id, index)
            assert action["orientation"] in ORIENTATIONS, (order_id, index)

            flb = action["flb_coordinates"]
            assert len(flb) == 3, (order_id, index)
            assert all(isinstance(value, int) and value >= 0 for value in flb), (order_id, index)


def test_each_action_places_an_item_of_its_order_at_most_once(raw_packing_plans, orders):
    for order_id, actions in raw_packing_plans.items():
        order_items = orders[order_id]["item_sequence"]
        placed_sequences = []
        for action in actions:
            item = action["item"]
            assert order_items[str(item["sequence"])] == item, (order_id, item["sequence"])
            placed_sequences.append(item["sequence"])

        assert len(placed_sequences) == len(set(placed_sequences)), order_id
        assert len(placed_sequences) <= len(order_items), order_id


def test_deserialization_round_trip(raw_packing_plans):
    packing_plans = load_packing_plan_sequence(PACKING_PLAN_PATH)

    assert [plan.id for plan in packing_plans] == list(raw_packing_plans)
    for plan in packing_plans:
        assert isinstance(plan, PackingPlan)
        assert plan.to_dict() == {plan.id: raw_packing_plans[plan.id]}
