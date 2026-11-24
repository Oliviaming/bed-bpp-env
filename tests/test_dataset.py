"""Sanity checks for the local copy of the BED-BPP dataset in `data/bed-bpp_v1.json`.

The dataset is not tracked by Git (see `.gitignore`). Download it from
https://floriankagerer.github.io/assets/bed-bpp/bed-bpp_v1.json into `data/`; otherwise these tests are skipped.
"""

import hashlib
import json
from pathlib import Path

import pytest

from bed_bpp_env.data_model.order import Order
from bed_bpp_env.io_utils import load_order_sequence

DATASET_PATH = Path(__file__).parents[1] / "data" / "bed-bpp_v1.json"
DATASET_SHA256 = "6ecc91d92b9ce88de113ac66c45a450ac3eec303018f14cd73e4bd0eaf8eb3cc"
"""Checksum of the published `bed-bpp_v1.json` (identical to `example_data/benchmark_data/bed-bpp_v1.json`)."""

TARGET_BASE_AREA_MM = {"euro-pallet": (1200, 800), "rollcontainer": (800, 700)}
ITEM_KEYS = {"article", "id", "product_group", "length/mm", "width/mm", "height/mm", "weight/kg", "sequence"}
PROPERTY_KEYS = {"id", "order_nr", "type", "target"}

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason=f"dataset not found at {DATASET_PATH}")


@pytest.fixture(scope="module")
def raw_dataset() -> dict:
    with open(DATASET_PATH) as file:
        return json.load(file)


def all_items(dataset: dict):
    for order in dataset.values():
        yield from order["item_sequence"].values()


def test_file_matches_published_checksum():
    digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert digest == DATASET_SHA256


def test_dataset_size_matches_publication(raw_dataset):
    items = list(all_items(raw_dataset))
    assert len(raw_dataset) == 10_003
    assert len(items) == 428_718
    assert len({item["article"] for item in items}) == 2_621
    assert len({(item["length/mm"], item["width/mm"], item["height/mm"]) for item in items}) == 974


def test_every_order_has_expected_schema(raw_dataset):
    for order_id, order in raw_dataset.items():
        assert set(order) == {"item_sequence", "properties"}, order_id
        assert set(order["properties"]) == PROPERTY_KEYS, order_id
        assert order["properties"]["target"] in TARGET_BASE_AREA_MM, order_id
        assert len(order["item_sequence"]) > 0, order_id


def test_item_sequence_is_contiguous_and_matches_sequence_field(raw_dataset):
    for order_id, order in raw_dataset.items():
        keys = list(order["item_sequence"])
        assert keys == [str(i) for i in range(1, len(keys) + 1)], order_id
        for key, item in order["item_sequence"].items():
            assert item["sequence"] == int(key), (order_id, key)


def test_items_have_valid_fields(raw_dataset):
    for item in all_items(raw_dataset):
        assert set(item) == ITEM_KEYS
        for dimension in ("length/mm", "width/mm", "height/mm"):
            assert isinstance(item[dimension], int) and item[dimension] > 0
        assert item["weight/kg"] > 0


def test_every_item_fits_on_its_target_in_some_orientation(raw_dataset):
    for order_id, order in raw_dataset.items():
        size_x, size_y = TARGET_BASE_AREA_MM[order["properties"]["target"]]
        for item in order["item_sequence"].values():
            length, width = item["length/mm"], item["width/mm"]
            fits = (length <= size_x and width <= size_y) or (width <= size_x and length <= size_y)
            assert fits, (order_id, item["sequence"])


def test_load_order_sequence_deserializes_all_orders(raw_dataset):
    order_sequence = load_order_sequence(DATASET_PATH)

    assert [order.id for order in order_sequence] == list(raw_dataset)
    assert all(isinstance(order, Order) for order in order_sequence)

    first = order_sequence[0]
    raw_first = raw_dataset[first.id]
    assert len(first.item_sequence) == len(raw_first["item_sequence"])
    assert first.properties.target == raw_first["properties"]["target"]
    assert first.properties.order_number == raw_first["properties"]["order_nr"]
    assert first.item_sequence[0].length_mm == raw_first["item_sequence"]["1"]["length/mm"]
