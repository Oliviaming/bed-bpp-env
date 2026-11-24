"""Tests for `heuristics.load_stability.PileModel`."""

from bed_bpp_env.heuristics.load_stability import FLOOR, PileModel


def test_box_on_the_floor_has_the_floor_as_support():
    pile = PileModel()
    box = pile.add((0, 0, 400, 300), 0, 200, 5.0)

    assert box.supports == [(FLOOR, (0, 0, 400, 300))]


def test_centered_box_on_top_is_stable():
    pile = PileModel()
    pile.add((0, 0, 400, 300), 0, 200, 5.0)

    margin = pile.min_stability_margin((50, 50, 350, 250), 200, 400, 5.0)

    # the support hull is the contact area (the top box's bottom); its center (200, 150) is 100 mm from the edge
    assert margin == 100


def test_box_whose_center_is_beyond_the_support_tips_over():
    pile = PileModel()
    pile.add((0, 0, 400, 300), 0, 200, 5.0)

    # 600 mm long box resting on x in [250, 400]: its center (x = 550) is outside of the support
    margin = pile.min_stability_margin((250, 0, 850, 300), 200, 400, 5.0)

    assert margin < 0


def test_box_without_support_is_rejected():
    pile = PileModel()
    pile.add((0, 0, 400, 300), 0, 200, 5.0)

    assert pile.min_stability_margin((500, 0, 900, 300), 200, 400, 5.0) < 0


def test_leaning_tower_is_detected_although_each_box_is_stable_on_its_own():
    pile = PileModel()
    pile.add((0, 0, 400, 300), 0, 200, 5.0)  # base on the floor
    # each box is shifted by 150 mm relative to the one below: its own center lies within its support
    pile.add((150, 0, 550, 300), 200, 400, 5.0)
    assert pile.min_stability_margin((150, 0, 550, 300), 200, 400, 5.0) > 0

    # the next box is stable relative to the box below, but the combined load of both upper boxes acts
    # at x = (350 + 500) / 2 = 425, beyond the support of the middle box (x in [150, 400])
    margin = pile.min_stability_margin((300, 0, 700, 300), 400, 600, 5.0)

    assert margin < 0


def test_column_slenderness_of_a_stack_of_single_supported_boxes():
    pile = PileModel()
    pile.add((0, 0, 400, 250), 0, 250, 5.0)
    pile.add((0, 0, 400, 250), 250, 500, 5.0)

    # third box on top: column from z = 0 to 750 mm, narrowest contact side 250 mm
    assert pile.column_slenderness((0, 0, 400, 250), 500, 750) == 3.0


def test_column_slenderness_is_zero_on_the_floor_and_for_interlocking_boxes():
    pile = PileModel()
    pile.add((0, 0, 400, 300), 0, 200, 5.0)
    pile.add((400, 0, 800, 300), 0, 200, 5.0)

    assert pile.column_slenderness((900, 0, 1100, 300), 0, 200) == 0.0  # on the floor
    assert pile.column_slenderness((200, 0, 600, 300), 200, 400) == 0.0  # bridges two boxes


def test_contacts_ignore_boxes_that_are_lower_than_the_tolerance():
    pile = PileModel(contact_tolerance_mm=5)
    pile.add((0, 0, 400, 300), 0, 200, 5.0)
    pile.add((400, 0, 800, 300), 0, 190, 5.0)

    contacts = pile.contacts((200, 0, 600, 300), 200)

    assert [index for index, _ in contacts] == [0]
