"""
Similar to `PalletizingEnvironment`, but with little changes for the tasks with preview and selection.
"""

import configparser
import copy
import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from bed_bpp_env.data_model.item import Item
from bed_bpp_env.data_model.order import Order
from bed_bpp_env.environment import MAXHEIGHT_OBSERVATION_SPACE, SIZE_EURO_PALLET, SIZE_ROLLCONTAINER
from bed_bpp_env.environment.cuboid import Cuboid
from bed_bpp_env.environment.space_3d import Space3D
from bed_bpp_env.evaluation.kpis import KPIs
from bed_bpp_env.utils import PARSEDARGUMENTS
from bed_bpp_env.utils.configuration import USEDCONFIGURATIONFILE

logger = logging.getLogger(__name__)


RENDER = PARSEDARGUMENTS.get("visualize", False)

conf = configparser.ConfigParser()
conf.read(USEDCONFIGURATIONFILE)


class SimPalEnv(gym.Env):
    """
    Similar to `PalletizingEnvironment`, but with little changes for the tasks with preview and selection.

    It has no render methods, so that it can be copied cheaply and pickled (needed for `multiprocessing`). Heuristics
    use copies of this environment to simulate the outcome of actions before applying an action to the
    `PalletizingEnvironment`.

    Items are `Item` objects. Missing items at the end of an item sequence are represented by `None` in the lists of
    the selection and preview items, as in `PalletizingEnvironment`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self) -> None:
        self._size = SIZE_EURO_PALLET
        """The palletizing target's size of the base area in x- and y-direction given in millimeters."""
        self._n_orientations = 2
        """The amount of different orientations that are allowed during palletization."""

        self.action_space = Dict(
            {
                "x": Discrete(self._size[0]),
                "y": Discrete(self._size[1]),
                "orientation": Discrete(self._n_orientations),
            }
        )
        """The action space consists of the simple action spaces for x-, y-coordinate and the orientation of the item."""

        self.observation_space = Box(
            low=0, high=MAXHEIGHT_OBSERVATION_SPACE, shape=(self._size[1], self._size[0]), dtype=int
        )
        """The observation space describes the heights in each coordinate on the palletizing target."""

        self._target_space = Space3D(self._size)
        """Represents the 3D space where the palletization takes place."""

        self._actions = []
        """A list that stores all actions done in the order of their palletizing point in time."""

        self._palletized_volume = 0.0
        """A float that stores the palletized volume of all items in cm^3."""

        self._orders: list[Order] = []
        """The orders that are given whenever the method `reset` is called with an `order_sequence`."""
        self._current_order: Optional[Order] = None
        """The order that is currently palletized."""
        self._order_sequence: list[str] = []
        """The ids of the given orders in the same order as they are given."""
        self._item_sequence_counter = None
        """This integer stores the position within the item sequence of the current order (starting at `0`)."""

        self._n_item_preview = int(conf.get("environment", "preview"))
        """The amount of preview items."""
        self._n_item_selection = int(conf.get("environment", "selection"))
        """The amount of items to select from, i.e., to select for the next step."""

        self._items_selection: list[Optional[Item]] = []
        """The items to select from for the next palletizing step."""
        self._items_preview: list[Optional[Item]] = []
        """The preview items."""

        self._kpis = KPIs()
        """Holds the values of the KPIs for each order."""

        self._mp_score_estimation = False
        """Indicates whether this environment is used for estimating scores. In this case, the selection and preview
        items are set with `setItems` and the preview is not refilled from the current order."""

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, dict]:
        """
        In the step function we have to palletize the given item at the given position. Translated to this implementation that means that we have to
        (a) calculate the z-coordinate of the placement for the given action (x-,y-coordinate and orientation of item),
        (b) append the item to the palletized items,
        (c) update the height map of the palletizing target,
        (d) obtain the allowed positions for the upcoming item,
        (e) obtain the reward of the step, and
        (f) obtain the additional information.

        Parameters.
        -----------
            action: dict
                The action contains the `"x"` and `"y"` coordinate of the placement, the `"item"` and its `"orientation"`.

        Returns.
        --------
            stepReturns: tuple
                The step method returns the new observation (`object`), the reward (`float`), whether the episode has ended (`bool`) and additional information (`dict`).

        Examples.
        ---------
        >>> action = {
                "x": 100, # int
                "y": 100, # int
                "orientation": 0, # int
                "item": Item(article='cake-00104295', id='c00104295', product_group='confectionery', length_mm=590, width_mm=200, height_mm=210, weight_kg=7.67, sequence=1)
            }
        """
        info = {}

        # get the variables that are needed here
        step_vars = self._get_step_variables(action)
        item_for_action: Item = action["item"]

        # define the item
        item = Cuboid(item_for_action)
        item.set_orientation(step_vars["orientation"])

        # create a np.ndarray that has the same shape as the target, its elements are 1 if the item is located in this region and 0 otherwise
        item_on_target = np.zeros((self._size[1], self._size[0]), dtype=int)
        item_delta_y, item_delta_x = item.array_representation.shape
        start_x, start_y = step_vars["xCoord"], step_vars["yCoord"]
        try:
            item_on_target[start_y : start_y + item_delta_y, start_x : start_x + item_delta_x] = np.ones(
                item.array_representation.shape, dtype=int
            )
        except Exception:
            # have to crop item like in `environment.Space3D.addItem`
            logger.warning("cropped item")
            cropped_shape = (
                min(item_on_target.shape[0], start_y + item_delta_y) - start_y,
                min(item_on_target.shape[1], start_x + item_delta_x) - start_x,
            )
            item_on_target[start_y : start_y + cropped_shape[0], start_x : start_x + cropped_shape[1]] = np.ones(
                cropped_shape, dtype=int
            )

        # obtain the FLB height for the item in the selected (x, y)-coordinate
        max_height_in_target_area = int(np.amax(np.multiply(self._target_space.getHeights(), item_on_target)))

        # define the action in the needed format
        action_extended = {
            "item": item_for_action,
            "flb_coordinates": [step_vars["xCoord"], step_vars["yCoord"], max_height_in_target_area],
            "orientation": step_vars["orientation"],
        }
        self._actions.append(action_extended)
        info.update({"action_for_vis": action_extended})
        logger.debug(f"step() -> extended action: {action_extended}")

        # add the item to the palletizing target
        self._target_space.addItem(item, action_extended["orientation"], action_extended["flb_coordinates"])
        info.update({"support_area/%": item.percentage_direct_support_surface})

        # prepare for next call of step
        additional_info = self.__prepareForNextStep(item_for_action)
        done = additional_info.pop("done")
        info.update(additional_info)

        # update the attributes
        self._palletized_volume += (step_vars["deltaX"] * step_vars["deltaY"] * step_vars["itemHeight"]) / 1000.0
        self._kpis.update()

        reward = self.__getReward(done)
        info = self.__getInfo("step", info, done)

        step_returns = self._target_space.getHeights(), reward, done, info
        return step_returns

    def reset(self, order_sequence: Optional[list[Order]] = None) -> tuple[np.ndarray, dict]:
        """
        This method is responsible for
        (a) the change of the orders, e.g., from "00100001" -> "00100002",
        (b) the reset of the stored attributes and heights of the virtual palletizing target,
        (c) obtaining the inital observation and information about the environment,
        (d) resetting the target's size of the base area, depending on the goal that is defined in the given data for the episodes, and
        (e) loading the order data that shoud be considered.

        Parameters.
        -----------
        order_sequence: Optional[list[Order]] (default = None)
            The orders that are palletized. If given, the palletization starts with the first order of this sequence.
            Otherwise, the next order of the previously given sequence is considered.

        Returns.
        --------
        observation: np.array
            The stored heights of the environment.
        info: dict
            A dictionary that contains additional information that might be useful for the machine learning agent.
        """
        # # # # # Change the Order that is considered # # # # #
        done = False
        # change the current order
        if self._current_order is None or order_sequence is not None:
            self._orders = order_sequence.copy()
            self._order_sequence = [order.id for order in self._orders]
            self._current_order = self._orders[0]

        elif self._current_order.id == self._order_sequence[-1]:
            # reached the last order in the data
            done = True

        else:
            next_order_index = self._order_sequence.index(self._current_order.id) + 1
            self._current_order = self._orders[next_order_index]

        if not done:
            logger.debug(f"current order: {self._current_order.id}")

        # # # # # Reset the Attributes # # # # #
        self._item_sequence_counter = 0
        # change the size related to the palletizing target and the action space
        palletizing_target = self._current_order.properties.target
        if palletizing_target == "rollcontainer":
            self._size = SIZE_ROLLCONTAINER
        elif palletizing_target == "euro-pallet":
            self._size = SIZE_EURO_PALLET
        else:
            # size is given as `"x,y,z"`
            sizes = palletizing_target.split(",")
            self._size = (int(sizes[0]), int(sizes[1]))

        self.action_space = Dict(
            {
                "x": Discrete(self._size[0]),
                "y": Discrete(self._size[1]),
                "orientation": Discrete(self._n_orientations),
            }
        )

        self._target_space.reset(self._size)
        self._actions = []
        self._palletized_volume = 0.0
        self._kpis.reset(self._target_space, self._current_order)

        self._items_selection = []
        self._items_preview = []
        if not done:
            # only update items if we do not
            self.__updateItemsSelection()
            self.__updateItemsPreview()

        # # # # # Obtain the Observation and Info # # # # #
        observation = self._target_space.getHeights()
        info = self.__getInfo("reset", done=done)

        return observation, info

    def render(self, mode="human") -> None:
        pass

    def close(self) -> None:
        pass

    def __getReward(self, done: bool) -> float:
        """
        You can define your reward here.
        """
        if done:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def __getInfo(self, calledby: str = "step", additionalinfo: Optional[dict] = None, done: bool = False) -> dict:
        """Get the info dictionary after reset or step."""
        if additionalinfo is None:
            info = {}
        else:
            info = additionalinfo

        if calledby == "step":
            info.update(
                {
                    "all_orders_considered": None,
                    "item_volume_on_target/cm^3": self._palletized_volume,
                    "n_items_in_order": len(self._current_order.item_sequence),
                }
            )

        elif calledby == "reset":
            # if done is True, then all orders are considered
            if not done:
                next_items = self.__obtainNextItems()
                corner_points = self.__determineCornerPoints(next_items["selection"])

                info.update(
                    {
                        "all_orders_considered": done,
                        "order_id": self._current_order.id,
                        "next_items_selection": next_items["selection"],
                        "next_items_preview": next_items["preview"],
                        "corner_points": corner_points,
                    }
                )

            else:
                # all orders considered
                info.update(
                    {
                        "all_orders_considered": done,
                        "order_id": None,
                        "next_items_selection": [],
                        "next_items_preview": [],
                        "corner_points": {},
                    }
                )

        return info

    def _get_step_variables(self, action: dict) -> dict:
        """This method creates the variables that are needed in the `step` method, depending on the given action."""
        orientation = action["orientation"]
        x_coord, y_coord = int(action["x"]), int(action["y"])

        item: Item = action["item"]
        item_length, item_width, item_height = int(item.length_mm), int(item.width_mm), int(item.height_mm)

        if orientation == 0:
            delta_x, delta_y = item_length, item_width
        elif orientation == 1:
            delta_x, delta_y = item_width, item_length

        step_var = {
            "xCoord": x_coord,
            "yCoord": y_coord,
            "deltaX": delta_x,
            "deltaY": delta_y,
            "itemHeight": item_height,
            "orientation": orientation,
        }

        return step_var

    def __prepareForNextStep(self, placeditem: Item) -> dict:
        """
        This method prepares the environment for the next call of the `step` method. Hence, the
        `_item_sequence_counter` is increased and the next palletizing items and their corner points are calculated,
        unless the current episode has finished (after the currently called `step`).

        Returns.
        --------
            info: dict
                Information that is returned by the `step` method.
        """
        info = {}

        self._item_sequence_counter += 1
        if self._item_sequence_counter >= len(self._current_order.item_sequence):
            done = True  # episode finished
            info.update({"next_items_selection": [], "next_items_preview": []})
        else:
            done = False
            self.__updateItemsSelection(placeditem)
            self.__updateItemsPreview()
            next_items = self.__obtainNextItems()

            # get the corner points for the items that can be selected
            corner_points = self.__determineCornerPoints(next_items["selection"])

            info.update(
                {
                    "next_items_selection": next_items["selection"],
                    "next_items_preview": next_items["preview"],
                    "corner_points": corner_points,
                }
            )

        info.update({"done": done})

        return info

    def __obtainNextItems(self) -> dict:
        """
        Returns a dictionary that contains the next items in the item sequence of order that is currently considered. Depending on the task, i.e., on the specified values for the preview `k` and the selection `s`, the lenght of the returned item list differs.

        As example, if the task `"O3DBP-k-s"` is loaded, then the length of the list with the selection items is `s`, and the list of the preview items is `k-s`. In general, the preview list is not empty, if and only if `k > s`.

        If less items are left in the item sequence as the values of `k` and `s` would request, `None` is appended instead of an item.

        Returns.
        --------
        next_items: dict
            Contains the next items in the item sequence, i.e., the keys `"selection"` and `"preview"` with lists of
            `Item` (or `None`) as values.
        """
        next_items = {"selection": self._items_selection, "preview": self._items_preview}
        return next_items

    def __determineCornerPoints(self, possibleitems: list[Optional[Item]]) -> dict:
        """
        Determines the corner points for all items that are given.

        Parameters.
        -----------
        possibleitems: list
            Contains the items that could be selected for palletization. `None` entries are ignored.

        Returns.
        --------
        corner_points: dict
            The corner points for each item article and orientation, e.g., `{article: {0: [...], 1: [...]}}`.
        """
        corner_points = {}
        for item in possibleitems:
            if item is not None:
                corner_points[item.article] = {}
                for orientation in range(self._n_orientations):
                    if orientation == 0:
                        length, width, height = item.length_mm, item.width_mm, item.height_mm
                    elif orientation == 1:
                        width, length, height = item.length_mm, item.width_mm, item.height_mm

                    corner_points[item.article][orientation] = self._target_space.getCornerPointsIn3D(
                        (length, width, height)
                    )

        return corner_points

    def getNPlacedItems(self) -> int:
        """Returns the amount of placed items in the environment."""
        return len(self._actions)

    def __updateItemsSelection(self, placeditem: Optional[Item] = None) -> None:
        """
        This method updates the items that can be selected.

        Parameters.
        -----------
        placeditem: Optional[Item]
            The item that was palletized in the last `env.step` call.

        Important.
        ----------
        Call this method before `self.__updateItemsPreview`.
        """
        if (self._item_sequence_counter == 0) or (len(self._items_preview) == 0):
            if self._item_sequence_counter != 0:
                # not first call -> remove placed item from selection
                self._items_selection.remove(placeditem)
            # first call in env.reset in an episode
            for s in range(self._n_item_selection):
                item_index = self._item_sequence_counter + s
                if item_index < len(self._current_order.item_sequence):
                    self._items_selection.append(self._current_order.item_sequence[item_index])
                else:
                    self._items_selection.append(None)
        else:
            # call in env.step
            self._items_selection.remove(placeditem)
            self._items_selection.append(self._items_preview.pop(0))

    def __updateItemsPreview(self) -> None:
        """
        This method updates the items that are known in advance, but cannot be selected.

        Different to `PalletizingEnvironment`: when estimating scores (see `setItems`), the preview is only padded with
        `None`, since the upcoming items are unknown to the heuristic.

        Important.
        ----------
        Call this method after `self.__updateItemsSelection`.
        """
        n_pure_preview_items = self._n_item_preview - self._n_item_selection

        if self._mp_score_estimation:
            for _ in range(self._n_item_selection, self._n_item_preview):
                if len(self._items_preview) < n_pure_preview_items:
                    self._items_preview.append(None)

        else:
            for k in range(self._n_item_selection, self._n_item_preview):
                item_index = self._item_sequence_counter + k
                if item_index < len(self._current_order.item_sequence):
                    preview_item = self._current_order.item_sequence[item_index]
                    if preview_item not in self._items_preview:
                        self._items_preview.append(preview_item)
                elif len(self._items_preview) < n_pure_preview_items:
                    self._items_preview.append(None)

    def setItems(self, preview: list[Optional[Item]], selection: list[Optional[Item]]) -> None:
        """Sets the preview and selection items and switches to the score estimation mode."""
        self._items_selection = copy.deepcopy(selection)
        self._items_preview = copy.deepcopy(preview)

        self._mp_score_estimation = True

    def remStoredOrder(self) -> None:
        """
        Deletes the stored orders, except the current order.

        => Reduce time copy.deepcopy needs
        """
        self._orders = []

    def copy_without_orders(self) -> "SimPalEnv":
        """
        Returns a deep copy of this environment without the stored orders (the current order is kept).

        This is equivalent to `copy.deepcopy(env)` followed by `remStoredOrder()`, but avoids copying all orders, which
        is expensive for large order sequences.
        """
        stored_orders = self._orders
        self._orders = []
        try:
            return copy.deepcopy(self)
        finally:
            self._orders = stored_orders
