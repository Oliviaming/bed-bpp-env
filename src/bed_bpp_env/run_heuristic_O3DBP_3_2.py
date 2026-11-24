"""
This script runs the heuristic solver `heuristics.O3DBP_3_2`.
"""

import logging

import bed_bpp_env.utils as utils

# configure the parser of the given arguments
parser = utils.arguments_parser.addGroupToParser(
    "DemoGymPalEnv", "the arguments of the gym PalletizingEnvironment demo"
)
parser.add_argument(
    "-d", "--debug", action="store_true", default=False, help="Indicates whether the simulation is debugged."
)
parser.add_argument(
    "-v", "--visualize", action="store_true", default=False, help="Indicates whether the simulation is visualized."
)
parser.add_argument(
    "--vis_debug", action="store_true", default=False, help="Indicates whether all visualizations should be displayed."
)
parser.add_argument("--task", type=str, default="O3DBP-3-2", help="Defines the palletizing task.")
parser.add_argument(
    "--data",
    type=str,
    default=utils.getPathToExampleData().joinpath("5_bed-bpp.json"),
    help="Defines which data is used.",
)
parser.add_argument(
    "--no_move_outwards",
    action="store_true",
    default=False,
    help="Do not shift corner points towards the outside of the target (the shift causes unsupported placements).",
)
parser.add_argument(
    "--no_origin_fallback",
    action="store_true",
    default=False,
    help="Do not place items at the origin when the selected corner point is higher than the origin.",
)
parser.add_argument(
    "--min_support", type=float, default=0.0, help="Minimum estimated support ratio of a placement, in [0, 1]."
)
parser.add_argument(
    "--max_base_height",
    type=int,
    default=None,
    help="Maximum height (mm) of the bottom of a placed item; the order is stopped if no placement is feasible.",
)
utils.arguments_parser.parse()


logger = logging.getLogger(__name__)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    import subprocess
    from pathlib import Path

    from bed_bpp_env.environment.palletizing_environment import PalletizingEnvironment
    from bed_bpp_env.environment.sim_pal_env import SimPalEnv
    from bed_bpp_env.heuristics.o3dbp_3_2 import O3DBP_3_2
    from bed_bpp_env.io_utils import load_order_sequence

    logger.info(f"given arguments: {utils.PARSEDARGUMENTS}")
    logger.info(f"the results are stored in {utils.OUTPUTDIRECTORY}")

    # init heuristic
    _, nPreview, nSelection = utils.PARSEDARGUMENTS.get("task").split("-")
    heuristic = O3DBP_3_2(
        preview=int(nPreview),
        selection=int(nSelection),
        move_corner_points_outwards=not utils.PARSEDARGUMENTS["no_move_outwards"],
        origin_fallback=not utils.PARSEDARGUMENTS["no_origin_fallback"],
        min_support=utils.PARSEDARGUMENTS["min_support"],
        max_base_height_mm=utils.PARSEDARGUMENTS["max_base_height"],
    )

    order_sequence = load_order_sequence(utils.PARSEDARGUMENTS["data"])

    # setup a simulation palletizing environment
    env = PalletizingEnvironment()
    observation, info = env.reset(order_sequence=order_sequence)
    simenv = SimPalEnv()
    _, _ = simenv.reset(order_sequence=order_sequence)
    heuristic.setSimEnv(simenv)

    # start simulation
    simDone = False
    while not (simDone):
        env.render()
        action, successful = heuristic.getAction(observation, info)  # User-defined policy function
        if successful:
            observation, reward, episodeDone, info = env.step(action)
        else:
            episodeDone = True

        if episodeDone:
            env.render()
            observation, info = env.reset()
            _, _ = simenv.reset()
            heuristic.setSimEnv(simenv)
            if info["all_orders_considered"]:
                simDone = True

    env.close()

    # run evaluation
    EVALUATION_SCRIPT_PATH = Path(__file__).parent / "script_evaluate_packing_plan.py"
    cmd = [
        "python3",
        EVALUATION_SCRIPT_PATH.as_posix(),
        "--data",
        utils.PARSEDARGUMENTS["data"],
        "--packing_plan",
        utils.OUTPUTDIRECTORY.joinpath("packing_plans.json"),
    ]
    subprocess.run(cmd, shell=False)
