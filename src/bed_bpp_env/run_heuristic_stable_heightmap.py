"""
This script runs the heuristic solver `heuristics.stable_heightmap.StableHeightmapHeuristic` and evaluates the
resulting packing plans (Blender stability check and KPIs).

Example (from the repository root, with `PYTHONPATH=src`):

    python src/bed_bpp_env/run_heuristic_stable_heightmap.py            # task O3DBP-3-2 on the 5-order sample
    python src/bed_bpp_env/run_heuristic_stable_heightmap.py -v         # with live visualization
"""

import logging

import bed_bpp_env.utils as utils

# configure the parser of the given arguments
parser = utils.arguments_parser.addGroupToParser(
    "StableHeightmap", "the arguments of the stable height map heuristic"
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
    "--params",
    type=str,
    default="{}",
    help='Override hyperparameters of StableHeightmapHeuristic as JSON, e.g. \'{"min_support": 0.4}\'. '
    "The defaults of the class are the tuned values.",
)
utils.arguments_parser.parse()

logger = logging.getLogger(__name__)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    import json
    import subprocess
    from pathlib import Path

    from bed_bpp_env.environment.palletizing_environment import PalletizingEnvironment
    from bed_bpp_env.heuristics.stable_heightmap import StableHeightmapHeuristic
    from bed_bpp_env.io_utils import load_order_sequence

    args = utils.PARSEDARGUMENTS
    logger.info(f"given arguments: {args}")
    logger.info(f"the results are stored in {utils.OUTPUTDIRECTORY}")

    heuristic = StableHeightmapHeuristic(**json.loads(args["params"]))

    order_sequence = load_order_sequence(args["data"])

    env = PalletizingEnvironment()
    observation, info = env.reset(order_sequence=order_sequence)

    # start simulation
    simDone = False
    while not simDone:
        env.render()
        action, successful = heuristic.getAction(observation, info)
        if successful:
            observation, reward, episodeDone, info = env.step(action)
        else:
            # no safe placement: stop this order
            episodeDone = True

        if episodeDone:
            env.render()
            observation, info = env.reset()
            if info["all_orders_considered"]:
                simDone = True

    env.close()

    # run evaluation
    EVALUATION_SCRIPT_PATH = Path(__file__).parent / "script_evaluate_packing_plan.py"
    cmd = [
        "python3",
        EVALUATION_SCRIPT_PATH.as_posix(),
        "--data",
        args["data"],
        "--packing_plan",
        utils.OUTPUTDIRECTORY.joinpath("packing_plans.json"),
    ]
    subprocess.run(cmd, shell=False)
