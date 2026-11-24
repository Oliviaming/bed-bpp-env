"""
Runs one algorithm on a set of orders, evaluates the packing plans with the official Blender stability check and KPIs,
and writes a summary to `experiments/results/<dataset>/<algorithm>.json`.

The algorithms are driven with `SimPalEnv` as environment, which handles items exactly like `PalletizingEnvironment`
(see tests/test_o3dbp_3_2_port.py) but without its VTK rendering. The scores are identical to the ones of the run
scripts (e.g. the original O3DBP-3-2 scores 0.115 on example_data/5_bed-bpp.json in both).

usage (from the repository root, conda env `bed-bpp-env` active):
    PYTHONPATH=src python experiments/run_experiment.py --algorithm stable_heightmap_tuned \
        --data example_data/5_bed-bpp.json --dataset sample5

The summary contains no item data, only per-order counts and metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ALGORITHMS = {
    "o3dbp_original": {
        "class": "o3dbp",
        "label": "O3DBP-3-2 (original)",
        "params": {},
    },
    "o3dbp_constrained": {
        "class": "o3dbp",
        "label": "O3DBP-3-2 + support filter + 2 m cap",
        "params": {
            "move_corner_points_outwards": False,
            "origin_fallback": False,
            "min_support": 0.75,
            "max_base_height_mm": 2000,
        },
    },
    "stable_heightmap_basic": {
        "class": "stable",
        "label": "StableHeightmap (support >= 0.6, lookahead)",
        "params": {
            "min_support": 0.6,
            "lookahead_top_n": 8,
            "w_area": 1.0,
            "load_check": False,
            "w_rise": 1.0,
            "w_contact": 0.2,
            "w_flush": 0.2,
        },
    },
    "stable_heightmap_load_check": {
        "class": "stable",
        "label": "StableHeightmap + load check",
        "params": {
            "min_support": 0.3,
            "lookahead_top_n": 8,
            "w_area": 1.0,
            "load_check": True,
            "stability_margin_mm": 20,
            "stability_margin_per_m": 0,
            "load_check_max_candidates": 64,
            "w_rise": 1.0,
            "w_contact": 0.2,
            "w_flush": 0.2,
        },
    },
    "stable_heightmap_tuned": {
        "class": "stable",
        "label": "StableHeightmap tuned",
        "params": {},  # the defaults of StableHeightmapHeuristic are the tuned values
    },
}

BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RESULTS_DIR = Path(__file__).parent / "results"
TARGET_BASE_AREA_MM2 = {"euro-pallet": 1200 * 800, "rollcontainer": 800 * 700}
N_COLORS = 20


def make_env(preview: int, selection: int):
    from bed_bpp_env.environment.sim_pal_env import SimPalEnv

    env = SimPalEnv()
    env._n_item_preview, env._n_item_selection = preview, selection
    return env


def make_heuristic(algorithm: dict, sim_env, preview: int, selection: int):
    if algorithm["class"] == "o3dbp":
        from bed_bpp_env.heuristics.o3dbp_3_2 import O3DBP_3_2

        heuristic = O3DBP_3_2(preview=preview, selection=selection, **algorithm["params"])
        heuristic.setSimEnv(sim_env)
        return heuristic

    from bed_bpp_env.heuristics.stable_heightmap import StableHeightmapHeuristic

    return StableHeightmapHeuristic(**algorithm["params"])


def solve(algorithm: dict, orders: list, preview: int = 3, selection: int = 2) -> tuple[dict, dict]:
    """Returns the packing plans (serialized) and the solve time per order."""
    plans, seconds = {}, {}
    for order in orders:
        env, sim_env = make_env(preview, selection), make_env(preview, selection)
        observation, info = env.reset(order_sequence=[order])
        sim_env.reset(order_sequence=[order])
        heuristic = make_heuristic(algorithm, sim_env, preview, selection)
        start = time.perf_counter()
        done = False
        while not done:
            action, successful = heuristic.getAction(observation, info)
            if not successful:
                break
            observation, _, done, info = env.step(action)
        seconds[order.id] = time.perf_counter() - start
        plans[order.id] = [
            {"item": a["item"].to_dict(), "flb_coordinates": a["flb_coordinates"], "orientation": a["orientation"]}
            for a in env._actions
        ]
    return plans, seconds


def evaluate(orders: list, plans: dict) -> dict:
    """Official evaluation (Blender stability check + KPIs). Returns the KPIs per order id."""
    from bed_bpp_env.data_model.packing_plan import PackingPlan
    from bed_bpp_env.evaluation import EVALOUTPUTDIR
    from bed_bpp_env.evaluation.blender.stability_check import run_blender_stability_check_in_subprocess
    from bed_bpp_env.evaluation.packing_plan_evaluator import PackingPlanEvaluator

    evaluator = PackingPlanEvaluator()
    kpis = {}
    for order in orders:
        plan = PackingPlan.from_dict({"id": order.id, "actions": plans[order.id]})
        if not plan.actions:
            kpis[order.id] = {"kpi_6": 1, "eval_score_absolute_n_stable_pal_items": 0}
            continue
        colors = {}
        for item in order.item_sequence:
            colors.setdefault(item.article, f"own_{(len(colors) % N_COLORS) + 1:02d}")
        run_blender_stability_check_in_subprocess(BLENDER, order, plan, EVALOUTPUTDIR, colors, True, False)
        kpis[order.id] = evaluator.evaluate(plan, order)
    return kpis


def plan_metrics(order, actions: list) -> dict:
    """Metrics that follow from the packing plan alone: pile height, volume utilization, mean support."""
    from bed_bpp_env.data_model.action import Action
    from bed_bpp_env.environment.cuboid import Cuboid
    from bed_bpp_env.environment.space_3d import Space3D

    if not actions:
        return {"max_top_mm": 0, "volume_utilization": None, "mean_support": None}

    target = order.properties.target
    space = Space3D((1200, 800) if target == "euro-pallet" else (800, 700))
    supports, volume, max_top = [], 0.0, 0
    for serialized in actions:
        action = Action.from_dict(serialized)
        cuboid = Cuboid(action.item)
        cuboid.set_orientation(action.orientation)
        space.addItem(cuboid, action.orientation, list(action.flb_coordinates.xyz))
        supports.append(cuboid.percentage_direct_support_surface)
        item = action.item
        volume += item.length_mm * item.width_mm * item.height_mm
        max_top = max(max_top, action.flb_coordinates.z + item.height_mm)

    return {
        "max_top_mm": int(max_top),
        "volume_utilization": round(volume / (TARGET_BASE_AREA_MM2[target] * max_top), 4),
        "mean_support": round(sum(supports) / len(supports), 4),
    }


def summarize(name: str, algorithm: dict, dataset: str, orders: list, plans: dict, kpis: dict, seconds: dict) -> dict:
    per_order = []
    for order in orders:
        k = kpis[order.id]
        per_order.append(
            {
                "order": order.id,
                "target": order.properties.target,
                "items": len(order.item_sequence),
                "placed": len(plans[order.id]),
                "stable": int(k["kpi_6"]),
                "counted": int(k["eval_score_absolute_n_stable_pal_items"]),
                "solve_s": round(seconds.get(order.id, float("nan")), 2),
                **plan_metrics(order, plans[order.id]),
            }
        )

    total_items = sum(o["items"] for o in per_order)
    return {
        "algorithm": name,
        "label": algorithm["label"],
        "params": algorithm["params"],
        "dataset": dataset,
        "score": round(sum(o["counted"] for o in per_order) / total_items, 4),
        "stable_orders": sum(o["stable"] for o in per_order),
        "orders": len(per_order),
        "placed_ratio": round(sum(o["placed"] for o in per_order) / total_items, 4),
        "per_order": per_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    parser.add_argument("--data", required=True, help="orders in the format of the benchmark data")
    parser.add_argument("--dataset", required=True, help="name of the result folder, e.g. sample5")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # bed_bpp_env parses sys.argv in some places

    from bed_bpp_env.io_utils import load_order_sequence

    algorithm = ALGORITHMS[args.algorithm]
    orders = load_order_sequence(Path(args.data))
    plans, seconds = solve(algorithm, orders)
    kpis = evaluate(orders, plans)
    summary = summarize(args.algorithm, algorithm, args.dataset, orders, plans, kpis, seconds)

    out = RESULTS_DIR / args.dataset / f"{args.algorithm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{args.algorithm} on {args.dataset}: score {summary['score']}, "
          f"stable {summary['stable_orders']}/{summary['orders']}, placed {summary['placed_ratio']} -> {out}")


if __name__ == "__main__":
    main()
