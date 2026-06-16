#!/usr/bin/env python3
"""
agent_manager.py
Rule-based prompt-to-mission manager for CoppeliaSim P3DX.

Later, replace choose_mission_from_prompt() with an LLM call that returns the
same JSON mission format. The rest of the robot control code does not need to
change.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any, Dict

from p3dx_interface import P3DXInterface
from p3dx_missions import P3DXMissionController, result_to_json


def _number_after(pattern: str, text: str, default: float) -> float:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def choose_mission_from_prompt(prompt: str) -> Dict[str, Any]:
    """Very small deterministic 'agent manager'.

    Output schema:
      {"mission": "go_to_goal|mapping|circle|rotate|forward|stop", "params": {...}}
    """
    p = prompt.lower().strip()

    if any(k in p for k in [
        "go to goal", "go to the goal", "move to goal", "reach goal",
        "go to target", "go to the target", "move to target", "reach target",
    ]):
        return {"mission": "go_to_goal", "params": {"timeout": 60.0, "tolerance": 0.20}}

    if any(k in p for k in ["map", "mapping", "explore", "survey"]):
        duration = _number_after(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)", p, 30.0)
        return {"mission": "mapping", "params": {
            "duration": duration,
            "mode": "wall_follow",
            "max_walls": 3,
            "output": "map_points.json",
            "output_png": "map_graph.png",
        }}

    if "circle" in p:
        diameter = _number_after(r"(?:diameter|dia|d)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)", p, 1.0)
        # Also support: "1 diameter" / "1m diameter"
        diameter = _number_after(r"(\d+(?:\.\d+)?)\s*(?:m\s*)?diameter", p, diameter)
        clockwise = "counter" not in p and "anticlockwise" not in p
        return {"mission": "circle", "params": {"diameter": diameter, "clockwise": clockwise}}

    if any(k in p for k in ["turn around", "rotate", "spin"]):
        clockwise = "counter" not in p and "anticlockwise" not in p
        angle = -2.0 * math.pi if clockwise else 2.0 * math.pi
        # Support "rotate 90 degree"
        deg = _number_after(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)", p, 360.0)
        angle = math.radians(deg) * (-1.0 if clockwise else 1.0)
        return {"mission": "rotate", "params": {"angle_rad": angle}}

    if any(k in p for k in ["forward", "move forward"]):
        distance = _number_after(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)", p, 0.5)
        return {"mission": "forward", "params": {"distance": distance}}

    if "stop" in p:
        return {"mission": "stop", "params": {}}

    return {"mission": "unknown", "params": {}, "error": "I cannot map this prompt to a mission yet."}


def execute_mission(plan: Dict[str, Any], controller: P3DXMissionController):
    mission = plan.get("mission")
    params = plan.get("params", {})

    if mission == "go_to_goal":
        return controller.go_to_goal(**params)
    if mission == "mapping":
        return controller.mapping(**params)
    if mission == "circle":
        return controller.circle(**params)
    if mission == "rotate":
        return controller.rotate(**params)
    if mission == "forward":
        return controller.open_loop_forward(**params)
    if mission == "stop":
        controller.stop()
        state = controller.robot.get_state(compact=True)
        from p3dx_missions import MissionResult
        return MissionResult(True, "stop", "stopped", 0.0, state, {})

    raise ValueError(plan.get("error", f"Unknown mission: {mission}"))


def main():
    parser = argparse.ArgumentParser(description="Prompt-based P3DX mission manager, no LLM yet")
    parser.add_argument("prompt", help="Natural-language command, e.g. 'Go to the goal'")
    parser.add_argument("--dry-run", action="store_true", help="Only print the selected mission JSON")
    parser.add_argument("--dt", type=float, default=0.10, help="Closed-loop control timestep")
    args = parser.parse_args()

    plan = choose_mission_from_prompt(args.prompt)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    robot = P3DXInterface()
    controller = P3DXMissionController(robot=robot, dt=args.dt)
    result = execute_mission(plan, controller)
    print(result_to_json(result))


if __name__ == "__main__":
    main()
