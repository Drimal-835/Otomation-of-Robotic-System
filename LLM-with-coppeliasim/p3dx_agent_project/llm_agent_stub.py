#!/usr/bin/env python3
"""
llm_agent_stub.py
LLM integration placeholder.

Do not connect a real LLM yet. This file defines the exact JSON contract that
future LLM output must follow, then falls back to the rule-based parser.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from agent_manager import choose_mission_from_prompt

SYSTEM_CONTRACT = """
You are a robot mission planner for a Pioneer P3DX in CoppeliaSim.
Return ONLY valid JSON with this schema:
{
  "mission": "go_to_goal" | "mapping" | "circle" | "rotate" | "forward" | "stop",
  "params": { }
}
Allowed params:
- go_to_goal: timeout, tolerance
- mapping: duration, output
- circle: diameter, clockwise
- rotate: angle_rad, angular_speed
- forward: distance, speed
Never output raw motor commands unless explicitly requested.
""".strip()


def plan_with_llm_later(prompt: str, state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Future replacement point.

    Later you can call OpenAI/local Ollama/etc here. For now it returns the same
    plan as the deterministic parser, so the rest of the stack already works.
    """
    return choose_mission_from_prompt(prompt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    args = parser.parse_args()
    print(json.dumps(plan_with_llm_later(args.prompt), indent=2))
