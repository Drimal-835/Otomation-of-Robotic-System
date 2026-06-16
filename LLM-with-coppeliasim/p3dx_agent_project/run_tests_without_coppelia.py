#!/usr/bin/env python3
"""Small offline tests for parser/kinematics. Does not connect to CoppeliaSim."""

import math
from agent_manager import choose_mission_from_prompt
from p3dx_interface import twist_to_wheels


def main():
    tests = [
        "Go to the goal",
        "Mapping the area for 10 seconds",
        "Turn around clockwise and make a circle with 1 diameter",
        "rotate 90 degree counterclockwise",
        "move forward 2 meters",
    ]
    for t in tests:
        print(t, "->", choose_mission_from_prompt(t))

    wl, wr = twist_to_wheels(0.2, 0.0, 0.0975, 0.331)
    assert abs(wl - wr) < 1e-9
    wl, wr = twist_to_wheels(0.0, math.pi / 4, 0.0975, 0.331)
    assert wl < 0 and wr > 0
    print("offline checks OK")


if __name__ == "__main__":
    main()
