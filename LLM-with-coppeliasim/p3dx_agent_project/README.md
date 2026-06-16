# P3DX Agent Project for CoppeliaSim

This project uses a deterministic Python controller first. The LLM can be connected later as a planner that selects a safe mission.

## Files

- `p3dx_interface.py` — CoppeliaSim ZeroMQ interface, state/action bridge.
- `p3dx_missions.py` — closed-loop missions and controllers.
- `agent_manager.py` — rule-based prompt-to-mission parser, no LLM yet.
- `llm_agent_stub.py` — future LLM integration placeholder.
- `run_tests_without_coppelia.py` — import/parser tests.

## Sensor order used in this version

The mission controller assumes this P3DX ultrasonic order:

```text
0  = left sensor
1  = 45 degree from left to front
2,3,4,5 = front sensors
6  = 45 degree from right to front
7  = right sensor
8  = right sensor but on the back
9  = 45 degree from right-back to back
10,11,12,13 = back sensors
14 = 45 degree from left-back to back
15 = left sensor but on the back
```

## Install

```bash
pip install -r requirements.txt
```

## First test

Open CoppeliaSim, load a scene with Pioneer P3DX and a `/Goal` dummy, then start simulation.

```bash
python p3dx_interface.py discover
python p3dx_interface.py get-state
```

Confirm that `sensors.ultrasonic` contains 16 values.

## Go to goal

```bash
python agent_manager.py "Go to the goal"
```

The go-to-goal controller now uses a Bug-style mode switch:

1. Normal mode: move toward the goal.
2. Avoid mode: if the front is blocked, follow the detected obstacle/wall.
3. It leaves avoid mode only when the front is clear, the goal is mostly ahead, and the robot is closer to the goal than when it first hit the obstacle.

This prevents the old problem where the robot turned 180 degrees and got stuck.

## Mapping

```bash
python agent_manager.py "Mapping the area for 60 seconds"
```

The mapping mission now uses wall-follow exploration:

1. Search for a wall.
2. Follow the wall.
3. When the robot returns near the first wall-follow point, close that wall loop.
4. Search for another wall that is far enough from the previous wall start point.
5. Save a map graph.

Outputs:

- `map_points.json`
- `map_graph.png`

This is not full SLAM. It is a practical ultrasonic wall-follow point map.

## Useful parameters

You can edit these in `agent_manager.py` or call `P3DXMissionController.mapping()` directly:

- `duration`
- `max_walls`
- `loop_close_distance`
- `disconnected_min_distance`
- `safe_distance`
- `emergency_distance`

For go-to-goal:

- `safe_distance`
- `emergency_distance`
- `wall_follow_distance`
- `tolerance`
- `timeout`
