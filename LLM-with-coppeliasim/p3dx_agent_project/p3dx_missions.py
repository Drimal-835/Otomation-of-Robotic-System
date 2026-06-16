#!/usr/bin/env python3
"""
p3dx_missions.py
Closed-loop mission controllers for Pioneer P3DX in CoppeliaSim.

Version: wall-follow mapping + improved go-to-goal obstacle avoidance.

Key assumptions from the user scene:
- Ultrasonic sensors are named /PioneerP3DX/ultrasonicSensor[i], i=0..15.
- Sensor order:
    0  = left front side
    1  = 45 deg from left to front
    2,3,4,5 = front arc
    6  = 45 deg from right to front
    7  = right front side
    8  = right rear side
    9  = 45 deg from right rear to back
    10,11,12,13 = rear arc
    14 = 45 deg from left rear to back
    15 = left rear side

Angle convention used below:
- 0 rad = robot front
- + angle = left
- - angle = right
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p3dx_interface import P3DXInterface, normalize_angle, clamp


def _finite_ranges(us: List[Optional[float]], default: float = 3.0) -> List[float]:
    """Convert ultrasonic readings where None means no detection."""
    return [default if v is None else float(v) for v in us]


# Your described P3DX ultrasonic order.
# These angles are approximate but much better than treating the ring as generic.
P3DX_US_ANGLES_RAD = [
    math.radians(90),    # 0 left
    math.radians(45),    # 1 left-front diagonal
    math.radians(30),    # 2 front-left
    math.radians(10),    # 3 front-left-center
    math.radians(-10),   # 4 front-right-center
    math.radians(-30),   # 5 front-right
    math.radians(-45),   # 6 right-front diagonal
    math.radians(-90),   # 7 right
    math.radians(-90),   # 8 right rear side
    math.radians(-135),  # 9 right-back diagonal
    math.radians(-160),  # 10 back-right
    math.radians(-175),  # 11 back center-right
    math.radians(175),   # 12 back center-left
    math.radians(160),   # 13 back-left
    math.radians(135),   # 14 left-back diagonal
    math.radians(90),    # 15 left rear side
]


@dataclass
class MissionResult:
    ok: bool
    mission: str
    reason: str
    elapsed: float
    final_state: Dict[str, Any]
    extra: Dict[str, Any]


class P3DXMissionController:
    """Reusable closed-loop mission library.

    All motion commands go through P3DXInterface.apply_action().
    This is the layer your agent manager should call after it decides the mission.
    """

    def __init__(self, robot: Optional[P3DXInterface] = None, dt: float = 0.10):
        self.robot = robot or P3DXInterface()
        self.dt = dt

    # ------------------------------------------------------------------
    # Basic robot control helpers
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self.robot.apply_action({"type": "stop"})

    def _ultrasonic_summary(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Summarize ultrasonic ring using the user's exact sensor order."""
        us = state.get("sensors", {}).get("ultrasonic", []) or []
        r = _finite_ranges(us, default=3.0)
        n = len(r)
        if n == 0:
            return {
                "front": 3.0, "front_left": 3.0, "front_right": 3.0,
                "left": 3.0, "right": 3.0, "rear": 3.0,
                "rear_left": 3.0, "rear_right": 3.0, "all_min": 3.0,
            }

        def mn(ids: List[int], fallback: float = 3.0) -> float:
            vals = [r[i] for i in ids if 0 <= i < n]
            return min(vals) if vals else fallback

        if n >= 16:
            # IMPORTANT: keep pure side sensors separate from diagonal sensors.
            # For wall following, using min([side, front-diagonal, rear-side]) makes
            # the robot overreact at corners and produces the wavy path you saw.
            return {
                "front": mn([2, 3, 4, 5]),
                "front_left": mn([1, 2, 3]),
                "front_right": mn([4, 5, 6]),
                "left": mn([0]),
                "right": mn([7]),
                "left_front_diag": mn([1]),
                "right_front_diag": mn([6]),
                "left_rear": mn([15]),
                "right_rear": mn([8]),
                "rear": mn([10, 11, 12, 13]),
                "rear_left": mn([13, 14, 15]),
                "rear_right": mn([8, 9, 10]),
                "all_min": min(r),
            }

        # Fallback if a custom model has fewer sensors.
        return {
            "front": mn(list(range(max(0, n // 2 - 2), min(n, n // 2 + 3)))),
            "front_left": mn(list(range(0, max(1, n // 2)))),
            "front_right": mn(list(range(max(0, n // 2), n))),
            "left": mn([0]),
            "right": mn([n - 1]),
            "rear": 3.0,
            "rear_left": 3.0,
            "rear_right": 3.0,
            "all_min": min(r),
        }

    def _choose_free_turn_direction(self, clear: Dict[str, float]) -> float:
        """Return +1 to turn left, -1 to turn right."""
        # If left is more open than right, turn left. Otherwise turn right.
        left_clearance = min(clear["front_left"], clear["left"])
        right_clearance = min(clear["front_right"], clear["right"])
        return 1.0 if left_clearance >= right_clearance else -1.0

    def _wall_follow_twist(
        self,
        clear: Dict[str, float],
        side: str = "right",
        desired_wall_distance: float = 0.38,
        base_speed: float = 0.09,
        max_w: float = 0.65,
    ) -> Tuple[float, float]:
        """Smooth wall follower using pure side sensors.

        The previous version used grouped side values like right=min([6,7,8])
        and left=min([0,1,15]). That is bad for wall following because diagonal
        and rear sensors make the robot think the side wall suddenly becomes
        too close/too far, causing large S-shaped waves.

        This version uses:
        - pure side sensor 0 or 7 to maintain wall distance
        - front diagonal 1 or 6 to predict corners
        - front arc 2..5 only for collision/corner handling
        """
        front = clear["front"]
        kp_side = 0.75
        kp_corner = 1.20

        if side == "right":
            side_dist = clear.get("right", 3.0)              # sensor 7 only
            front_diag = clear.get("right_front_diag", clear["front_right"])  # sensor 6

            # If the side wall disappears, gently turn right to reacquire it.
            if side_dist > 2.5:
                v = base_speed * 0.80
                w = -0.30
            else:
                # right wall too close -> turn left (+w)
                # right wall too far   -> turn right (-w)
                w = kp_side * (desired_wall_distance - side_dist)
                v = base_speed

            # Corner prediction: front-right sees wall before side sensor does.
            if front_diag < desired_wall_distance * 1.15:
                w += kp_corner * (desired_wall_distance * 1.15 - front_diag)

            # Front wall/corner: turn left, not backward unless very close.
            if front < 0.22:
                v = 0.01
                w = 0.65
            elif front < 0.40:
                v = base_speed * 0.45
                w += 0.35
            elif front < 0.60:
                v = min(v, base_speed * 0.70)

        else:
            side_dist = clear.get("left", 3.0)               # sensor 0 only
            front_diag = clear.get("left_front_diag", clear["front_left"])    # sensor 1

            # If the side wall disappears, gently turn left to reacquire it.
            if side_dist > 2.5:
                v = base_speed * 0.80
                w = 0.30
            else:
                # left wall too close -> turn right (-w)
                # left wall too far   -> turn left (+w)
                w = -kp_side * (desired_wall_distance - side_dist)
                v = base_speed

            # Corner prediction: front-left sees wall before side sensor does.
            if front_diag < desired_wall_distance * 1.15:
                w -= kp_corner * (desired_wall_distance * 1.15 - front_diag)

            # Front wall/corner: turn right, not backward unless very close.
            if front < 0.22:
                v = 0.01
                w = -0.65
            elif front < 0.40:
                v = base_speed * 0.45
                w -= 0.35
            elif front < 0.60:
                v = min(v, base_speed * 0.70)

        return clamp(v, 0.0, 0.12), clamp(w, -max_w, max_w)
    
    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _sensor_points_from_state(
        self,
        state: Dict[str, Any],
        max_range: float = 2.5,
        wall_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Convert ultrasonic detections into world-coordinate obstacle points."""
        pose = state["robot_pose"]
        us = state.get("sensors", {}).get("ultrasonic", []) or []
        obstacle_points: List[Dict[str, float]] = []

        for i, dist in enumerate(us):
            if dist is None:
                continue
            dist = float(dist)
            if dist <= 0.02 or dist > max_range:
                continue

            if i < len(P3DX_US_ANGLES_RAD):
                local_angle = P3DX_US_ANGLES_RAD[i]
            else:
                local_angle = -math.pi + (2.0 * math.pi * i / max(1, len(us)))

            world_angle = pose["yaw"] + local_angle
            obstacle_points.append({
                "x": pose["x"] + dist * math.cos(world_angle),
                "y": pose["y"] + dist * math.sin(world_angle),
                "source": f"ultrasonic[{i}]",
                "distance": dist,
                "t": state.get("time", 0.0),
                "wall_id": wall_id,
            })

        robot_path_point = [{
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
            "t": state.get("time", 0.0),
            "wall_id": wall_id,
        }]
        return obstacle_points, robot_path_point

    def _save_map_plot(
        self,
        obstacle_points: List[Dict[str, float]],
        path_points: List[Dict[str, float]],
        output_png: str,
        wall_loops: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save a simple map graph as PNG."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))

        if obstacle_points:
            # Plot points with wall_id=None first, then numeric wall IDs.
            # Do NOT directly sort mixed values like [None, 0, 1], because
            # Python cannot compare NoneType with int.
            wall_ids_raw = set(p.get("wall_id") for p in obstacle_points)
            wall_ids = [None] if None in wall_ids_raw else []
            wall_ids += sorted(wid for wid in wall_ids_raw if wid is not None)

            for wid in wall_ids:
                pts = [p for p in obstacle_points if p.get("wall_id") == wid]
                if not pts:
                    continue
                ox = [p["x"] for p in pts]
                oy = [p["y"] for p in pts]
                label = "Search obstacle points" if wid is None else f"Wall {wid} points"
                ax.scatter(ox, oy, s=8, alpha=0.55, label=label)

        if path_points:
            px = [p["x"] for p in path_points]
            py = [p["y"] for p in path_points]
            ax.plot(px, py, linewidth=1.5, label="Robot path")
            ax.scatter([px[0]], [py[0]], marker="o", s=60, label="Start")
            ax.scatter([px[-1]], [py[-1]], marker="x", s=80, label="End")

        if wall_loops:
            for loop in wall_loops:
                sx = loop.get("start_x")
                sy = loop.get("start_y")
                if sx is not None and sy is not None:
                    ax.scatter([sx], [sy], marker="s", s=60, label=f"Wall {loop.get('wall_id')} follow start")

        ax.set_title("P3DX ultrasonic wall-follow exploration map")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_png, dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Missions
    # ------------------------------------------------------------------
    def go_to_goal(
        self,
        timeout: float = 60.0,
        tolerance: float = 0.20,
        safe_distance: float = 0.58,
        emergency_distance: float = 0.24,
        wall_follow_distance: float = 0.38,
    ) -> MissionResult:
        """Go to /Goal while avoiding obstacles without spinning 180 degrees.

        This is a simple Bug-style controller:
        - NORMAL: face and move toward goal.
        - AVOID: if blocked, follow the nearer wall.
        - Leave AVOID only when the front is clear and the goal direction is visible again.
        """
        start = time.time()
        reason = "timeout"
        last_state: Dict[str, Any] = {}
        avoid_count = 0

        mode = "NORMAL"
        follow_side = "right"
        hit_distance_to_goal = float("inf")
        avoid_enter_time = 0.0
        last_mode_change = time.time()

        while time.time() - start < timeout:
            state = self.robot.get_state(compact=True)
            last_state = state
            goal = state.get("goal_pose")
            if not goal:
                reason = "No goal dummy found. Create /Goal or edit CONFIG['goal']."
                break

            pose = state["robot_pose"]
            dx = goal["x"] - pose["x"]
            dy = goal["y"] - pose["y"]
            distance = math.hypot(dx, dy)
            if distance <= tolerance:
                reason = "goal_reached"
                break

            clear = self._ultrasonic_summary(state)
            target_heading = math.atan2(dy, dx)
            heading_error = normalize_angle(target_heading - pose["yaw"])

            front_blocked = clear["front"] < safe_distance or clear["front_left"] < 0.42 or clear["front_right"] < 0.42
            front_clear = clear["front"] > safe_distance * 1.35 and clear["front_left"] > safe_distance and clear["front_right"] > safe_distance
            goal_mostly_ahead = abs(heading_error) < math.radians(45)

            if mode == "NORMAL" and front_blocked:
                mode = "AVOID"
                avoid_count += 1
                avoid_enter_time = time.time()
                last_mode_change = time.time()
                hit_distance_to_goal = distance
                # Follow the side where the wall/obstacle is nearer.
                # If obstacle is more on left, keep it on left; if more on right, keep it on right.
                follow_side = "left" if clear["front_left"] < clear["front_right"] else "right"

            elif mode == "AVOID":
                # Prevent rapid mode switching. Leave wall-follow only when path is clear
                # and we are closer to the goal than when we hit the obstacle.
                enough_time_following = (time.time() - avoid_enter_time) > 1.8
                closer_than_hit = distance < hit_distance_to_goal - 0.05
                if enough_time_following and front_clear and goal_mostly_ahead and closer_than_hit:
                    mode = "NORMAL"
                    last_mode_change = time.time()

            if mode == "NORMAL":
                # Do not drive forward if the target is behind the robot.
                # Rotate toward the goal with a small forward component only when aligned.
                heading_factor = max(0.0, math.cos(heading_error))
                desired_v = clamp(0.45 * distance, 0.00, 0.24) * heading_factor
                if abs(heading_error) > math.radians(70):
                    desired_v = 0.0
                desired_w = clamp(1.25 * heading_error, -0.85, 0.85)
                v, w = desired_v, desired_w
            else:
                v, w = self._wall_follow_twist(
                    clear,
                    side=follow_side,
                    desired_wall_distance=wall_follow_distance,
                    base_speed=0.10,
                    max_w=0.75,
                )
                # Gentle bias toward the goal, but do not let it override wall safety.
                w = clamp(w + 0.20 * clamp(heading_error, -1.0, 1.0), -0.95, 0.95)

            # Emergency close-range behavior: short reverse + turn toward more open side.
            if clear["front"] < emergency_distance:
                turn = self._choose_free_turn_direction(clear)
                v = -0.04
                w = 0.75 * turn

            self.robot.apply_action({"type": "body_twist", "linear": v, "angular": w})
            time.sleep(self.dt)

        self.stop()
        elapsed = time.time() - start
        return MissionResult(
            reason == "goal_reached",
            "go_to_goal",
            reason,
            elapsed,
            self.robot.get_state(compact=True),
            {
                "avoidance_events": avoid_count,
                "last_state": last_state,
                "final_mode": mode,
                "follow_side": follow_side,
            },
        )

    def mapping(
        self,
        duration: float = 45.0,
        safe_distance: float = 0.60,
        emergency_distance: float = 0.20,
        output: str = "map_points.json",
        output_png: str = "map_graph.png",
        max_range: float = 2.5,
        mode: str = "wall_follow",
        max_walls: int = 2,
        loop_close_distance: float = 0.40,
        min_follow_time: float = 10.0,
        disconnected_min_distance: float = 0.70,
    ) -> MissionResult:
        """Explore and map using a wall-follow strategy.

        Algorithm:
        1. SEARCH_WALL: wander until a wall/obstacle is detected.
        2. FOLLOW_WALL: follow that wall boundary.
        3. When robot returns near the first follow point, close that wall loop.
        4. SEARCH_WALL again and ignore walls too close to previous wall start points.

        This is not full SLAM, but it matches the requested behavior better than random wandering.
        """
        if mode not in {"wall_follow", "wander"}:
            mode = "wall_follow"

        start = time.time()
        obstacle_points: List[Dict[str, float]] = []
        path_points: List[Dict[str, float]] = []
        wall_loops: List[Dict[str, Any]] = []
        last_state: Dict[str, Any] = {}

        state_machine = "SEARCH_WALL"
        current_wall_id: Optional[int] = None
        follow_side = "right"
        follow_start_pose: Optional[Dict[str, float]] = None
        follow_start_time = 0.0
        last_turn_switch = time.time()
        wander_bias = 0.25
        completed_wall_starts: List[Tuple[float, float]] = []
        completed_wall_points: List[Tuple[float, float]] = []
        current_wall_points: List[Tuple[float, float]] = []

        def far_from_completed_walls(x: float, y: float) -> bool:
            if not completed_wall_starts:
                return True
            return all(math.hypot(x - sx, y - sy) >= disconnected_min_distance for sx, sy in completed_wall_starts)

        def candidate_wall_is_new(points: List[Dict[str, float]]) -> bool:
            """Return True if the currently detected obstacle is not part of a completed wall.

            Comparing only robot position to the first wall-start point makes the robot
            rediscover the same outer wall again. Compare the detected obstacle points
            against all completed wall points instead.
            """
            if not completed_wall_points or not points:
                return True
            for p in points:
                px, py = p["x"], p["y"]
                if all(math.hypot(px - qx, py - qy) > disconnected_min_distance for qx, qy in completed_wall_points):
                    return True
            return False

        while time.time() - start < duration:
            state = self.robot.get_state(compact=True)
            last_state = state
            pose = state["robot_pose"]
            clear = self._ultrasonic_summary(state)

            new_obstacles, new_path = self._sensor_points_from_state(state, max_range=max_range, wall_id=current_wall_id)
            obstacle_points.extend(new_obstacles)
            path_points.extend(new_path)

            wall_detected = min(clear["front"], clear["front_left"], clear["front_right"], clear["left"], clear["right"]) < safe_distance

            if mode == "wander":
                # Old behavior kept as fallback.
                if time.time() - last_turn_switch > 5.0:
                    wander_bias = random.choice([-0.30, -0.20, 0.20, 0.30])
                    last_turn_switch = time.time()
                desired_v = 0.18
                desired_w = wander_bias
                if clear["front"] < safe_distance:
                    desired_v = 0.04
                    desired_w = 0.75 * self._choose_free_turn_direction(clear)
                v, w = desired_v, desired_w

            elif state_machine == "SEARCH_WALL":
                current_wall_id = None

                if wall_detected and far_from_completed_walls(pose["x"], pose["y"]) and candidate_wall_is_new(new_obstacles):
                    current_wall_id = len(wall_loops) + 1
                    state_machine = "FOLLOW_WALL"
                    follow_start_pose = {"x": pose["x"], "y": pose["y"], "yaw": pose["yaw"]}
                    follow_start_time = time.time()
                    current_wall_points = [(p["x"], p["y"]) for p in new_obstacles]
                    # Follow side closest to wall using pure side sensors 0 and 7.
                    # If both side sensors are far, use the front-left/front-right arc.
                    if min(clear["left"], clear["front_left"]) < min(clear["right"], clear["front_right"]):
                        follow_side = "left"
                    else:
                        follow_side = "right"
                    v, w = self._wall_follow_twist(
                        clear,
                        side=follow_side,
                        desired_wall_distance=0.38,
                        base_speed=0.09,
                        max_w=0.65,
                    )
                else:
                    # Search pattern: drive forward with changing bias. If blocked by a completed wall,
                    # turn away and move to another area.
                    if time.time() - last_turn_switch > 4.0:
                        wander_bias = random.choice([-0.35, -0.20, 0.20, 0.35])
                        last_turn_switch = time.time()

                    v = 0.18
                    w = wander_bias
                    if clear["front"] < safe_distance:
                        v = 0.03
                        w = 0.85 * self._choose_free_turn_direction(clear)
                    if wall_detected and not candidate_wall_is_new(new_obstacles):
                        # This is probably the already-followed outer wall. Move away from it
                        # so the robot can search inward and find the center wall.
                        v = 0.12
                        if min(clear["left"], clear["front_left"]) < min(clear["right"], clear["front_right"]):
                            w = -0.55  # wall on left -> turn right, away from it
                        else:
                            w = 0.55   # wall on right -> turn left, away from it

            else:  # FOLLOW_WALL
                assert follow_start_pose is not None
                v, w = self._wall_follow_twist(
                    clear,
                    side=follow_side,
                    desired_wall_distance=0.38,
                    base_speed=0.09,
                    max_w=0.65,
                )
                current_wall_points.extend((p["x"], p["y"]) for p in new_obstacles)
                follow_elapsed = time.time() - follow_start_time
                dist_to_start = math.hypot(pose["x"] - follow_start_pose["x"], pose["y"] - follow_start_pose["y"])

                # Close loop when the robot comes back near the first follow point.
                if follow_elapsed > min_follow_time and dist_to_start < loop_close_distance:
                    completed_wall_starts.append((follow_start_pose["x"], follow_start_pose["y"]))
                    completed_wall_points.extend(current_wall_points)
                    current_wall_points = []
                    wall_loops.append({
                        "wall_id": current_wall_id,
                        "start_x": follow_start_pose["x"],
                        "start_y": follow_start_pose["y"],
                        "follow_time": follow_elapsed,
                        "closed_by_distance": dist_to_start,
                    })

                    # Leave the wall and search for another disconnected wall.
                    state_machine = "SEARCH_WALL"
                    current_wall_id = None
                    follow_start_pose = None
                    follow_start_time = 0.0

                    # If requested number of walls is complete, end early.
                    if len(wall_loops) >= max_walls:
                        break

                    # Move away from the completed wall before searching again.
                    # If the wall is on the right, turn left; if wall is on the left, turn right.
                    turn = 1.0 if follow_side == "right" else -1.0
                    self.robot.apply_action({"type": "body_twist", "linear": 0.12, "angular": 0.55 * turn})
                    time.sleep(2.0)
                    continue

            if clear["front"] < emergency_distance:
                turn = self._choose_free_turn_direction(clear)
                v = 0.00
                w = 0.65 * turn

            self.robot.apply_action({"type": "body_twist", "linear": v, "angular": w})
            time.sleep(self.dt)

        self.stop()

        out_json = Path(output)
        out_png = Path(output_png)
        map_data = {
            "meta": {
                "type": "ultrasonic_wall_follow_map",
                "note": "Wall-follow mapping demo. This is not full SLAM/localization correction.",
                "sensor_model": "/PioneerP3DX/ultrasonicSensor[i], i=0..15, user-provided order",
                "duration": duration,
                "mode": mode,
                "max_walls": max_walls,
            },
            "wall_loops": wall_loops,
            "path": path_points,
            "obstacle_points": obstacle_points,
        }
        out_json.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
        self._save_map_plot(obstacle_points, path_points, str(out_png), wall_loops=wall_loops)

        elapsed = time.time() - start
        return MissionResult(
            True,
            "mapping",
            f"saved {len(obstacle_points)} obstacle points, {len(wall_loops)} wall loop(s), graph to {out_png}",
            elapsed,
            self.robot.get_state(compact=True),
            {
                "map_file": str(out_json),
                "map_graph": str(out_png),
                "wall_loops": wall_loops,
                "obstacle_points": len(obstacle_points),
                "path_points": len(path_points),
                "last_state": last_state,
            },
        )

    def circle(self, diameter: float = 1.0, clockwise: bool = True,
               speed: float = 0.15) -> MissionResult:
        """Drive a circle using body-space twist converted to wheel space."""
        radius = max(0.05, diameter / 2.0)
        angular = speed / radius
        if clockwise:
            angular = -angular
        duration = (2.0 * math.pi * radius) / max(speed, 1e-6)

        start = time.time()
        while time.time() - start < duration:
            self.robot.apply_action({"type": "body_twist", "linear": speed, "angular": angular})
            time.sleep(self.dt)
        self.stop()
        elapsed = time.time() - start
        return MissionResult(True, "circle", "completed", elapsed,
                             self.robot.get_state(compact=True), {"diameter": diameter, "clockwise": clockwise})

    def rotate(self, angle_rad: float = -2.0 * math.pi, angular_speed: float = 0.6) -> MissionResult:
        """Rotate in place. Negative angle = clockwise."""
        duration = abs(angle_rad) / max(abs(angular_speed), 1e-6)
        w = math.copysign(abs(angular_speed), angle_rad)
        start = time.time()
        while time.time() - start < duration:
            self.robot.apply_action({"type": "body_twist", "linear": 0.0, "angular": w})
            time.sleep(self.dt)
        self.stop()
        elapsed = time.time() - start
        return MissionResult(True, "rotate", "completed", elapsed,
                             self.robot.get_state(compact=True), {"angle_rad": angle_rad})

    def open_loop_forward(self, distance: float = 0.5, speed: float = 0.2) -> MissionResult:
        """Use the interface preset. This is intentionally open-loop."""
        start = time.time()
        self.robot.apply_action({"type": "preset", "name": "forward", "params": {"distance": distance, "speed": speed}})
        elapsed = time.time() - start
        return MissionResult(True, "open_loop_forward", "completed", elapsed,
                             self.robot.get_state(compact=True), {"distance": distance, "speed": speed})


def result_to_json(result: MissionResult) -> str:
    return json.dumps(asdict(result), indent=2, default=str)
