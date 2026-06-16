#!/usr/bin/env python3
"""
p3dx_interface.py
==================

Single-file (state, action) interface between CoppeliaSim and an LLM /
agent manager, for a Pioneer P3DX mobile robot.

Two ways to use it
------------------
1. As a Python module (in-process closed loop):

    from p3dx_interface import P3DXInterface
    robot = P3DXInterface()
    state = robot.get_state()
    state = robot.step({"type": "body_twist", "linear": 0.2, "angular": 0.0})
    robot.reset()

2. As a CLI (for agent managers that mainly run shell commands):

    python p3dx_interface.py discover
    python p3dx_interface.py get-state
    python p3dx_interface.py get-state --full
    python p3dx_interface.py apply-action '{"type": "body_twist", "linear": 0.2, "angular": 0.0}'
    python p3dx_interface.py step '{"type": "preset", "name": "forward", "params": {"distance": 0.5}}'
    python p3dx_interface.py reset

Each CLI call opens a fresh ZMQ connection, does one thing, prints a
JSON result to stdout, and exits. CoppeliaSim itself holds all
persistent state, so the Python side stays stateless.

BEFORE THIS WORKS ON YOUR SCENE
--------------------------------
With the scene open, run:

    python p3dx_interface.py discover

This prints every object's path and type. Use it to fix the paths in
CONFIG below (robot base, wheel joints, ultrasonic sensors, vision
sensor, lidar, goal dummy) to match your actual scene.

NOTE ON API NAMES
------------------
A few sim.* function/constant names differ slightly between CoppeliaSim
versions (4.2 / 4.4 / 4.5 / 4.6+). If something throws
"AttributeError: ... has no attribute ...", that's almost always a
one-line fix to the function/constant name for your version - tell me
the error and I'll adjust it.
"""

import sys
import json
import math
import time
import argparse

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ---------------------------------------------------------------------------
# CONFIG - edit to match your scene (run `discover` to find real paths)
# ---------------------------------------------------------------------------
CONFIG = {
    "robot": "/PioneerP3DX",
    "left_motor": "/PioneerP3DX/leftMotor",
    "right_motor": "/PioneerP3DX/rightMotor",

    # Ultrasonic ring: tries "<prefix>[i]" for i in 0..n-1, then "<prefix>i"
    # (1-indexed) as a fallback, since model versions differ.
    "ultrasonic_prefix": "/PioneerP3DX/ultrasonicSensor",
    "n_ultrasonic": 16,

    # Set to None until you've added these in the scene
    "vision_sensor": None,   # e.g. "/PioneerP3DX/visionSensor"
    "lidar": None,           # e.g. "/PioneerP3DX/fastHokuyo"

    "goal": "/Goal",

    # Robot geometry - check these against your model. `discover` prints
    # wheel positions; wheel_separation = distance between wheel centers,
    # wheel_radius = radius of the wheel shape.
    "wheel_radius": 0.0975,      # m
    "wheel_separation": 0.331,   # m

    # Safety clamps
    "max_wheel_speed": 6.0,      # rad/s
    "max_linear_speed": 0.5,     # m/s, used by body_pose controller
    "max_angular_speed": 1.5,    # rad/s
}


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------
def twist_to_wheels(v, w, wheel_radius, wheel_separation):
    """Differential-drive: body twist (v, w) -> (left, right) wheel angular vel."""
    wl = (v - w * wheel_separation / 2.0) / wheel_radius
    wr = (v + w * wheel_separation / 2.0) / wheel_radius
    return wl, wr


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
class P3DXInterface:
    def __init__(self, config=None):
        self.cfg = {**CONFIG, **(config or {})}
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject("sim")
        self.h = self._resolve_handles()

    # -- setup ---------------------------------------------------------
    def _resolve_handles(self):
        sim = self.sim
        h = {}
        h["robot"] = sim.getObject(self.cfg["robot"])
        h["left_motor"] = sim.getObject(self.cfg["left_motor"])
        h["right_motor"] = sim.getObject(self.cfg["right_motor"])

        h["ultrasonic"] = []
        prefix = self.cfg["ultrasonic_prefix"]
        for i in range(self.cfg["n_ultrasonic"]):
            for path in (f"{prefix}[{i}]", f"{prefix}{i + 1}"):
                try:
                    h["ultrasonic"].append(sim.getObject(path))
                    break
                except Exception:
                    continue

        h["vision_sensor"] = None
        if self.cfg.get("vision_sensor"):
            try:
                h["vision_sensor"] = sim.getObject(self.cfg["vision_sensor"])
            except Exception:
                pass

        h["lidar"] = None
        if self.cfg.get("lidar"):
            try:
                h["lidar"] = sim.getObject(self.cfg["lidar"])
            except Exception:
                pass

        h["goal"] = None
        if self.cfg.get("goal"):
            try:
                h["goal"] = sim.getObject(self.cfg["goal"])
            except Exception:
                pass

        return h

    # -- state -----------------------------------------------------------
    def get_state(self, compact=True):
        sim = self.sim
        pos = sim.getObjectPosition(self.h["robot"], -1)
        ori = sim.getObjectOrientation(self.h["robot"], -1)
        lin_vel, ang_vel = sim.getObjectVelocity(self.h["robot"])

        state = {
            "time": sim.getSimulationTime(),
            "robot_pose": {"x": pos[0], "y": pos[1], "z": pos[2], "yaw": ori[2]},
            "velocity": {"linear": lin_vel, "angular": ang_vel},
            "sensors": {},
        }

        if self.h["goal"] is not None:
            gpos = sim.getObjectPosition(self.h["goal"], -1)
            state["goal_pose"] = {"x": gpos[0], "y": gpos[1], "z": gpos[2]}
        else:
            state["goal_pose"] = None

        ultrasonic = []
        for sh in self.h["ultrasonic"]:
            res = sim.readProximitySensor(sh)
            detected, dist = res[0], res[1]
            ultrasonic.append(round(dist, 3) if detected else None)
        state["sensors"]["ultrasonic"] = ultrasonic

        if self.h["vision_sensor"] is not None:
            state["sensors"]["camera"] = self._read_camera(compact)

        if self.h["lidar"] is not None:
            state["sensors"]["lidar"] = self._read_lidar(compact)

        return state

    def _read_camera(self, compact):
        sim = self.sim
        img, res = sim.getVisionSensorImg(self.h["vision_sensor"])
        try:
            depth, dres = sim.getVisionSensorDepth(self.h["vision_sensor"])
        except Exception:
            depth, dres = None, None

        import numpy as np

        if not compact:
            rgb = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
            rgb = np.flipud(rgb)
            out = {"resolution": res, "rgb": rgb}
            if depth is not None:
                d = np.frombuffer(depth, dtype=np.float32).reshape(dres[1], dres[0])
                out["depth"] = np.flipud(d)
            return out

        # compact: small base64 JPEG thumbnail, safe for JSON / LLM prompts
        import cv2
        import base64

        rgb = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        rgb = np.flipud(rgb)
        thumb = cv2.resize(rgb, (64, 64))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        out = {
            "resolution": res,
            "thumbnail_jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii") if ok else None,
        }
        if depth is not None:
            d = np.frombuffer(depth, dtype=np.float32).reshape(dres[1], dres[0])
            out["depth_mean"] = float(np.mean(d))
            out["depth_min"] = float(np.min(d))
        return out

    def _read_lidar(self, compact):
        # Placeholder - depends on the lidar model you add. Many CoppeliaSim
        # lidar models (e.g. fastHokuyo) publish their scan on a string
        # signal as a packed float table. Adjust the signal name, or replace
        # this with a sweep over proximity sensors for a custom scanner.
        sim = self.sim
        try:
            data = sim.getStringSignal("fastHokuyoData")
            ranges = sim.unpackFloatTable(data) if data is not None else None
        except Exception:
            ranges = None
        if not ranges:
            return None
        if compact and len(ranges) > 36:
            step = len(ranges) // 36
            ranges = ranges[::step]
        return {"ranges": ranges}

    # -- actions -----------------------------------------------------------
    def apply_action(self, action):
        a_type = action.get("type")

        if a_type == "joint_velocity":
            self._set_wheel_velocities(action["left"], action["right"])

        elif a_type == "joint_torque":
            self._set_wheel_torques(action["left"], action["right"])

        elif a_type == "body_twist":
            wl, wr = twist_to_wheels(
                action.get("linear", 0.0), action.get("angular", 0.0),
                self.cfg["wheel_radius"], self.cfg["wheel_separation"],
            )
            self._set_wheel_velocities(wl, wr)

        elif a_type == "body_pose":
            wl, wr = self._pose_controller_step(
                action["x"], action["y"], action.get("theta")
            )
            self._set_wheel_velocities(wl, wr)

        elif a_type == "preset":
            self._run_preset(action["name"], action.get("params", {}))

        elif a_type == "stop":
            self._set_wheel_velocities(0.0, 0.0)

        else:
            raise ValueError(f"Unknown action type: {a_type!r}")

    def step(self, action, settle_time=0.0):
        """Apply an action, optionally wait settle_time seconds, then
        return the new state. Preset actions handle their own timing,
        so settle_time is usually 0 for those."""
        self.apply_action(action)
        if settle_time > 0:
            time.sleep(settle_time)
        return self.get_state()

    def reset(self):
        sim = self.sim
        sim.stopSimulation()
        time.sleep(0.2)
        sim.startSimulation()
        time.sleep(0.2)
        return self.get_state()

    # -- low-level helpers -------------------------------------------------
    def _set_wheel_velocities(self, wl, wr):
        m = self.cfg["max_wheel_speed"]
        wl, wr = clamp(wl, -m, m), clamp(wr, -m, m)
        self.sim.setJointTargetVelocity(self.h["left_motor"], wl)
        self.sim.setJointTargetVelocity(self.h["right_motor"], wr)

    def _set_wheel_torques(self, tl, tr):
        # Torque/force control: set the target force/torque, and a small
        # target velocity in that direction so the motor actually turns
        # under the applied force.
        sim = self.sim
        for h, t in ((self.h["left_motor"], tl), (self.h["right_motor"], tr)):
            sim.setJointTargetForce(h, abs(t))
            sim.setJointTargetVelocity(
                h, math.copysign(self.cfg["max_wheel_speed"], t) if t != 0 else 0.0
            )

    def _pose_controller_step(self, gx, gy, gtheta=None,
                               k_rho=0.6, k_alpha=2.0, k_beta=-0.5):
        """One step of a go-to-pose controller. Call repeatedly via step()
        until rho is small - this is the closed-loop part of 'body_pose'."""
        pos = self.sim.getObjectPosition(self.h["robot"], -1)
        ori = self.sim.getObjectOrientation(self.h["robot"], -1)
        x, y, yaw = pos[0], pos[1], ori[2]

        dx, dy = gx - x, gy - y
        rho = math.hypot(dx, dy)
        alpha = normalize_angle(math.atan2(dy, dx) - yaw)

        v = k_rho * rho
        w = k_alpha * alpha
        if gtheta is not None and rho < 0.1:
            w += k_beta * normalize_angle(gtheta - yaw)

        v = clamp(v, -self.cfg["max_linear_speed"], self.cfg["max_linear_speed"])
        w = clamp(w, -self.cfg["max_angular_speed"], self.cfg["max_angular_speed"])
        return twist_to_wheels(v, w, self.cfg["wheel_radius"], self.cfg["wheel_separation"])

    def _run_preset(self, name, params):
        if name == "forward":
            distance = params.get("distance", 0.5)
            speed = params.get("speed", 0.2)
            sign = 1 if distance >= 0 else -1
            wheel_speed = sign * speed / self.cfg["wheel_radius"]
            duration = abs(distance) / max(speed, 1e-6)
            self._set_wheel_velocities(wheel_speed, wheel_speed)
            time.sleep(duration)
            self._set_wheel_velocities(0.0, 0.0)

        elif name == "rotate":
            angle = params.get("angle", math.pi / 2)  # radians
            speed = params.get("speed", 0.5)           # rad/s
            sign = 1 if angle >= 0 else -1
            w = sign * speed
            wl, wr = twist_to_wheels(0.0, w, self.cfg["wheel_radius"], self.cfg["wheel_separation"])
            duration = abs(angle) / max(speed, 1e-6)
            self._set_wheel_velocities(wl, wr)
            time.sleep(duration)
            self._set_wheel_velocities(0.0, 0.0)

        elif name == "stop":
            self._set_wheel_velocities(0.0, 0.0)

        else:
            raise ValueError(f"Unknown preset: {name!r}")


# ---------------------------------------------------------------------------
# Scene discovery - run this first to fill in CONFIG
# ---------------------------------------------------------------------------
def discover():
    client = RemoteAPIClient()
    sim = client.getObject("sim")
    handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)

    type_names = {
        sim.object_shape_type: "shape",
        sim.object_joint_type: "joint",
        sim.object_dummy_type: "dummy",
        sim.object_proximitysensor_type: "proximity sensor",
        sim.object_visionsensor_type: "vision sensor",
    }

    print(f"{'PATH':45s} TYPE")
    print("-" * 60)
    for h in handles:
        path = sim.getObjectAlias(h, 1)
        otype = sim.getObjectType(h)
        print(f"{path:45s} {type_names.get(otype, otype)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="P3DX <-> CoppeliaSim interface")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("discover")

    p_state = sub.add_parser("get-state")
    p_state.add_argument("--full", dest="compact", action="store_false", default=True)

    p_action = sub.add_parser("apply-action")
    p_action.add_argument("action_json")

    p_step = sub.add_parser("step")
    p_step.add_argument("action_json")
    p_step.add_argument("--settle", type=float, default=0.0)

    sub.add_parser("reset")

    args = parser.parse_args()

    if args.command == "discover":
        discover()
        return

    robot = P3DXInterface()

    if args.command == "get-state":
        print(json.dumps(robot.get_state(compact=args.compact), indent=2, default=str))

    elif args.command == "apply-action":
        action = json.loads(args.action_json)
        robot.apply_action(action)
        print(json.dumps({"ok": True}))

    elif args.command == "step":
        action = json.loads(args.action_json)
        state = robot.step(action, settle_time=args.settle)
        print(json.dumps(state, indent=2, default=str))

    elif args.command == "reset":
        state = robot.reset()
        print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    main()
