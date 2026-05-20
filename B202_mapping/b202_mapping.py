# %%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.startSimulation()
print("Simulation Started")

def transformMat(alpha, beta, gamma, tx, ty, tz):
    rotx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha),  math.cos(alpha)]
        ])
    roty = np.array([
        [ math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
        ])
    rotz = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma),  math.cos(gamma), 0],
        [0,0,1]
        ])
    rot_total = np.matmul(rotx, roty)
    rot_total = np.matmul(rot_total, rotz)
    trans_vector = np.array([[tx],[ty],[tz]])
    R_t_3x4  = np.hstack((rot_total, trans_vector))
    homogeneous_row = np.array([[0, 0, 0, 1]])
    transform_matrix_4x4 = np.vstack((R_t_3x4, homogeneous_row))
    return transform_matrix_4x4

def transformMatMap(alpha, beta, gamma, tx, ty, tz):
    rotx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha),  math.cos(alpha)]
    ])
    roty = np.array([
        [ math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])
    rotz = np.array([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma),  math.cos(gamma), 0],
        [0, 0, 1]
    ])
    rot_total = np.matmul(np.matmul(rotx, roty), rotz)
    trans_vector = np.array([[tx], [ty], [tz]])
    R_t_3x4 = np.hstack((rot_total, trans_vector))
    homogeneous_row = np.array([[0, 0, 0, 1]])
    return np.vstack((R_t_3x4, homogeneous_row))

sim.addLog(1, "Hello from Python!")
p3dx     = sim.getObject("/PioneerP3DX")
p3dx_rw  = sim.getObject("/PioneerP3DX/rightMotor")
p3dx_lw  = sim.getObject("/PioneerP3DX/leftMotor")
LH_Handle   = sim.getObject("/LH")
perp_Handle = sim.getObject("/Perp")
path_Handle = []
pointNum = 48
for i in range(0, pointNum):
    path_Handle.append(sim.getObject(f"/p[{i}]"))

SENSOR_INDICES = [0, 3, 4, 7]
sensor_handles = []
for idx in SENSOR_INDICES:
    h = sim.getObject("/ultrasonicSensor", {"index": idx})
    sensor_handles.append(h)
    print(f"  Sensor index {idx} handle: {h}")

hits_x = {idx: [] for idx in SENSOR_INDICES}
hits_y = {idx: [] for idx in SENSOR_INDICES}

path_x, path_y = [], []

rw = 0.195/2
rb = 0.318/2
d  = 0.05
LH_distance = 1

x_odom = []
y_odom = []

# %%
try:
    start_time   = time.time()
    elapsed_prev = 0.0
    step = 0
    while (time.time() - start_time) < 90:

        elapsed      = time.time() - start_time
        dt           = elapsed - elapsed_prev
        elapsed_prev = elapsed

        # --- Robot pose ---
        p3dx_position    = sim.getObjectPosition(p3dx, sim.handle_world)
        p3dx_orientation = sim.getObjectOrientation(p3dx, sim.handle_world)
        alpha, beta, gamma = p3dx_orientation
        tx, ty, tz         = p3dx_position
        p3dx_pos = sim.getObjectPosition(p3dx, sim.handle_world)
        p3dx_ori = sim.getObjectOrientation(p3dx, sim.handle_world)
        path_x.append(p3dx_pos[0])
        path_y.append(p3dx_pos[1])

        T_body_world = transformMatMap(
            p3dx_ori[0], p3dx_ori[1], p3dx_ori[2],
            p3dx_pos[0], p3dx_pos[1], p3dx_pos[2]
        )

        # --- LH in world frame ---
        T = transformMat(alpha, beta, gamma, tx, ty, tz)
        LH_local = np.array([[LH_distance],[0],[0],[1]])
        LH_position_to_world = (T @ LH_local)[:3, :]   # (3,1)

        # --- Get all path points ---
        path_points = []
        for i in range(0, pointNum):
            pos = sim.getObjectPosition(path_Handle[i], sim.handle_world)
            path_points.append(np.array(pos).reshape((3,1)))

        # --- Scalar projection onto each segment, pick closest ---
        best_perp_dist = float('inf')
        best_proj_point = path_points[0]

        segments = [(path_points[i], path_points[i+1]) for i in range(len(path_points) - 1)]
        segments.append((path_points[-1], path_points[0]))

        for i in range(len(path_points) - 1):
            A = path_points[i]      # (3,1)
            B = path_points[i+1]    # (3,1)

            vec_AB  = B - A
            vec_ALH = LH_position_to_world - A

            AB_len = np.linalg.norm(vec_AB)
            if AB_len < 1e-6:   # skip degenerate segments
                continue

            # Scalar projection (clamped so point stays within segment)
            t = float(np.dot(vec_ALH.T, vec_AB) / (AB_len ** 2))
            t = max(0.0, min(1.0, t))   # clamp to [0, 1]

            # Projection point on segment
            proj_point = A + t * vec_AB  # (3,1)

            # Distance from LH to this projection point
            perp_dist = np.linalg.norm(LH_position_to_world - proj_point)

            if perp_dist < best_perp_dist:
                best_perp_dist  = perp_dist
                best_proj_point = proj_point

        # best_proj_point is the desired position (on the path, closest to LH)
        desired_position = best_proj_point  # (3,1)

        # --- Transform desired position to robot frame ---
        T_world_robot = transformMat(0, 0, gamma, tx, ty, tz)
        desired_position_wrt_robot = np.linalg.inv(T_world_robot) @ np.array([
            [desired_position[0][0]],
            [desired_position[1][0]],
            [desired_position[2][0]],
            [1]
        ])

        for i, (idx, h) in enumerate(zip(SENSOR_INDICES, sensor_handles)):
            res, dist, prox_point, prox_obj, prox_n = sim.readProximitySensor(h)

            if res and dist > 0:
                # ── FIX 1: use prox_point (actual hit XYZ in sensor local frame)
                # prox_point = [x, y, z] of the detected point in sensor frame
                sensor_hit = np.array([
                    [prox_point[0]],
                    [prox_point[1]],
                    [prox_point[2]],
                    [1]
                ])

                # Sensor pose relative to robot body
                s_pos = sim.getObjectPosition(h, p3dx)
                s_ori = sim.getObjectOrientation(h, p3dx)
                T_sensor_body = transformMatMap(
                    s_ori[0], s_ori[1], s_ori[2],
                    s_pos[0], s_pos[1], s_pos[2]
                )

                # sensor frame → body frame → world frame
                hit_body  = np.matmul(T_sensor_body, sensor_hit)
                hit_world = np.matmul(T_body_world,  hit_body)

                hits_x[idx].append(hit_world[0][0])
                hits_y[idx].append(hit_world[1][0])

        step += 1

        # --- Errors ---
        ed = math.sqrt(desired_position_wrt_robot[0]**2 + desired_position_wrt_robot[1]**2)
        eh = math.atan2(desired_position_wrt_robot[1], desired_position_wrt_robot[0])

        # --- Control ---
        vx = 0.3 * ed
        wx = 0.9 * eh

        wr_vel = (vx + (rb * wx) / 2) / rw
        wl_vel = (vx - (rb * wx) / 2) / rw

        sim.setJointTargetVelocity(p3dx_rw, wr_vel)
        sim.setJointTargetVelocity(p3dx_lw, wl_vel)

        # --- Update markers ---
        sim.setObjectPosition(LH_Handle,   sim.handle_world, LH_position_to_world.flatten().tolist())
        sim.setObjectPosition(perp_Handle, sim.handle_world, best_proj_point.flatten().tolist())

finally:
    sim.stopSimulation()
    print("\nSimulation Stopped")

all_x = []
all_y = []
for idx in SENSOR_INDICES:
    all_x.extend(hits_x[idx])
    all_y.extend(hits_y[idx])

plt.figure()
plt.plot(all_x, all_y, '.')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Sensor Readings in World Frame')
plt.axis('equal')
plt.show()
