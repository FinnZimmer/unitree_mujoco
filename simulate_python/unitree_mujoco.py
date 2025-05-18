import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np
from test.kinematics import generate_target_pos, get_foot_pos, get_relative_hip_end_pos, get_relative_knee_pos
from test.inv_kinematics import get_angles_for_target
from test.transform import rotate_by_quaternion
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread(unitree):
    global mj_data, mj_model

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    # Create visualization markers
    marker_size = 0.02
    marker_colors = {
        "FR": (1, 0, 0, 1),  # Red
        "FL": (0, 1, 0, 1),  # Green
        "RR": (0, 0, 1, 1),  # Blue
        "RL": (1, 1, 0, 1)   # Yellow
    }

    step_count = 0
    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        mujoco.mj_step(mj_model, mj_data)
        step_count += 1

        # Update foot positions first
        # Get robot's base position and rotation
        base_pos = mj_data.qpos[:3]  # x, y, z position
        base_quat = mj_data.qpos[3:7]  # quaternion rotation

        #print(mj_data.qfrc_passive)

        # Use robot's low_state
        state = unitree.low_state
        if state is not None:
            # FR leg
            fr_hip = state.motor_state[0].q
            fr_thigh = state.motor_state[1].q
            fr_calf = state.motor_state[2].q
                
            # FL leg
            fl_hip = state.motor_state[3].q
            fl_thigh = state.motor_state[4].q
            fl_calf = state.motor_state[5].q
                
            # RR leg
            rr_hip = state.motor_state[6].q
            rr_thigh = state.motor_state[7].q
            rr_calf = state.motor_state[8].q
                
            # RL leg
            rl_hip = state.motor_state[9].q
            rl_thigh = state.motor_state[10].q
            rl_calf = state.motor_state[11].q

        # FR foot
        fr_pos = get_foot_pos(base_quat, fr_hip, fr_thigh, fr_calf, "FR")
        viewer.user_scn.geoms[0].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[0].size = [marker_size] * 3
        viewer.user_scn.geoms[0].pos = base_pos + fr_pos
        viewer.user_scn.geoms[0].rgba = marker_colors["FR"]
        viewer.user_scn.geoms[0].mat = np.eye(3)
        
        # FL foot
        fl_pos = get_foot_pos(base_quat, fl_hip, fl_thigh, fl_calf, "FL")
        viewer.user_scn.geoms[1].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[1].size = [marker_size] * 3
        viewer.user_scn.geoms[1].pos = base_pos + fl_pos
        viewer.user_scn.geoms[1].rgba = marker_colors["FL"]
        viewer.user_scn.geoms[1].mat = np.eye(3)
        
        # RR foot
        rr_pos = get_foot_pos(base_quat, rr_hip, rr_thigh, rr_calf, "RR")
        viewer.user_scn.geoms[2].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[2].size = [marker_size] * 3
        viewer.user_scn.geoms[2].pos = base_pos + rr_pos
        viewer.user_scn.geoms[2].rgba = marker_colors["RR"]
        viewer.user_scn.geoms[2].mat = np.eye(3)
        
        # RL foot
        rl_pos = get_foot_pos(base_quat, rl_hip, rl_thigh, rl_calf, "RL")
        viewer.user_scn.geoms[3].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[3].size = [marker_size] * 3
        viewer.user_scn.geoms[3].pos = base_pos + rl_pos
        viewer.user_scn.geoms[3].rgba = marker_colors["RL"]
        viewer.user_scn.geoms[3].mat = np.eye(3)

        # Target foot position
        target_pos = generate_target_pos("RL") # relative to the body frame
        target_hip, target_thigh, target_calf = get_angles_for_target(target_pos, "RL")
        calculated_target_pos = get_foot_pos(base_quat, target_hip, target_thigh, target_calf, "RL")

        # Print debug information every 100 steps
        if step_count % 100 == 0:
            pass

        # Visualize the target leg configuration
        # Hip position
        hip_pos = get_relative_hip_end_pos(target_hip, "RL")
        hip_pos = rotate_by_quaternion(hip_pos, base_quat)
        viewer.user_scn.geoms[4].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[4].size = [marker_size] * 3
        #viewer.user_scn.geoms[4].pos = base_pos + hip_pos
        viewer.user_scn.geoms[4].rgba = (1, 0, 0, 1)  # Red
        viewer.user_scn.geoms[4].mat = np.eye(3)

        # Knee position
        knee_pos = get_relative_knee_pos(target_hip, target_thigh, "RL")
        knee_pos = rotate_by_quaternion(knee_pos, base_quat)
        viewer.user_scn.geoms[5].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[5].size = [marker_size] * 3
        #viewer.user_scn.geoms[5].pos = base_pos + knee_pos
        viewer.user_scn.geoms[5].rgba = (0, 1, 0, 1)  # Green
        viewer.user_scn.geoms[5].mat = np.eye(3)

        # Foot position
        viewer.user_scn.geoms[6].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[6].size = [marker_size] * 3
        #viewer.user_scn.geoms[6].pos = base_pos + calculated_target_pos
        viewer.user_scn.geoms[6].rgba = (0.5, 0, 0.5, 0.5)
        viewer.user_scn.geoms[6].mat = np.eye(3)

        # True target position
        viewer.user_scn.geoms[7].type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.geoms[7].size = [marker_size] * 3
        #viewer.user_scn.geoms[7].pos = base_pos + rotate_by_quaternion(target_pos, base_quat)
        viewer.user_scn.geoms[7].rgba = (0, 0, 1, 1)
        viewer.user_scn.geoms[7].mat = np.eye(3)

        viewer.user_scn.ngeom = 8


        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )


        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


def LowStatePublisherThread(unitree):
    while viewer.is_running():
        locker.acquire()
        if unitree:
            unitree.PublishLowState()
        locker.release()
        time.sleep(config.SIMULATE_DT)


if __name__ == "__main__":
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread, args=(unitree,))
    lowstate_thread = Thread(target=LowStatePublisherThread, args=(unitree,))

    viewer_thread.start()
    sim_thread.start()
    lowstate_thread.start()
