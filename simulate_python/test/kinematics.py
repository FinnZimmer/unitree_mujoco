import sys
import os
from time import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transform import rotate_by_quaternion, quaternion_from_euler_rad, quat_to_euler
import numpy as np


HIP_OFFSETS = {
    "FR": [0.1934, -0.0465, 0],
    "FL": [0.1934, 0.0465, 0],
    "RR": [-0.1934, -0.0465, 0],
    "RL": [-0.1934, 0.0465, 0],
}

HIP_VEC = {
    "FR": [0, 1, 0],
    "FL": [0, -1, 0],
    "RR": [0, 1, 0],
    "RL": [0, -1, 0],
}

HIP_LENGTH = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213


def get_foot_pos(body_rotation, hip_angle, thigh_angle, calf_angle, joint_name):
    relative_foot_pos = get_relative_foot_pos(hip_angle, thigh_angle, calf_angle, joint_name)
    foot_pos = rotate_by_quaternion(relative_foot_pos, body_rotation)
    return foot_pos


def get_relative_foot_pos(hip_angle, thigh_angle, calf_angle, joint_name):
    hip_vec = np.multiply(HIP_VEC[joint_name], [HIP_LENGTH])

    thigh_vec = np.multiply([0, 0, -1], [THIGH_LENGTH])
    thigh_vec = rotate_by_quaternion(thigh_vec, quaternion_from_euler_rad(0, -thigh_angle, 0))

    calf_vec = np.multiply([0, 0, -1], [CALF_LENGTH])
    calf_vec = rotate_by_quaternion(calf_vec, quaternion_from_euler_rad(0, -thigh_angle-calf_angle, 0))

    leg_vec = np.add(thigh_vec, calf_vec)
    leg_vec = np.add(hip_vec, leg_vec)
    leg_vec = rotate_by_quaternion(leg_vec, quaternion_from_euler_rad(0, 0, -hip_angle))

    hip_base_pos = HIP_OFFSETS[joint_name]
    relative_foot_pos = np.add(hip_base_pos, leg_vec)

    return relative_foot_pos


def get_relative_hip_end_pos(hip_angle, joint_name):
    hip_base_pos = HIP_OFFSETS[joint_name]

    hip_vec = np.multiply(HIP_VEC[joint_name], [HIP_LENGTH])
    hip_vec = rotate_by_quaternion(hip_vec, quaternion_from_euler_rad(0, 0, -hip_angle))

    hip_end_pos = np.add(hip_base_pos, hip_vec)

    return hip_end_pos


def get_relative_knee_pos(hip_angle, thigh_angle, joint_name):
    thigh_vec = np.multiply([0, 0, -1], [THIGH_LENGTH])
    thigh_vec = rotate_by_quaternion(thigh_vec, quaternion_from_euler_rad(0, -thigh_angle, 0))
    thigh_vec = rotate_by_quaternion(thigh_vec, quaternion_from_euler_rad(0, 0, -hip_angle))

    hip_end_pos = get_relative_hip_end_pos(hip_angle, joint_name)
    knee_pos = np.add(hip_end_pos, thigh_vec)
    return knee_pos


def get_absolute_hip_base_pos(joint_name, body_rotation):
    return rotate_by_quaternion(HIP_OFFSETS[joint_name], body_rotation)


temp_period = 3000
# stuff for testing
def generate_target_pos(joint_name):
    #target_pos_center = [0.2, 0.2, -0.2]
    target_pos_center = [HIP_OFFSETS[joint_name][0]-0.05, HIP_OFFSETS[joint_name][1] * 4, -0.2]

    # create circle path around target_pos_center
    temp_counter = int(time() * 1000) % temp_period
    circle_radius_forward = 0.04
    circle_radius_side = 0.06
    target_pos = [target_pos_center[0] + circle_radius_forward * np.cos(temp_counter / temp_period * 2 * np.pi), target_pos_center[1] + circle_radius_side * np.sin(temp_counter / temp_period * 2 * np.pi), target_pos_center[2]]

    # move side to side in y direction
    #target_pos = [target_pos_center[0], target_pos_center[1] + 0.1 * np.sin(temp_counter / temp_period * 2 * np.pi), target_pos_center[2]]

    return target_pos




    

