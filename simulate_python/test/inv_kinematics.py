from kinematics import HIP_OFFSETS, HIP_VEC, HIP_LENGTH, THIGH_LENGTH, CALF_LENGTH
import numpy as np
from transform import rotate_by_quaternion, quaternion_from_euler_rad


# target pos is relative to the body frame
def get_angles_for_target(target_pos, joint_name):
    right_side = joint_name == "FR" or joint_name == "RR"

    hip_base_pos = HIP_OFFSETS[joint_name]

    # Mujoco and Unitree use right-handed coordinate system with z-axis pointing up
    target_vec = np.subtract(target_pos, hip_base_pos)

    # Hip angle calculation
    vertical_diff = target_vec[2]
    side_diff = target_vec[1]
    forward_diff = target_vec[0]
    
    a = side_diff**2 + vertical_diff**2 - HIP_LENGTH**2
    if a > 0:
        sqrt_a = np.sqrt(a)
    else:
        sqrt_a = 0
    angle1 = np.pi + np.arctan2(vertical_diff, side_diff)
    angle2 = np.arctan2(sqrt_a, HIP_LENGTH * (-1 if right_side else 1))
    hip_angle = -(np.pi - angle1 - angle2)

    # Get hip end position after hip rotation
    hip_vec = np.multiply(HIP_VEC[joint_name], [HIP_LENGTH])
    hip_vec = rotate_by_quaternion(hip_vec, quaternion_from_euler_rad(0, 0, -hip_angle))
    hip_end_pos = np.add(hip_base_pos, hip_vec)

    # Calculate leg length (distance from hip end to target)
    leg_length = np.linalg.norm(np.subtract(target_pos, hip_end_pos))
    # Check if target is reachable
    if leg_length > THIGH_LENGTH + CALF_LENGTH:
        print("Warning: Target position is too far to reach")
        leg_length = THIGH_LENGTH + CALF_LENGTH
    
    # Calculate calf angle using law of cosines
    cos_alpha = (THIGH_LENGTH**2 + CALF_LENGTH**2 - leg_length**2) / (2 * THIGH_LENGTH * CALF_LENGTH)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    # Fix phase by using negative of the angle and adjusting the reference point
    calf_angle = np.pi - np.arccos(cos_alpha)

    G = [hip_end_pos[0], target_vec[1], target_vec[2]]
    GC = target_pos[0] - hip_end_pos[0]
    AG = -np.linalg.norm(np.subtract(G, hip_end_pos))

    angle3 = np.arctan2(GC, -AG)
    angle4 = np.arctan2(CALF_LENGTH * np.sin(calf_angle), THIGH_LENGTH + CALF_LENGTH * np.cos(calf_angle))
    thigh_angle = (angle3 - angle4)

    return hip_angle, -thigh_angle, -calf_angle