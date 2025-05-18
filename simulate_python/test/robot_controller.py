from time import time
import numpy as np

from kinematics import get_relative_foot_pos
from transform import rotate_by_quaternion
from movement_controller import MovementController
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from nd_pid_controller import ND_PIDController
from kinematics import HIP_OFFSETS


# Actuator mapping dictionary
ACTUATOR_MAP = {
    'FR_hip': 0,
    'FR_thigh': 1,
    'FR_calf': 2,
    'FL_hip': 3,
    'FL_thigh': 4,
    'FL_calf': 5,
    'RR_hip': 6,
    'RR_thigh': 7,
    'RR_calf': 8,
    'RL_hip': 9,
    'RL_thigh': 10,
    'RL_calf': 11
}

# Legs
LEGS = ["FR", "FL", "RR", "RL"]

DEFAULT_FOOT_POS = {
    "FR": [0.23, -0.17, -0.27],
    "FL": [0.23, 0.17, -0.27],
    "RR": [-0.23, -0.17, -0.27],
    "RL": [-0.23, 0.17, -0.27],
}


temp_counter = 0
circle_period = 500
circle_amplitude_forward = 0.1
circle_amplitude_sideways = 0.06

class RobotController:
    def __init__(self):
        self.sensor_data = None
        self.body_state = None
        self.joint_state = None

        self.feet = {"FR" : {},
                     "FL" : {},
                     "RR" : {},
                     "RL" : {}}
        self.feet_relative = {"FR" : {},
                              "FL" : {},
                              "RR" : {},
                              "RL" : {}}
                
        self.last_low_state_update_time = None
        self.low_cmd_publisher = None
        self.crc = CRC()

        self.stepping_foot_idx = None
        self.default_foot_pos = DEFAULT_FOOT_POS
        self.hip_offsets = HIP_OFFSETS
        self.legs = LEGS
        self.actuator_map = ACTUATOR_MAP
        # TODO would it make sense to have different gains for different axes? i.e. higher for vertical axis?
        self.joint_pid_controller = ND_PIDController(kp=0.08, ki=2.0, kd=0.01, nd=len(ACTUATOR_MAP))
        self.movement_controller = MovementController(self)

        self.movement_direction = np.zeros(3)


    def handle_low_state_update(self, msg: LowState_):
        if self.last_low_state_update_time:
            delta_time = time() - self.last_low_state_update_time
        else:
            delta_time = 1000000

        self.last_low_state_update_time = time()

        self.sensor_data = msg
        self.body_state = msg.imu_state
        self.joint_state = msg.motor_state

        self.update_feet(delta_time)
        target_angles = self.movement_controller.update_target_angles(delta_time, self.movement_direction)
        self.apply_pid_control(target_angles)


    def update_feet(self, delta_time):
        for leg_name in LEGS:
            relative_foot_pos = get_relative_foot_pos(self.joint_state[ACTUATOR_MAP[f"{leg_name}_hip"]].q,
                                                      self.joint_state[ACTUATOR_MAP[f"{leg_name}_thigh"]].q,
                                                      self.joint_state[ACTUATOR_MAP[f"{leg_name}_calf"]].q,
                                                      leg_name)
            foot_pos = rotate_by_quaternion(relative_foot_pos, self.body_state.quaternion)
            
            # TODO calculate velocity based on dq in joint_state (and imu_state for absolute velocity)
            self.feet_relative[leg_name]["vel"] = (np.subtract(relative_foot_pos, self.feet_relative[leg_name].get("pos", relative_foot_pos))) / delta_time
            self.feet_relative[leg_name]["pos"] = relative_foot_pos

            self.feet[leg_name]["vel"] = (np.subtract(foot_pos, self.feet[leg_name].get("pos", foot_pos))) / delta_time
            self.feet[leg_name]["pos"] = foot_pos

        lowest_foot = min(self.feet, key=lambda a: self.feet[a]["pos"][2])
        for leg_name in LEGS:
            self.feet[leg_name]["on_ground"] = self.feet[leg_name]["pos"][2] < 0 and self.feet[leg_name]["pos"][2] < self.feet[lowest_foot]["pos"][2] + 0.01
    

    def apply_pid_control(self, target_angles):
        target_angle_arr = joint_dict_to_array(target_angles)
        current_angle_arr = np.array([self.joint_state[i].q for i in range(len(ACTUATOR_MAP))])
        error = np.subtract(target_angle_arr, current_angle_arr)

        error = self.joint_pid_controller.update(error)
        target_angle_arr = np.add(target_angle_arr, error)
        self.publish_low_cmd(target_angle_arr)


    def publish_low_cmd(self, target_angle_arr):
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0]=0xFE
        cmd.head[1]=0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(len(target_angle_arr)):
            cmd.motor_cmd[i].q = target_angle_arr[i]
            cmd.motor_cmd[i].kp = 50.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 1.5
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)

        if self.low_cmd_publisher:
            self.low_cmd_publisher.Write(cmd)


def joint_dict_to_array(joint_dict):
    arr = np.zeros(len(joint_dict) * 3)
    for i, leg_name in enumerate(LEGS):
        arr[i * 3] = joint_dict[leg_name]["hip"]
        arr[i * 3 + 1] = joint_dict[leg_name]["thigh"]
        arr[i * 3 + 2] = joint_dict[leg_name]["calf"]
    return arr
