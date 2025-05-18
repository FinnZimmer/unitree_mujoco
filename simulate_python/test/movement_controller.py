from time import time
import numpy as np
from step_task import StepTask
from transform import rotate_by_quaternion, invert_unit_quaternion, quat_to_euler, quaternion_from_x_rotation, quaternion_from_y_rotation
from kinematics import get_relative_hip_end_pos
from inv_kinematics import get_angles_for_target


LEG_DIRECTIONS = {
    "FR": np.array([1, -1, 0]),
    "FL": np.array([1, 1, 0]),
    "RR": np.array([-1, -1, 0]),
    "RL": np.array([-1, 1, 0])
}


class MovementController:
    def __init__(self, robot_controller):
        self.step_tasks = {}
        self.stepping_foot_idx = None

        self.robot_controller = robot_controller
        self.default_foot_pos = robot_controller.default_foot_pos
        self.hip_offsets = robot_controller.hip_offsets
        self.legs = robot_controller.legs
        self.level_gain = 0.3  # keeps robot level
        self.equal_leg_length_gain = 5.0  # keeps height of all hip joints above ground constant
        self.body_shift_gain = 0.1  # keeps robot centered within foot contact area
        self.default_hip_height = 0.3
        self.default_stance = np.array([0.011, 0, -0.289])

        self.min_step_distance = 0.06  # minimum offset from target position before taking a step
        self.step_time = 0.2
        self.step_height = 0.1
        self.step_distance = 0.06


    def update_target_angles(self, dt, movement_vector):
        self.last_update_time = time()

        self.update_step_tasks(dt, movement_vector)
        self.adjust_body_pos(dt, movement_vector)

        return self.calc_target_angles()


    def update_step_tasks(self, dt, movement_vector):
        if not self.step_tasks:
            self.stand_up(-0.1, 1)

        stance_center = self.calc_stance_center()
        current_body_shift = stance_center - self.default_stance

        # position feet
        active_step_tasks = [task for task in self.step_tasks.values() if task.active]
        if len(active_step_tasks) == 0:

            raw_target_foot_pos = np.array([np.array(self.default_foot_pos[leg_name]) * 0.3 + np.array(self.get_default_foot_pos_rot_rp(leg_name)) * 0.7 for leg_name in self.legs])
            raw_target_foot_pos_abs = np.array([rotate_by_quaternion(raw_target_foot_pos[i], self.robot_controller.body_state.quaternion) for i in range(4)])
            
            # TODO adjust absolute target foot positions for body shift?
            #raw_target_foot_pos_abs[:, :2] -= current_body_shift[:2] * 1.0

            raw_target_foot_pos = np.array([rotate_by_quaternion(raw_target_foot_pos_abs[i], invert_unit_quaternion(self.robot_controller.body_state.quaternion)) for i in range(4)])

            foot_pos_rel = np.array([self.robot_controller.feet_relative[leg_name]["pos"] for leg_name in self.legs])
            foot_offsets = raw_target_foot_pos - foot_pos_rel
            
            body_shift_direction_bonuses = np.array([np.dot(current_body_shift[:2], LEG_DIRECTIONS[leg_name][:2]) for leg_name in self.legs])

            max_foot_offset_idx = 0
            next_stepping_foot_idx = None
            if self.stepping_foot_idx is None:
                # slightly favor the legs in the direction of the movement
                foot_offset_dists = [np.linalg.norm(foot_offset[:2]) - body_shift_direction_bonuses[i] * 0.1 for i, foot_offset in enumerate(foot_offsets)]
                next_stepping_foot_idx = np.argmax(foot_offset_dists)
            else:
                if movement_vector[0] < 0 or movement_vector[1] > 0:
                    next_stepping_foot_idx = (self.stepping_foot_idx + 1) % 4
                elif movement_vector[1] < 0 or movement_vector[0] > 0:
                    next_stepping_foot_idx = (self.stepping_foot_idx - 1) % 4
                else:
                    # TODO optimize stepping pattern for balancing
                    next_stepping_foot_idx = (self.stepping_foot_idx + 1) % 4

            # Reset stepping foot idx if all feet are within min step distance
            #if np.linalg.norm(foot_offsets[next_stepping_foot_idx]) <= self.min_step_distance and np.linalg.norm(movement_vector) <= 0.01:
            #    self.stepping_foot_idx = None
            #    return
            
            target_foot_pos = self.robot_controller.feet_relative[self.legs[next_stepping_foot_idx]]["pos"] + foot_offsets[next_stepping_foot_idx]

            if np.linalg.norm(foot_offsets[next_stepping_foot_idx]) > self.min_step_distance or np.linalg.norm(movement_vector) > 0.01:
                # TODO clip movement vector to some max length max_step_length
                target_foot_pos += movement_vector * self.step_distance * 1.0
                self.step_tasks[self.legs[next_stepping_foot_idx]] = StepTask(self.robot_controller.feet_relative[self.legs[next_stepping_foot_idx]]["pos"], target_foot_pos, self.step_time, self.step_height)
                self.stepping_foot_idx = next_stepping_foot_idx


    # position body relative to feet
    def adjust_body_pos(self, dt, movement_vector):
        current_body_shift = self.calc_stance_center() - self.default_stance
        #print("current_body_shift: ", np.linalg.norm(current_body_shift))
        #print("#")
        #print("current_body_shift: ", current_body_shift)
        #print("rpy: ", quat_to_euler(self.robot_controller.body_state.quaternion))

        # most of this, if not all, could be done outside of a loop with matrices
        for leg_name in self.legs:
            if self.step_tasks[leg_name].active:
                continue

            target_foot_pos_relative = self.step_tasks[leg_name].calc_target_pos(time())
            hip_base_pos_abs = rotate_by_quaternion(get_relative_hip_end_pos(self.robot_controller.joint_state[self.robot_controller.actuator_map[f"{leg_name}_hip"]].q, leg_name), self.robot_controller.body_state.quaternion)
            target_foot_pos_abs = rotate_by_quaternion(target_foot_pos_relative, self.robot_controller.body_state.quaternion)

            # adjust for roll and pitch to keep robot level
            temp = target_foot_pos_abs[2]
            roll, pitch, yaw = quat_to_euler(self.robot_controller.body_state.quaternion)
            target_foot_pos_abs[2] -= pitch * self.level_gain * (1 if leg_name in ["FR", "FL"] else -1) * dt
            target_foot_pos_abs[2] -= roll * self.level_gain * (1 if leg_name in ["FR", "RR"] else -1) * dt
            #print(leg_name)
            #print("target pos rp adjust: ", temp - target_foot_pos_abs[2])

            # keep leg lengths constantish
            height_diff = target_foot_pos_abs[2] - hip_base_pos_abs[2]
            #print("leg_height: ", leg_name, height_diff)

            # adjust for egual leg length
            target_foot_pos_abs[2] -= (1 - height_diff / -self.default_hip_height) * self.equal_leg_length_gain * dt

            # adjust for body shift to keep robot centered within foot contact area
            current_body_shift_abs = rotate_by_quaternion(current_body_shift, self.robot_controller.body_state.quaternion)
            target_foot_pos_abs -= current_body_shift_abs * self.body_shift_gain * dt

            # shift body towards movement vector
            movement_vector_abs = rotate_by_quaternion(movement_vector, self.robot_controller.body_state.quaternion)
            target_foot_pos_abs -= movement_vector_abs * 0.15 * dt

            target_foot_pos_relative = rotate_by_quaternion(target_foot_pos_abs, invert_unit_quaternion(self.robot_controller.body_state.quaternion))
            self.step_tasks[leg_name].target_pos = target_foot_pos_relative


    # TODO refine this
    def stand_up(self, step_height, step_duration):
        self.step_tasks = {leg_name : StepTask(self.robot_controller.feet_relative[leg_name]["pos"], np.array(self.default_foot_pos[leg_name]), step_duration, step_height) for leg_name in self.legs}


    def calc_target_angles(self):
        target_angles = {}
        for leg_name in self.legs:
            target_foot_pos_relative = self.step_tasks[leg_name].calc_target_pos(time())
            target_hip, target_thigh, target_calf = get_angles_for_target(target_foot_pos_relative, leg_name)
            target_angles[leg_name] = {"hip": target_hip, "thigh": target_thigh, "calf": target_calf}
        
        return target_angles
    

    def calc_stance_center(self):
        body_shift = np.zeros(3)
        num_feet_on_ground = 0

        for leg_name in self.legs:
            # TODO: only use feet on ground?
            #if self.robot_controller.feet[leg_name]["on_ground"]:
            body_shift = np.add(body_shift, self.get_feet_relative_rot_rp(leg_name))
            num_feet_on_ground += 1

        #if num_feet_on_ground <= 0:
        #    return np.zeros(3)

        body_shift /= num_feet_on_ground
        return body_shift
    

    # TODO: This is wrong, but works well enough for now
    def get_default_foot_pos_rot_rp(self, leg_name):
        roll, pitch, yaw = quat_to_euler(self.robot_controller.body_state.quaternion)
        rot2 = quaternion_from_x_rotation(-roll)
        rot = quaternion_from_y_rotation(-pitch)

        res_pos = rotate_by_quaternion(self.default_foot_pos[leg_name], rot2)
        res_pos = rotate_by_quaternion(res_pos, rot)

        return res_pos
    

    # TODO: This is wrong, but works well enough for now
    def get_feet_relative_rot_rp(self, leg_name):
        roll, pitch, yaw = quat_to_euler(self.robot_controller.body_state.quaternion)
        rot2 = quaternion_from_x_rotation(-roll)
        rot = quaternion_from_y_rotation(-pitch)

        res_pos = rotate_by_quaternion(self.robot_controller.feet_relative[leg_name]["pos"], rot2)
        res_pos = rotate_by_quaternion(res_pos, rot)

        return res_pos
