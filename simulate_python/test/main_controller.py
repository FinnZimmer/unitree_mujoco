import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from robot_controller import RobotController
import numpy as np


DEBUG_PRINT = False
PRINT_TIMEOUT = 1
last_print_time = time.time()


movement_speed = 1.0

def get_keyboard_input():
    key = input("Enter direction (w/a/s/d): ").lower()
    if key == 'w':
        robot_controller.movement_direction = np.array([movement_speed * 1.5, 0.0, 0.0])
    elif key == 's':
        robot_controller.movement_direction = np.array([-movement_speed * 0.7, 0.0, 0.0])
    elif key == 'a':
        robot_controller.movement_direction = np.array([0.0, movement_speed, 0.0])
    elif key == 'd':
        robot_controller.movement_direction = np.array([0.0, -movement_speed, 0.0])
    else:
        robot_controller.movement_direction = np.array([0.0, 0.0, 0.0])

# for testing/debugging
def HighStateHandler(msg: SportModeState_):
    print("High state update")
    print(msg)


# for testing/debugging
def LowStateHandler(msg: LowState_):
    global last_print_time

    if not DEBUG_PRINT:
        return

    if time.time() - last_print_time < PRINT_TIMEOUT:
        return
    last_print_time = time.time()

    print(msg)
    
    # Calculate and print foot positions
    #print("\n=== Foot Positions ===")
    #print("FR foot:", get_foot_pos(msg.imu_state.quaternion, msg.motor_state[0].q, msg.motor_state[1].q, msg.motor_state[2].q, "FR"))
    #print("FL foot:", get_foot_pos(msg.imu_state.quaternion, msg.motor_state[3].q, msg.motor_state[4].q, msg.motor_state[5].q, "FL"))
    #print("RR foot:", get_foot_pos(msg.imu_state.quaternion, msg.motor_state[6].q, msg.motor_state[7].q, msg.motor_state[8].q, "RR"))
    #print("RL foot:", get_foot_pos(msg.imu_state.quaternion, msg.motor_state[9].q, msg.motor_state[10].q, msg.motor_state[11].q, "RL"))
    #print("=====================\n")
    
    print(msg.motor_state[0])
    print("\n=== Current Sensor Data ===")
    print("IMU Data:")
    print(f"  Quaternion: {msg.imu_state.quaternion}")
    print(f"  Gyroscope: {msg.imu_state.gyroscope}")
    print(f"  Accelerometer: {msg.imu_state.accelerometer}")
    print(f"  RPY: {msg.imu_state.rpy}")
    
    print("\nMotor States:")
    for i in range(12):  # Print first 12 motors (leg motors)
        print(f"Motor {i}:")
        print(f"  Position: {msg.motor_state[i].q}")
        print(f"  Velocity: {msg.motor_state[i].dq}")
        print(f"  Torque: {msg.motor_state[i].tau_est}")
    
    print("========================\n")



if __name__ == "__main__":
    robot_controller = RobotController()

    ChannelFactoryInitialize(1, "lo0")
    hight_state_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)

    hight_state_subscriber.Init(HighStateHandler, 10)
    #low_state_subscriber.Init(LowStateHandler, 10)
    low_state_subscriber.Init(robot_controller.handle_low_state_update, 10)

    low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    low_cmd_publisher.Init()

    robot_controller.low_cmd_publisher = low_cmd_publisher
    
    robot_controller.movement_controller.movement_direction = np.zeros(3)
    
    #cmd = unitree_go_msg_dds__LowCmd_()
    #print(cmd)


    while True:
        get_keyboard_input()
        time.sleep(0.01)
