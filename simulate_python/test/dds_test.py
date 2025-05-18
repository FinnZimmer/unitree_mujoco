from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
import time

def handler(msg):
    print("[TEST] Received lowstate:", msg)

ChannelFactoryInitialize(0)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)

while True:
    time.sleep(1)
