from scipy.spatial.transform import Rotation as R
import numpy as np


def quat_to_euler(q):
    # Convert to [x, y, z, w] order used by scipy
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('xyz', degrees=False)


def quaternion_mult(q,r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]


def rotate_by_quaternion(point,q):
    r = [0] + list(point)  # Convert point [x,y,z] to quaternion [0,x,y,z]
    
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    return np.array(quaternion_mult(quaternion_mult(q,r),q_conj)[1:])


def quaternion_from_z_rotation(theta):
    w = np.cos(theta / 2)
    x = 0.0
    y = 0.0
    z = np.sin(theta / 2)
    return [w, x, y, z]


def quaternion_from_y_rotation(theta):
    w = np.cos(theta / 2.0)
    x = 0.0
    y = np.sin(theta / 2.0)
    z = 0.0
    return [w, x, y, z]


def quaternion_from_x_rotation(theta):
    w = np.cos(theta / 2.0)
    x = np.sin(theta / 2.0)
    y = 0.0
    z = 0.0
    return [w, x, y, z]


def quaternion_from_euler_rad(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_quat()


def invert_unit_quaternion(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])
