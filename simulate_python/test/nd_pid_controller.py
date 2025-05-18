from time import time
import numpy as np

class ND_PIDController:
    def __init__(self, kp, ki, kd, nd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.nd = nd
        self.integral = np.zeros(self.nd)
        self.last_error = np.zeros(self.nd)
        self.last_update_time = None

    def update(self, error):
        if self.last_update_time:
            dt = time() - self.last_update_time
            self.integral += error * dt
            self.integral = np.clip(self.integral, -0.05, 0.05)
            derivative = (error - self.last_error) / dt
        else:
            self.integral = np.clip(self.integral, -0.05, 0.05)
            derivative = np.zeros(self.nd)
        
        self.last_update_time = time()
        self.last_error = error

        #print("e: ", error[1], "i: ", self.integral[1], "d: ", derivative[1])
        return self.kp * error + self.ki * self.integral - self.kd * derivative
    
    def reset(self):
        self.last_update_time = time()
        self.integral = np.zeros(self.nd)
        self.last_error = np.zeros(self.nd)
