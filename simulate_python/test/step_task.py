import numpy as np

from time import time
from scipy.interpolate import CubicSpline


class StepTask:
    def __init__(self, start_pos, target_pos, step_time, step_height):
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.step_time = step_time
        self.start_time = time()
        self.step_height = step_height
        self.active = True

        step_peak = max(self.start_pos[2], self.target_pos[2]) + self.step_height
        spline_coords = np.array([
            [0, self.start_pos[2]],
            [0.5, step_peak],
            [1, self.target_pos[2]]
        ])
        self.spline = CubicSpline(spline_coords[:, 0], spline_coords[:, 1])


    def calc_target_pos(self, t):
        if t > self.start_time + self.step_time:
            self.active = False
            return self.target_pos
        
        dt = (t - self.start_time) / self.step_time
        
        # interpolate horizontal components linearly
        interp_pos = self.start_pos + (self.target_pos - self.start_pos) * dt
        
        # interpolate vertical component in the shape of a spline, starting at current_pos, reaching a peak int he middle and ending at target_pos
        interp_pos[2] = self.spline(dt)
        return np.array(interp_pos)



