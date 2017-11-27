#!/usr/bin/python
"""
===========================================================

A python implementation of critical dampled system smoother

===========================================================
"""
import numpy as np

import matplotlib.pyplot as plt

class CDSSmoother:
    def __init__(self, state_dim=None, init_vel=None, omega=25.0, dt=0.01):
        if state_dim is None:
            #default
            self.state_ = np.array([0])
        else:
            self.state_ = np.zeros(state_dim)
        if init_vel is None:
            self.vel_ = 0.0 * self.state_

        self.target_ = np.array(self.state_)
        self.omega_ = omega
        self.dt_ = dt
        #by default, there is no state bounds
        self.set_state_bounds(upper=np.inf, lower=-np.inf)
        self.enable_state_limits(enable=False)
        return
    
    def set_state(self, state, vel=None):
        if len(self.target_) != len(state):
            print 'Dimension of the specified state is inconsistent with the target.'
            return False
        self.state_ = state - self.target_

        if vel is not None:
            if len(vel) == self.vel_:
                self.vel_ = vel
            else:
                print 'Dimension of the specified velocity is inconsistent with the state.'

        return True

    def set_target(self, target):
        if len(target) != len(self.state_):
            print 'Dimension of the specified target is inconsistent with the state.'
            return False

        #update state
        self.state_ += (self.target_ - target)
        self.target_  = target

        return True

    def set_dt(self, dt):
        self.dt_ = dt
        return True

    def set_omega(self, omega):
        self.omega_ = omega
        return True

    def set_state_bounds(self, upper=None, lower=None):
        if upper is not None:
            if upper==np.inf:
                self.state_bounds_upper_ = np.inf * np.ones(len(self.state_))
            elif len(upper) == len(self.state_):
                self.state_bounds_upper_ = upper
            else:
                print 'Dimension of the specified upper bounds is not consistent with the state.'
            
        if lower is not None:
            if lower==-np.inf:
                self.state_bounds_lower_ = -np.inf * np.ones(len(self.state_))
            elif len(lower) == len(self.state_):
                self.state_bounds_lower_ = lower
            else:
                print 'Dimension of the specified lower bounds is not consistent with the state.'

        return True

    def enable_state_limits(self, enable):
        self.state_lim_enable = enable
        return True

    def get_state(self):
        return self.state_ + self.target_

    def update(self, dt=None, vel_scale=1.0):
        if dt is None:
            dt = self.dt_

        #update state
        tmp_state = np.array(self.state_)
        self.state_ += self.vel_ * dt
        self.vel_ = (-self.omega_*tmp_state + (1.0 - self.omega_*dt)*(self.vel_+self.omega_*tmp_state)) * np.exp(-self.omega_*dt) * vel_scale

        if self.state_lim_enable:
            for state_idx in range(len(self.state_)):
                if self.state_[state_idx] + self.target_[state_idx] > self.state_bounds_upper_:
                    self.state_[state_idx] = self.state_bounds_upper_ - self.target_[state_idx]

                if self.state_[state_idx] + self.target_[state_idx] < self.state_bounds_lower_:
                    self.state_[state_idx] = self.state_bounds_lower_ - self.target_[state_idx]

        return self.get_state()


def cdssmoother_test():

    smoother = CDSSmoother(init_state=np.zeros(1))
    smoother.set_omega(10.0)
    #test from 0.0 to 5.0
    smoother.set_state(state=np.array([0.0]))
    smoother.set_target(target=np.array([5.0]))
    #go
    traj = np.array([smoother.update() for i in range(100)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(traj[:, 0]))*smoother.dt_, traj[:, 0])

    ax.hold(True)

    #realtime update position
    raw_input('ENTER to continue with the next test...')
    smoother.set_target(target=np.array([10.0]))
    traj = []
    for i in range(10):
        traj.append(smoother.update())
    #now a new target comes...
    smoother.set_target(target=np.array([4.0]))
    for i in range(100):
        traj.append(smoother.update())
    traj = np.array(traj)
    ax.plot(np.arange(len(traj[:, 0]))*smoother.dt_, traj[:, 0])

    plt.show()
    return


