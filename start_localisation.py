#!/usr/bin/env python3
""" script to perform EKF localisation
This module uses a simple motion model,
x+      =    x + v cos(yaw) * delta_t
y+      =    y + v sin(yaw) * delta_t
yaw+    =    yaw + yaw_rate * delta_t
using notations from S Thrun Probabilistic robotics
sensor model : range and bearing model.

Example:
    To run the code use. You can modify the module attributes to check
    the EKF working

      $ python3 start_localisation.py

Attributes:
    X(numpy.ndarray): true state vector(internal to robot). consists of
        [x, y, yaw/phi, vel]

    u(list): input command given to robot -[lin_vel, yaw_rate]

    INPUT_NOISE(numpy.ndarray): covariance matrix for perturbing
        input linear velocity and yaw_rate

    Q(numpy.ndarray): covariance matrix for obervation noise
        [range noise, bearing noise]

    R(numpy.ndarray): covariance matrix for Process noise

    G(numpy.ndarray): Jacobian of system function

    H(numpy.ndarray): Jacobian of observation function
"""

# import pdb
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np


from myrobot import Robot
# from localise_ekf import EKF

# pylint:disable-msg=C0103
landmarks = [[25.0, 30.0], [90.0, 80.0], [10.0, 80.0], [80.0, 10.0], [-50., -50.],[-50., 50.], [50, -50], [-10, -20], [10, -20], [20, -10]]

p_pred = []
path_actual = []
p_act = []
plt.ion()
fig, ax = plt.subplots(1, 1)
# ax.set_autoscaley_on(True)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])

ax.grid()
ax.legend()
line_particles = []
line_est = []
line_true = []
line_path = []
line_path, = ax.plot([], [], 'ro', label="waypoint", ms=25, alpha=0.8)
line_est, = ax.plot([], [], 'k.', label="waypoint", ms=10)
line_true, = ax.plot([], [], 'r.', label="waypoint", ms=5)
line_particles, = ax.plot([], [], 'g.', label="particles", ms=5)
MAP_L = np.array(landmarks, dtype=np.float)
ax.scatter(MAP_L[:, 0], MAP_L[:, 1], c='k', marker='*', label="landmarks")



INPUT_NOISE = np.array([0.01, np.deg2rad(30)])              # input noise std
Q = np.diag([3, np.deg2rad(30.0)]) ** 2                   # measurement noise
R = np.diag([1.0, 2.0, np.pi/180, 1.0]) ** 2                      # state covariance

X = np.array([20, 10, 0, 0])                                  # initial state
myrobot = Robot([2, 11, 0])
Robot.__set_map__(landmarks)

# estimated variables
x = np.array([30, 8, np.deg2rad(0.8), 0])   # initial state
Sigma = np.eye(4, dtype=np.float)   * 40                       # intial cov

q = np.sqrt(np.diag(Q)) / 2 * 4
r = np.sqrt(np.diag(R)) / 2 * 2
delta_t = 0.1
p_x = list("")
p_y = list("")
p_x.append(x[0])
p_y.append(x[1])
pt_x = list("")
pt_y = list("")
pt_x.append(X[0])
pt_y.append(X[1])
u = [0, 0]

def ekf_correction(x, P, u, y, delta_t):
    """TODO: Docstring for ekf_correction.

    :x: state_vector (mean)
    :P: covariance
    :y: measurement
    :u: input [turn, forward]
    :delta_t: time interval
    :returns: [next_state, P]

    """
    x, F_x, F_n = myrobot.predict(x, u, [0, 0], delta_t)
    P = np.linalg.multi_dot((F_x, P, F_x.T)) + \
            np.linalg.multi_dot((F_n, Q, F_n.T))

    # correction step
    e, H = myrobot.sense_linear(x)
    E = np.linalg.multi_dot((H, P, H.T))

    z = y - e               # innovations
    Z = E + R
    # for j, Hi in enumerate(H):
    K = np.linalg.multi_dot((P, H.T, np.linalg.inv(Z)))
     # print('x_adv : {}, K : {}, Padv : {} shape : {}'.format(x_adv.shape, K.shape, P_adv.shape, z[j].shape))
    x += np.dot(K, z)
    P -= np.linalg.multi_dot((K, H, P))
    return [x, P]

def motion_model(state, cmd, delta_t):
    """advance the robot according to the input cmd
       x_{t}    = x_{t-1} + v * cos(yaw) * delta_t
       y_{t}    = y_{t-1} + v * sin(yaw) * delta_t
       yaw_{t}  = yaw_{t-1} + v * sin(yaw) * delta_t

    :state: [x, y, yaw, v]
    :cmd: [v, yaw_rate]
    :noise: [noise_v, noise_yaw_rate]
    :delta_t: time step
    :returns: [next_state, jac(motion_model)_{state}

    """
    x, y, yaw, _ = state
    v, yaw_rate = cmd
    x += v * np.cos(yaw) * delta_t
    y += v * np.sin(yaw) * delta_t
    yaw += yaw_rate * delta_t

    next_state = [x, y, yaw, v]
    jac_g_x = np.array([
        [1.0, 0.0, -v * math.sin(yaw) * delta_t, math.cos(yaw) * delta_t],
        [0.0, 1.0, v * math.cos(yaw) * delta_t, math.sin(yaw) * delta_t],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return [next_state, jac_g_x]


def observation_model(state):
    """observation model for the robot
    z = h(x,y) = [ (px^2 + py^2) ^ (1/2), arctan(py/px)]

    :state: current state [x, y, yaw, v]
    :returns: [z, jac_h_x]

    """
    x, y, _, _ = state
    diff = np.array([x,y]) - MAP_L
    z = np.hstack((np.hypot(diff[:, 0], diff[:, 1]), np.arctan2(diff[:, 1], diff[:, 0]))).reshape(-1, len(MAP_L)).T
    index = np.argwhere(z[:,0] < 80).flatten()
    # print(index)
    return z, index

def jac_h_x(state, index):
    """returns jacobian of observation function w.r.to x

    :state: state_predicton \mu_{t}
    :z: measurement [range, bearing]
    :returns: J(h)_{x}|_{t}

    """
    x, y, _, _ = state
    diff = np.array([x,y]) - MAP_L
    z = np.hypot(diff[:, 0], diff[:, 1])
    H = list('')
    for i, _ in enumerate(z):
        Hi = np.array([[diff[i, 0] / z[i], diff[i, 1] / z[i], 0, 0],
                          [-diff[i,1] / z[i] **2, diff[i, 0] / z[i] ** 2, 0, 0]], dtype=np.float)
        H.append(Hi)
    return H

def ekf_pr(x, Sigma, u, z, delta_t, index):
    """Main EKF method

    :x: state vector [x, y, yaw, v]
    :Sigma: state covariance matrix
    :u: velocity command
    :z: observed value(measurement)
    :returns: [next state, Sigma]

    """

    # Predicting mean and covariance
    x, G = motion_model(x, u, delta_t)
    Sigma = np.linalg.multi_dot((G, Sigma, G.T)) + R

    # measurement prediction
    zPred, _ = observation_model(x)
    H = jac_h_x(x, index)
    # innovation
    y = z - zPred                                   
    Sigma_ad = np.zeros(Sigma.shape)
    for i in index:
        # print(i, index)
        S = np.linalg.multi_dot((H[i], Sigma, H[i].T))  + Q
        K = np.linalg.multi_dot((Sigma, H[i].T, np.linalg.inv(S)))
        x += np.dot(K, y[i]) / len(index)
        Sigma_ad  += np.dot(K,H[i]) / len(index)
    Sigma  =  (np.eye(len(Sigma)) - Sigma_ad) @ Sigma
    return x, Sigma


def ekf_est(x, P, u, y, delta_t):
    """TODO: Docstring for ekf_correction.

    :x: state_vector (mean)
    :P: covariance
    :y: measurement
    :u: input [turn, forward]
    :delta_t: time interval
    :returns: [next_state, P]

    """
    x, F_x, F_n = myrobot.predict(x, u, [0, 0], delta_t)
    P = np.linalg.multi_dot((F_x, P, F_x.T)) + \
            np.linalg.multi_dot((F_n, Q, F_n.T))

    # correction step
    e, H = myrobot.sense_linear(x)
    E = np.linalg.multi_dot((H, P, H.T))

    z = y - e               # innovations
    Z = E + R
    K = np.linalg.multi_dot((P, H.T, np.linalg.inv(Z)))
    x += np.dot(K, z)
    P -= np.linalg.multi_dot((K, H, P))
    return [x, P]


for i in range(500):

    if(i % 50 == 0):
        yaw_rate = np.random.normal() * 0.5 + 0.2
        v = np.random.normal() * 2 + 5
        u = np.array([v, yaw_rate])
    n = np.random.normal(loc=0, scale=q, size=2)
    X, _ = motion_model(X, u, delta_t)
    z, index = observation_model(X)
    z += np.random.randn(len(z), 2) * q
    ud = u + np.random.randn(2) * INPUT_NOISE  

    
    x, Sigma = ekf_pr(x, Sigma, u, z, delta_t, index)

    # ------------------------------------------------------
    pt_x.append(X[0])
    pt_y.append(X[1])
    line_true.set_data(pt_x, pt_y)
    if(i % 10 == 0):
        p_x.append(x[0])
        p_y.append(x[1])
        line_est.set_data(p_x, p_y)
        line_path.set_data([X[0], X[1]])

# --    ---------------------cov ellipse-------------------#
        pearson = Sigma[0, 1]/np.sqrt(Sigma[0, 0] * Sigma[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y  *2, facecolor='none', edgecolor='gray')
        scale_x = np.sqrt(Sigma[0, 0])
        
        # calculating the stdandarddeviation of y from  the squareroot of the variance
        # np.sqrt(cov[1, 1])
        scale_y = np.sqrt(Sigma[1, 1])
        mean_x, mean_y = x[0], x[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y)  \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
# --    --------------------------------cov ellipse ends-----#

        fig.canvas.draw()
        fig.canvas.flush_events()
