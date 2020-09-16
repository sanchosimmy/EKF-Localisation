#!/usr/bin/env python3
""" script to start EKF localisation """

import pdb

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np


from myrobot import Robot
# from localise_ekf import EKF

# pylint:disable-msg=C0103
landmarks = [[25.0, 30.0], [90.0, 80.0], [10.0, 80.0], [80.0, 10.0]]

p_pred = []
path_actual = []
p_act = []
plt.ion()
fig, ax = plt.subplots(1, 1)
# ax.set_autoscaley_on(True)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_xlim([-200, 1200])
ax.set_ylim([-200, 800])

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
p = []
N = 5000



X = np.array([50.0, 50.0, 0.0, 0])   # initial state
myrobot = Robot([50, 50, 0])
Robot.__set_map__(landmarks)
myrobot.set_noise(0.01, 0.5, 5)

Q = np.diag([0.01, 0.5]) ** 2              # input noise
R = np.diag([5.0]) ** 2      # measurement noise

# estimated variables
x = np.array([50.0, 50.0, 0.0, 0])   # initial state
P = np.diag([0.5, 0.5, 0.01]) ** 2     # state covariance
# T = np.diag([100,5,100])
# T = np.array([[100, 20, 0],[80, 100, 0],[0, 0, 0]])
# perturbation levels

q = np.sqrt(np.diag(Q)) / 2
r = np.sqrt(np.diag(R)) / 2
Sigma = np.diag([50, 50, np.pi * 2, 0.1]) ** 2
p_x = list("")
p_y = list("")
p_x.append(x[0])
p_y.append(x[1])
pt_x = list("")
pt_y = list("")
pt_x.append(x[0])
pt_y.append(x[1])

for i in range(50):

    if(i % 50 == 0):
        turn = np.random.normal() * 0.2
        forward = np.random.normal()  + 2
        turn, forward = 0.0, 2
    noise = [np.random.normal(0, q[0]), np.random.normal(0, q[1])]
    X, _, _ = myrobot.predict(X, [turn, forward], noise)
    y, _ = myrobot.sense_linear(X, np.random.normal(0, r, 4))
    # print(Z.shape)

    # estimate prediction
    # print('x : {} shape: {} y : {} shape : {}'.format(x, x.shape, y, y.shape))
    x, F_x, F_n = myrobot.predict(x, [turn, forward], [0, 0])
    Sigma = np.linalg.multi_dot((F_x, Sigma, F_x.T)) + \
            np.linalg.multi_dot((F_n, Q, F_n.T))

    # correction step
    e, H = myrobot.sense_linear(x, [0,0,0,0])

    z = y - e               # innovations
    x_adv = np.array(x)
    P_adv = np.zeros(16, dtype=np.float).reshape(4, 4)
    for j, Hi in enumerate(H):
        E = np.linalg.multi_dot((Hi, Sigma, Hi.T))
        Z = E + R
        K = np.dot(Sigma, Hi.T) / Z
        # print('x_adv : {}, K : {}, Padv : {} shape : {}'.format(x_adv.shape, K.shape, P_adv.shape, z[j].shape))
        x_adv += K.reshape(4) * z[j]
        P_adv += np.dot(K, Hi)
        #pdb.set_trace()

    #pdb.set_trace()
    x = np.array(x_adv)
    print(x)
    Sigma = np.dot((np.eye(4) - P_adv), Sigma)
    # print('z:{}, H: {}, y: {}, e: {} E: {}'.format(z.shape, H.shape, y.shape, e.shape, E.shape))
    # line_particles.set_data([K[0], 5])
    # print(x - myrobot.pose)
    # print(type(Sigma), Sigma.shape, Sigma)
    # ------------------------------------------------------
    p_x.append(x[0])
    p_y.append(x[1])
    line_path.set_data([X[0], X[1]])
    line_est.set_data(p_x, p_y)
    pt_x.append(X[0])
    pt_y.append(X[1])
    line_true.set_data(pt_x, pt_y)

    # pearson = T[0, 1]/np.sqrt(T[0, 0] * T[1, 1])
    # ell_radius_x = np.sqrt(1 + pearson)
    # ell_radius_y = np.sqrt(1 - pearson)
    # ellipse = Ellipse((x[0],x[1]), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='red')
    # ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='red')
    # scale_x = np.sqrt(Sigma[0, 0])
    # 
    # # calculating the stdandarddeviation of y from  the squareroot of the variance
    # # np.sqrt(cov[1, 1])
    # scale_y = np.sqrt(Sigma[1, 1])
    # mean_x, mean_y = x[0], x[1] 
    # transf = transforms.Affine2D() \
    #     .rotate_deg(45) \
    #     .scale(scale_x, scale_y)  \
    #     .translate(mean_x, mean_y)
    # ellipse.set_transform(transf + ax.transData)
    # ax.add_patch(ellipse)


    # print(my_pf.particles.shape,'test')
    fig.canvas.draw()
    fig.canvas.flush_events()
#part = np.array(p_pred, dtype=np.float)
#print(type(path[0]), path.shape)
#print(type(part), part.shape, part[0].shape, part[:,1].shape)
#print(kl)
## plt.scatter(r[:,0],r[:,1])
## plt.scatter(t[:,0],t[:,1])
# print(path_actual,path_pred) #Leave this print statement for grading purposes!
