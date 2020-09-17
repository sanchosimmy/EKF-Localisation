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
ax.set_xlim([-100, 20])
ax.set_ylim([-600, 20])

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


dt = 0.1

Q = np.diag([0.01, 0.01]) ** 2              # input noise
R = np.diag([0.1, np.pi/180]) ** 2      # measurement noise

X = np.array([2, 1, 0, 0])   # initial state
myrobot = Robot([2, 1, -1])
Robot.__set_map__(landmarks)
# myrobot.set_noise(0.01, 0.5, 5)


# estimated variables
x = np.array([3, 3, 1, 0])   # initial state
P = np.diag([1, 2, 1, 1]) ** 2     # state covariance
# T = np.diag([100,5,100])
# T = np.array([[100, 20, 0],[80, 100, 0],[0, 0, 0]])
# perturbation levels

q = np.sqrt(np.diag(Q)) / 2
r = np.sqrt(np.diag(R)) / 2
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

for i in range(200):

    if(i % 50 == 0):
        turn = np.random.normal() * 0.2
        forward = np.random.normal()  + 2
        # u = [-1*(u[1] + i % 3) / 3 , -1 * (u[0] + i %4) / 4 ]
    u = [-500 * np.pi * 0.01 * np.cos(2 * np.pi * 0.1 * i * delta_t),
         -500 * np.pi * 0.01 * np.sin(2 * np.pi * 0.1 * i * delta_t)]
    n = np.random.normal(loc=0, scale=q, size=2)
    X, _, _ = myrobot.predict(X, u, n, delta_t)
    v = np.random.normal(loc=0, scale=r, size=2)
    y, _ = myrobot.sense_linear(X)
    y += v
    # print(Z.shape)

    # estimate prediction
    # print('x : {} shape: {} y : {} shape : {}'.format(x, x.shape, y, y.shape))
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
    #pdb.set_trace()

    #pdb.set_trace()
    # print(x)
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
