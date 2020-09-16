#!/usr/bin/env python3
"""
    Adapted to python from JoanSola EKF in octave
    http://www.iri.upc.edu/people/jsola/JoanSola/objectes/curs_SLAM/SLAM2D/SLAM%20course.pdf


   KF Kalman Filter

   I. System
        x+ = f(x, u, n)
        y  = h(x) + v

        x : state vector            - P : covariance matrix
        u : control vector
        n : perturbation vector     - Q : covariance matrix
        y : meansurement vector
        v : measurement noise       - R : covariance matrix

        f() : transition function
        h() : measurement function
        H   : measurement matrix

    II. Initialisation

        Define f(), and h()

        precise x, P, Q, R


    III. Temporal loop

        IIIa. Prediction of mean value of x at the arrival of u

            Jacobian computation
            F_x : Jac. of x+ wrt to state
            F_u : Jac. of x+ wrt to control
            F_n : Jac. of x+ wrt to perturbation

            x+ = f(x, u, n)
            P+ = F_x * P F_x' + F_n * Q * F_n'

        IIIb. correction of mean(x) and P at the arrival of y

            Jacobian computation
            H : jac. of y wrt to x

            e  = h(x)                 ...(expected measurement)
            E  = H * P * H'

            z  =  y - e              ...(innovation/ new information)
            Z  = R + E

            K  =  P * H' * Z^(-1)    ...(Kalman gain)
            x+ = x + K * z

            P+ = P - K * H * P  // P - K * Z * K'  // and Joseph form

        IV. Plot results

        V.  How to setup KF examples

            1. Simulate system, get x, u and y trajectories

            2. Estimate x with the KF. Get x and P trajectories

            3. Plot results.
"""

import numpy as np


class EKF(object):
    """
     define system
     x = [px, py, vx, vy]'
     y = [range, bearing]'
     n = [nx, ny]'
     u = [ux, uy]'
     r = [rd, ra]'
     px+ = px + vx *dt
     py+ = py + vy * dt
     vx+ = vx + ax*dt + nx
     vy+ = vy + ay*dt + ny

     range = sqrt(px^2 + py^2) + rd
     bearing = atan2(py, px) + ra
    """

    def __init__(self, Q, R, P):
        """
        args: Cov Matrices
            :Q: input pertrubations
            :R: measurement noise
            :P: process noise (uncertainity in state)
        """
        self.__Q = Q                #pylint: disable-msg=C0103
        self.__R = R                #pylint: disable-msg=C0103
        self.__R = P                #pylint: disable-msg=C0103
        self.delta_t = 0.1
        self.cur_state = np.zeros(1, 4)

    def set_simulation_params(self, delta_t):
        """ set simulation time step """
        self.delta_t = delta_t

    def predict(self, ain):
        """predict next state based on previous state and state model
        :vin: input acceleration
        """
        x, F_x, F_u, F_n = Robot.__


if(__name__ == '__main__'):
    pass
