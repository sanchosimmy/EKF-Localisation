import numpy as np
def observation_model(state):
    """observation model for the robot
    z = h(x,y) = [ (px^2 + py^2) ^ (1/2), arctan(py/px)]

    :state: current state [x, y, yaw, v]
    :returns: [z, jac_h_x]

    """
    x, y, _, _ = state
    diff = MAP_L - np.array([x,y])
    z = [np.hypot(diff[0, 0], diff[0, 1]), np.arctan2(diff[0, 1], diff[0, 0])]
    t = (diff[0, 1]/diff[0, 0]) ** 2 + 1
    h_wrt_x = np.array([[diff[0, 0] / z[0], diff[0, 1] /z[0], 0, 0],
                        [-diff[0, 1] / ((diff[0, 0] **2) * t), 1/(diff[0, 0] * t), 0, 0]])
    return [z, h_wrt_x]

