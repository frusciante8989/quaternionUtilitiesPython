import numpy as np


def wrap_to_pi(angle):
    # Method to wrap an angle to [-pi; pi]
    angle_pi = (angle - 2 * np.pi * np.floor((angle + np.pi)/(2*np.pi)))
    return angle_pi


def quat_multiply(quat_b, quat_a):
    # Method to find the production of a quaternion multiplication
    if len(quat_b) != 4 or len(quat_a) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    w0, x0, y0, z0 = quat_a
    w1, x1, y1, z1 = quat_b
    quat_mult = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                          x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                          -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                          x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    return quat_mult


def quat_conj(quat):
    # Method to find the conjugate (or inverse) of a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    w, x, y, z = quat
    quat_conjugate = (w, -x, -y, -z)
    return quat_conjugate


def quat_normalize(quat, tolerance=0.00001):
    # Method to normalize a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    mag_square = sum(q * q for q in quat)
    if abs(mag_square - 1.0) > tolerance:
        mag = np.sqrt(mag_square)
        quat_normalized = tuple(q / mag for q in quat)
        return quat_normalized
    return quat


def quat_norm(quat):
    # Method to find the norm of a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    quat_norm2 = np.sqrt(sum(q * q for q in quat))
    return quat_norm2


def quat_to_rotm(quat):
    # This method computes the Direction Cosine Matrix (DCM) by applying the Euler-Rodrigues Parameterization
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')

    quat_normalized = quat_normalize(quat)
    w, x, y, z = quat_normalized
    rotm = np.zeros((3, 3))
    rotm[0][0] = w*w + x*x - y*y - z*z
    rotm[0][1] = 2*(x*y - w*z)
    rotm[0][2] = 2*(w*y + x*z)

    rotm[1][0] = 2*(w*z + x*y)
    rotm[1][1] = w*w - x*x + y*y - z*z
    rotm[1][2] = 2*(y*z - w*x)

    rotm[2][0] = 2*(x*z - w*y)
    rotm[2][1] = 2*(w*x + y*z)
    rotm[2][2] = w*w - x*x - y*y + z*z
    return rotm


def rotm_to_quat(rotm):
    # This method computes the quaternion from the Direction Cosine Matrix (DCM)
    if rotm.shape[0] != 3 or rotm.shape[1] != 3:
        raise ValueError('Matrix input size not correct, should be 3x3')
    rotm_trans = np.transpose(rotm)
    m00 = rotm_trans[0][0]
    m01 = rotm_trans[0][1]
    m02 = rotm_trans[0][2]
    m10 = rotm_trans[1][0]
    m11 = rotm_trans[1][1]
    m12 = rotm_trans[1][2]
    m20 = rotm_trans[2][0]
    m21 = rotm_trans[2][1]
    m22 = rotm_trans[2][2]

    if m22 < 0:
        if m00 > m11:
            t = 1 + m00 - m11 - m22
            q = [t, m01 + m10, m20 + m02, m12 - m21]
        else:
            t = 1 - m00 + m11 - m22
            q = [m01 + m10, t, m12 + m21, m20 - m02]
    else:
        if m00 < -m11:
            t = 1 - m00 - m11 + m22
            q = [m20 + m02, m12 + m21, t, m01 - m10]
        else:
            t = 1 + m00 + m11 + m22
            q = [m12 - m21, m20 - m02, m01 - m10, t]

    q = np.multiply(q, 0.5) / np.sqrt(t)
    quat = [q[3], q[0], q[1], q[2]]
    quat_from_rotm = quat_normalize(quat)
    return quat_from_rotm
