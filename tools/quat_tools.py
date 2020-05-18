import numpy as np


def wrap_to_pi(angle):
    # Method to wrap an angle to [-pi; pi]
    angle_pi = (angle - 2 * np.pi * np.floor((angle + np.pi)/(2*np.pi)))
    return angle_pi


def quat_conj(quat):
    # Method to find the conjugate (inverse) of a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    qw, qx, qy, qz = quat
    quat_conjugate = (qw, -qx, -qy, -qz)
    return quat_conjugate


def vect_normalize(vect, tolerance=0.00001):
    # Method to normalize a vector
    if len(vect) != 3:
        raise ValueError('vector input length not correct, should be 1x3')
    mag_square = sum(v * v for v in vect)
    if abs(mag_square - 1.0) > tolerance:
        mag = np.sqrt(mag_square)
        vect_normalized = tuple(v / mag for v in vect)
        return vect_normalized
    return vect


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


def quat_multiply(quat_b, quat_a):
    # Method to find the product of a quaternion multiplication
    if len(quat_b) != 4 or len(quat_a) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    qw0, qx0, qy0, qz0 = quat_a
    qw1, qx1, qy1, qz1 = quat_b
    quat_mult = np.array([-qx1 * qx0 - qy1 * qy0 - qz1 * qz0 + qw1 * qw0,
                          qx1 * qw0 + qy1 * qz0 - qz1 * qy0 + qw1 * qx0,
                          -qx1 * qz0 + qy1 * qw0 + qz1 * qx0 + qw1 * qy0,
                          qx1 * qy0 - qy1 * qx0 + qz1 * qw0 + qw1 * qz0], dtype=np.float64)
    return quat_mult


def quat_to_rotm(quat):
    # This method computes the Direction Cosine Matrix (DCM) by applying the Euler-Rodrigues parameterization from a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')

    quat_normalized = quat_normalize(quat)
    qw, qx, qy, qz = quat_normalized
    rotm = np.zeros((3, 3))
    rotm[0][0] = qw*qw + qx*qx - qy*qy - qz*qz
    rotm[0][1] = 2*(qx*qy - qw*qz)
    rotm[0][2] = 2*(qw*qy + qx*qz)

    rotm[1][0] = 2*(qw*qz + qx*qy)
    rotm[1][1] = qw*qw - qx*qx + qy*qy - qz*qz
    rotm[1][2] = 2*(qy*qz - qw*qx)

    rotm[2][0] = 2*(qx*qz - qw*qy)
    rotm[2][1] = 2*(qw*qx + qy*qz)
    rotm[2][2] = qw*qw - qx*qx - qy*qy + qz*qz
    return rotm


def rotm_to_quat(rotm):
    # This method computes the quaternion from the Direction Cosine Matrix (DCM) by applying the Euler-Rodrigues parameterization
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


def axisangle_to_quat(vect, angle):
    # This method computes the quaternion from the axis and angle representation
    if len(vect) != 3 or not np.isscalar(angle):
        raise ValueError('Vector or angle input size not correct')
    vect = vect_normalize(vect)
    x, y, z = vect
    angle /= 2
    qw = np.cos(angle)
    qx = x * np.sin(angle)
    qy = y * np.sin(angle)
    qz = z * np.sin(angle)
    return qw, qx, qy, qz


def quat_to_axisangle(quat):
    # This method computes the axis and angle representation from a quaternion
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    quat_normalized = quat_normalize(quat)
    w, vect = quat_normalized[0], quat_normalized[1:]
    theta = np.arccos(w) * 2.0
    return vect_normalize(vect), theta


def rpy_to_quat(roll=0, pitch=0, yaw=0):
    # This method computes the quaternion from roll, pitch, yaw angle (ZYX Euler sequence)
    # Corresponding Matlab function is eul2quat(([yaw, pitch, roll]), 'ZYX')
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    return [qw, qx, qy, qz]


def quat_to_rpy(quat):
    # This method computes the roll, pitch, yaw angle (ZYX Euler sequence) from the quaternion
    # Corresponding Matlab function is quat2eul('ZYX')
    if len(quat) != 4:
        raise ValueError('Quaternion input length not correct, should be 1x4')
    quat_normalized = quat_normalize(quat)
    qw, qx, qy, qz = quat_normalized
    t0 = 2 * (qw * qx + qy * qz)
    t1 = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(t0, t1)
    t2 = 2 * (qw * qy - qz * qx)
    t2 = 1 if t2 > +1 else t2
    t2 = -1 if t2 < -1 else t2
    pitch = np.arcsin(t2)
    t3 = 2 * (qw * qz + qx * qy)
    t4 = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw
