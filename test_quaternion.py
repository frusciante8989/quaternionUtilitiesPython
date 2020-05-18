from tools.quat_tools import *


# Quaternion conjugate
q = [0.7071, 0.7071, 0.0, 0.0]
q_conj = quat_conj(q)
print('Quaternion conjugate is', q_conj)


# Quaternion norm/normalize
q = [7071, 7071, 0.0, 0.0]
q_norm = quat_norm(q)
print('Original norm is ', q_norm)

q_norm = quat_norm(quat_normalize(q))
print('Normalized norm is ', q_norm)


# Quaternion multiply
q0 = [0, 1, 0, 0]
q1 = [0, 0, 1, 0]
q_mult = quat_multiply(q0, q1)
print('Quaternion multiplication result is ', q_mult)


# Quaternion <-> rotation matrix
rotm = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
q = rotm_to_quat(rotm)
print('Quaternion is', q)

rotm = quat_to_rotm(q)
print('Rotation matrix is', rotm)


# Quaternion <-> axis angle
angle = 0.12345
axis = [1, 2, 3]
axis_norm = vect_normalize(axis)
print('Axis normalized is', axis_norm)

q = axisangle_to_quat(axis_norm, angle)
print('Quaternion is', q)

axis, angle = quat_to_axisangle(q)
print('Axis is', axis, 'Angle is', angle)


# Quaternion <-> roll, pitch, yaw (Euler 'ZXY')
yaw = 0.1
pitch = 0.2
roll = 0.3
q = rpy_to_quat(roll, pitch, yaw)
print('Quaternion is', q)

roll, pitch, yaw = quat_to_rpy(q)
print('Roll =', roll, 'pitch=', pitch, 'yaw=', yaw)
