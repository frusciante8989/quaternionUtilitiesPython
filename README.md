# Quaternion Utilities

This repo contains methods to allow transformation from quaternion 
to other common 3D representation method.

In particular, the tools contained are:
    
* **wrap_to_pi**: this method to wrap an angle (in rad) to the range [-pi; pi]
* **quat_conj**: this method returns the conjugate (inverse) of a quaternion
* **quat_multiply**: this method returns the product of a quaternion multiplication
* **vect_normalize**: this method normalizes a 1x3 vector
* **quat_normalize**: this method normalizes a quaternion
* **quat_norm**: this method returns the norm of a quaternion
* **quat_to_rotm**: this method returns the Direction Cosine Matrix (DCM) by applying the Euler-Rodrigues parameterization, given an input quaternion
* **rotm_to_quat**: this method returns the quaternion by applying the Euler-Rodrigues parameterization, given an input Direction Cosine Matrix (DCM)
* **axisangle_to_quat**: this method returns the quaternion from the input axis and angle representation
* **quat_to_axisangle**: this method returns the axis and angle representation from the input quaternion
* **rpy_to_quat**: this method returns the quaternion from roll, pitch, yaw angle (ZYX Euler sequence)
* **quat_to_rpy**: this method returns the roll, pitch, yaw angle (ZYX Euler sequence) from the quaternion

