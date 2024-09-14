from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat_(xyz):
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements


def euler_2_quat(xyz):
    return R.from_euler("xyz", xyz).as_quat()


def normalize_quaternion(q):
    return q / np.linalg.norm(q)


def quaternion_distance(q1, q2):
    # return rotation angle in degrees, between q1 and q2
    q1_normalized = normalize_quaternion(q1)
    q2_normalized = normalize_quaternion(q2)

    R1 = R.from_quat(q1_normalized).as_matrix()
    R2 = R.from_quat(q2_normalized).as_matrix()

    rot_product = np.dot(R1, R2.T)

    angle_rad = np.arccos((np.trace(rot_product) - 1) / 2)

    return np.degrees(angle_rad)


def quaternion_product(q1, q2):
    # return rotation angle in degrees, between q1 and q2
    q1_normalized = normalize_quaternion(q1)
    q2_normalized = normalize_quaternion(q2)
    return np.sum(q1_normalized * q2_normalized)



def euler_translation_to_homogeneous_matrix(pose_6d):
    rotation_matrix = R.from_euler("xyz", pose_6d[3:]).as_matrix()

    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = pose_6d[:3]

    return homogeneous_matrix

def homogeneous_matrix_to_euler_translation(homogeneous_matrix):
    rotation_matrix = homogeneous_matrix[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz")

    translation_vector = homogeneous_matrix[:3, 3]

    return np.concatenate([translation_vector, euler_angles])