import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig
from scipy.spatial.transform import Rotation as R


def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = R.from_quat([quat[1], quat[2],
                            quat[3], quat[0]]).as_euler('xyz', degrees=degree).tolist()

    return eulerZYX

def pose_xyzw2wxyz(pose):
    return [pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]]

def pose_wxyz2xyzw(pose):
    return [pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[3]]
    

class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        # "wrist_1": "130322273075",
        "wrist_1": "230322272566",
        # "wrist_2": "230322272566",
    }
    # TARGET_POSE = np.array(
    #     [
    #         0.5906439143742067,
    #         0.07771711953459341,
    #         0.0937835826958042,
    #         3.1099675,
    #         0.0146619,
    #         -0.0078615,
    #     ]
    # )
    
    # TARGET_POSE = [0.7743712067604065,-0.03834529593586922,0.2679073214530945,-0.0024682849179953337,0.06168980151414871,0.9980902671813965,-0.0020353312138468027]
    # TARGET_POSE = [0.6008051633834839,-0.052534788846969604,0.22573457658290863,0.024103181436657906,0.9996849298477173,0.004863678943365812,0.005044732708483934]
    #TARGET_POSE = [0.6897345781326294,-0.0855565145611763,0.23157787322998047,-0.10760285705327988,0.994091808795929,-0.0054414356127381325,0.0131[71523809432983]
    TARGET_POSE = [0.6483319401741028,-0.0007605736027471721,0.22615499794483185,-0.009632401168346405,0.018566621467471123,0.9997801780700684,0.001442493638023734]
    
    # TARGET_POSE = [0.68,-0.11,0.29,-0.00,0.00,1.00,-0.00] # [x,y,z,w,x,y,z] 
    # ['0.68', '-0.11', '0.29', '0.00', '-0.00', '1.00', '-0.00']
    TARGET_POSE = np.concatenate([TARGET_POSE[:3], quat2eulerZYX(TARGET_POSE[3:])]) # Quaternion input in [w,x,y,z] order
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0])
    #RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.00, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.003, 0.003, 0.01, 0.2, 0.2, 0.2])
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    # RANDOM_RESET = False
    RANDOM_RESET = True
    # RANDOM_XY_RANGE = 0.05 # orin
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = np.pi / 6 # orin
    # RANDOM_RZ_RANGE = np.pi / 12
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
