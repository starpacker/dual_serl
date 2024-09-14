"""Gym Interface for Franka"""
import numpy as np
import gym
import cv2
import copy
from scipy.spatial.transform import Rotation, Slerp
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from ipdb import set_trace
from tqdm import trange

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler, quaternion_distance, euler_translation_to_homogeneous_matrix, homogeneous_matrix_to_euler_translation, quaternion_product


class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("RealSense Cameras", frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = (False,)
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    BINARY_GRIPPER_THREASHOLD: float = 0.5
    APPLY_GRIPPER_PENALTY: bool = True
    GRIPPER_PENALTY: float = 0.1


##############################################################################


class FrankaEnv(gym.Env):
    def __init__(
        self,
        hz=15,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        max_episode_length=100,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = max_episode_length

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )

        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6, 7))

        self.curr_gripper_pos = 0
        self.gripper_binary_state = 0  # 0 for open, 1 for closed
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "wrist_1": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                        # "wrist_2": gym.spaces.Box(
                        #     0, 255, shape=(128, 128, 3), dtype=np.uint8
                        # ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        self.img_queue = queue.Queue()
        self.displayer = ImageDisplayer(self.img_queue)
        self.displayer.start()
        print("Initialized Franka")
        
        # homing the robot
        self.go_to_rest(joint_reset=True)
        requests.post(self.url + "clearforce")
        time.sleep(0.1)
        
        
        # reward cache
        self.prev_trans_dist_xy = None
        self.prev_trans_dist_z = None
        self.prev_rot_dist = None


    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        print("xyz_delta: ", xyz_delta)

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        gripper_action_effective = self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward, is_success = self.compute_reward(ob, gripper_action_effective)
        done = self.curr_path_length >= self.max_episode_length or is_success
        return ob, reward, done, False, {}

    # def compute_reward(self, obs, gripper_action_effective) -> bool:
    #     """We are using a sparse reward function."""
    #     current_pose = obs["state"]["tcp_pose"]
    #     # convert from quat to euler first
    #     euler_angles = quat_2_euler(current_pose[3:])
    #     euler_angles = np.abs(euler_angles)
    #     current_pose = np.hstack([current_pose[:3], euler_angles])
    #     delta = np.abs(current_pose - self._TARGET_POSE)
    #     if np.all(delta < self._REWARD_THRESHOLD):
    #         reward = 1
    #     else:
    #         # print(f'Goal not reached, the difference is {delta}, the desired threshold is {_REWARD_THRESHOLD}')
    #         reward = 0

    #     if self.config.APPLY_GRIPPER_PENALTY and gripper_action_effective:
    #         reward -= self.config.GRIPPER_PENALTY

    #     return reward
    

    def compute_reward(self, obs, gripper_action_effective=True):
        ''' compute the reward based on the current pose 
            whether need to calculate the ry,rz reward respectively?
        '''
        current_pose = obs["state"]["tcp_pose"]
        # wxyz to xyzw
        current_quat_wxyz = current_pose[3:]
        current_quat_xyzw = np.array([current_quat_wxyz[1], current_quat_wxyz[2], current_quat_wxyz[3], current_quat_wxyz[0]])
        current_pose = np.concatenate([current_pose[:3], current_quat_xyzw])
        
        _current_pose_6d = np.concatenate([current_pose[:3], quat_2_euler(current_pose[3:])])
        _target_pose_matrix = euler_translation_to_homogeneous_matrix(self._TARGET_POSE)
        _current_pose_matrix = euler_translation_to_homogeneous_matrix(_current_pose_6d)
        target_T_current = np.linalg.inv(_target_pose_matrix) @ _current_pose_matrix   
        target_T_current_6d = homogeneous_matrix_to_euler_translation(target_T_current)
        target_T_target = np.eye(4)
        target_T_target_6d = homogeneous_matrix_to_euler_translation(target_T_target)
        # compute delta
        euler_angles = quat_2_euler(current_pose[3:])
        euler_angles = np.abs(euler_angles)
        current_pose = np.hstack([current_pose[:3], euler_angles])

        # if self.reward_noise:
        #     # to simulate the pose-estimation noise
        #     for i in range(6):
        #         current_pose[i] += np.random.uniform(-self.reward_disturbance_6d[i], self.reward_disturbance_6d[i])
        
        _target_pose = np.hstack([self._TARGET_POSE[:3], np.abs(self._TARGET_POSE[3:])])

        current_pose = target_T_current_6d
        _target_pose = target_T_target_6d

        delta = np.abs(current_pose - _target_pose)
        print("delta: ", delta)

        # compute success identifier
        is_success = np.all(delta < self._REWARD_THRESHOLD) # successfully inserted
        # is_reward_z = np.all(delta[:2] < 0.01) or (current_pose - self._TARGET_POSE)[2] > 0.015 # xy position is within 1cm, or cable is too high, then it's okay to reward z
        is_reward_z = np.all(delta[:2] < 0.008) or (current_pose - _target_pose)[2] > 0.015 # xy position is within 1cm, or cable is too high, then it's okay to reward z

        # compute translation distance
        curr_trans_dist_xy = np.linalg.norm(delta[:2])
        curr_trans_dist_z = np.linalg.norm(delta[2:3])

        # compute rotation distance
        # curr_quat = euler_2_quat(current_pose[3:])
        # target_quat = euler_2_quat(self._TARGET_POSE[3:])
        curr_quat = euler_2_quat(current_pose[3:])
        target_quat = euler_2_quat(_target_pose[3:])
        curr_rot_dist = quaternion_distance(curr_quat, target_quat) # degrees
        # curr_rot_dist = quaternion_distance(curr_quat, target_quat) # dot_product

        if self.prev_trans_dist_xy is None and self.prev_trans_dist_z is None and self.prev_rot_dist is None:
            reward = 0
        else:
            trans_scale = 2000
            rot_scale = 1
            xy_ratio = 0.7

            trans_reward_xy = (self.prev_trans_dist_xy - curr_trans_dist_xy) * xy_ratio
            trans_reward_z = (self.prev_trans_dist_z - curr_trans_dist_z) * (1 - xy_ratio) * is_reward_z
            trans_reward = trans_reward_xy + trans_reward_z 
            rot_reward = self.prev_rot_dist - curr_rot_dist
            reward = trans_reward * trans_scale + rot_reward * rot_scale

            # to distinguish success from failures
            if is_success:
                reward += 100

            # print success
            print("is_reward_z: ", is_reward_z)
            print("is_success: ", is_success)
            
            # print dists
            print("rot_dist: ", curr_rot_dist)
            print("trans_dist_xy: ", curr_trans_dist_xy)
            print("trans_dist_z: ", curr_trans_dist_z)

            # print rewards
            print("trans_reward_xy: ", trans_reward_xy * trans_scale)
            print("trans_reward_z: ", trans_reward_z * trans_scale)
            print("trans_reward: ", trans_reward * trans_scale)
            print("rot_reward: ", rot_reward * rot_scale)
            print("total_reward: ", reward)
        
        self.prev_trans_dist_xy = np.array(curr_trans_dist_xy)
        self.prev_trans_dist_z = np.array(curr_trans_dist_z)
        self.prev_rot_dist = np.array(curr_rot_dist)
    
        return reward, is_success


    def crop_image(self, name, image) -> np.ndarray:
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[:, 80:560, :]
        elif name == "wrist_2":
            return image[:, 80:560, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )
        self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        steps = int(timeout * self.hz)
        
        # steps = 2
        # set_trace()
        self._update_currpos()
        ### error, quat cannot use linear interpolation

        # path = np.linspace(self.currpos, goal, steps)
        # for p in path:
        #     self._send_pos_command(p)
        #     time.sleep(1 / self.hz)
        
        slerp = Slerp(
            times=[0,1], rotations=Rotation.from_quat([self.currpos[3:], goal[3:]])
        )
        path = np.linspace(0, 1, steps)
        interp_rots = slerp(path).as_quat()
        interp_trans = np.linspace(self.currpos[:3], goal[:3], steps)
        
        for trans, rot in zip(interp_trans, interp_rots):
            self._send_pos_command(np.concatenate([trans, rot]))
        
        self._update_currpos()

    def go_to_rest(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)
        
        # set_trace()
        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)
            # self._send_pos_command(reset_pose)
            time.sleep(1.5)
        else:
            # set_trace()
            reset_pose = self.resetpos.copy()
            # self._send_pos_command(reset_pose)
            # time.sleep(1.5)
            # set_trace()
            self.interpolate_move(reset_pose, timeout=1.5)
            time.sleep(1.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True
        
        self.go_to_rest(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0
        
        #set_trace()

        self._update_currpos()
        obs = self._get_obs()
        
        
        # reward cache
        self.prev_trans_dist_xy = None
        self.prev_trans_dist_z = None
        self.prev_rot_dist = None

        return obs, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, cam_serial in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, serial_number=cam_serial, depth=False)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (
                pos <= -self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 0
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 1
                return True
            elif (
                pos >= self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 1
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 0
                return True
            else:  # do nothing to the gripper
                return False
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos[:] = np.array(ps["pose"])
        self.currvel[:] = np.array(ps["vel"])

        self.currforce[:] = np.array(ps["force"])
        self.currtorque[:] = np.array(ps["torque"])
        self.currjacobian[:] = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
