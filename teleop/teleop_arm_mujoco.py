import math
import numpy as np
import torch

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices, last_3_palm_indices, last_3_tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
import cv2
from human_interface.teleop_core import TeleopAction

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from lm_ik import LevenbegMarquardtIK

class VuerTeleop:
    # record_playback_realtime: 0: no record, realtime vr, 1: record, 2: playback
    def __init__(self, config_file_path, record_data_path="", record_playback_realtime=0, resolution=(720, 1280)):
        self.resolution = resolution
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.record_playback_realtime = record_playback_realtime
        #self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True, record_data_path=record_data_path, record_playback_realtime=record_playback_realtime)

        self.processor = VuerPreprocessor()
        self.step_index = 0

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        if self.record_playback_realtime == 2:
            self.tv.step_record_data()
        
        # get body poses from teleop control device
        # left_hand_mat and right_hand_mat are finger positions in the left_wrist and right_wrist coorindate
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)
        #print("left_hand_mat: ", left_hand_mat)

        # head rotation matrix
        head_rmat = head_mat[:3, :3]
        
        # [-0.6, 0, 1.6] are for gym demo settings, may be related to the table position, x forward, z up
        pos_offset = np.array([-0.6, 0, 1.6])
        left_pose = np.concatenate([left_wrist_mat[:3, 3] + pos_offset,
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + pos_offset,
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        # finger pose and ik
        left_finger_pos = left_hand_mat[tip_indices]
        right_finger_pos = right_hand_mat[tip_indices]
        left_qpos = self.left_retargeting.retarget(left_finger_pos)[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_finger_pos)[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        left_gripper_dist = np.linalg.norm(left_finger_pos[0] - left_finger_pos[1])
        right_gripper_dist = np.linalg.norm(right_finger_pos[0] - right_finger_pos[1])

        # calulate the last three finger is pressed or not
        left_last_3_tip_finger_pos = left_hand_mat[last_3_tip_indices]
        left_last_3_paml_finger_pose = left_hand_mat[last_3_palm_indices]
        right_last_3_tip_finger_pos = right_hand_mat[last_3_tip_indices]
        right_last_3_palm_finger_pos = right_hand_mat[last_3_palm_indices]
        left_press_dist = 0
        right_press_dist = 0
        for i in range(3):
            left_press_dist += np.linalg.norm(left_last_3_tip_finger_pos[i] - left_last_3_paml_finger_pose[i])
            right_press_dist += np.linalg.norm(right_last_3_tip_finger_pos[i] - right_last_3_palm_finger_pos[i])
        left_pressed = left_press_dist < 0.2
        right_pressed = right_press_dist < 0.2
        if self.step_index % 100 == 0:
            print("step index: ", self.step_index)
        if self.record_playback_realtime == 1 and self.step_index == 5000:
            self.tv.save_record_data()
            print("Recording data")

        self.step_index += 1
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos, left_gripper_dist, right_gripper_dist, left_pressed, right_pressed

class Sim:
    def __init__(self, print_freq=False, record_playback_realtime=0):
        # get the full path
        xml_path = '../assets/robot_assemble.xml'

        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname, xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.record_playback_realtime = record_playback_realtime
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.data_sim = mj.MjData(self.model)            # data structure for forward/inverse kinematics


        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        window_size = (1200, 900)
        self.window = glfw.create_window(window_size[0], window_size[1], "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)


        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # Set camera configuration 
        # cam:  88.35000000000005 -57.00000000000002 0.3593839214309096 [3.40591518e-04 1.10704845e-04 1.74931750e+00]
        # print("cam: ", self.cam.azimuth, self.cam.elevation, self.cam.distance, self.cam.lookat)  
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options
        self.cam.azimuth = 89.6999999999999
        self.cam.elevation = -43.59999999999983
        self.cam.distance = 0.5222320046244449
        self.cam.lookat = np.array([-7.36667992e-03,  9.29655102e-05,  1.64719830e+00])
        
        # initialize visualization data structures
        #mj.mjv_defaultCamera(self.cam)
        #mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)


        self.ik_solver = LevenbegMarquardtIK(self.model, self.data_sim)

        # control components
        self.arms = ['left', 'right']
        self.ee_body_names = \
            {"left": "L_Link6", "right": "R_Link6"}
        self.gripper_joint_names = \
            {"left": ["L_finger1_joint", "L_finger2_joint"],
            "right": ["R_finger1_joint", "R_finger2_joint"]}
        self.press_flag = {'left': True, 'right': True}
        self.gripper_limit = {}
        for arm in self.arms:
            joint_name = self.gripper_joint_names[arm][0] # assume all gripper joints have the same limit
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            joint_limits = self.model.jnt_range[joint_id]
            self.gripper_limit[arm] = joint_limits[1] - joint_limits[0]

        import pickle
        with open('action.pkl', 'rb') as file:
            self.action_list = pickle.load(file)
            self.action_iter = 0
        self.init_controller(self.model, self.data)    
        self.print_freq = print_freq
        self.prev_action = {}
 

    def init_controller(self, model, data):
        mj.mj_step(model, data)

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            self.init_controller(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        elif act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.terminate()
            exit(0)

    def mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        self.lastx
        self.lasty
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window sizarmse
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/width, dy/height, self.scene, self.cam)
        print("cam: ", self.cam.azimuth, self.cam.elevation, self.cam.distance, self.cam.lookat) 

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                        yoffset, self.scene, self.cam)
        print("scroll cam: ", self.cam.azimuth, self.cam.elevation, self.cam.distance, self.cam.lookat) 

        
    def process_action(self, prev_action, action, press_flag):
        for arm in ['left', 'right']:
            if prev_action.get(arm, None) is None:
                prev_action[arm] = action[arm]
            
            # if not pressed, just set current action as previous action, so no change for action
            # otherwise, just calulate the change of action
            if press_flag[arm] == False:
                prev_action[arm] = action[arm]

        return prev_action, action

        
    def set_gripper_joint(self, model, data, action, joint_names):
        # use leftTrig and rightTrig button value for gripper
        for arm in ['left', 'right']:
            for joint_name in joint_names[arm]:
                joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                joint_limits = model.jnt_range[joint_id]
                press_val = action.get(arm + 'Trig', (0.0,))[0]
                joint_q = joint_limits[0] + \
                    (joint_limits[1] - joint_limits[0]) * press_val
                data.qpos[joint_id] = joint_q


    def trans_gripper_base(self, T_hand2head, arm='left', vr_height=1.0):
        # using robot coordinate defined in xml/urdf
        # head and hand are in oculus coordinate
        # base and gripper are in robot coordinate
        R_head2base = R.from_euler("X", np.array([np.pi]))
        T_head2base = np.identity(4)
        T_head2base[:3, :3] = R_head2base.as_matrix()
        T_head2base[2, -1] = vr_height # from vr headset to ground

        if arm == 'left':
            R_gripper2hand = R.from_euler("XZ", np.array([np.pi/2, np.pi]))
        else:
            R_gripper2hand = R.from_euler("X", np.array([np.pi/2]))
        
        T_gripper2hand = np.identity(4)
        T_gripper2hand[:3, :3] = R_gripper2hand.as_matrix()

        T_gripper2base = T_head2base @ T_hand2head @ T_gripper2hand
        
        return T_gripper2base
    
    def oculus_pose2mujoco(self, action, shape_factor=1.0):
        mujoco_action = {}
        for arm in ['left', 'right']:
            T_hand2head = np.identity(4)
            T_hand2head[:3, :3] = R.from_quat(action[arm][3:]).as_matrix()
            T_hand2head[:3, -1] = action[arm][:3]
            T_gripper2base = self.trans_gripper_base(T_hand2head)
            mujoco_action[arm] = \
                np.concatenate((T_gripper2base[:3, -1] * shape_factor, 
                                R.from_matrix(T_gripper2base[:3, :3]).as_quat()))
        return mujoco_action

    def oculus_delta_pose2mujoco(self, action, prev_action, cur_pose_dict, cur_quat_dict, press_flag):
        mujoco_action = {}
        for arm in ['left', 'right']:
            # T_gripper2base = self.trans_gripper_base(self.vec2mat(action[arm]))
            T_gripper2base = self.vec2mat(action[arm])
            cur_xpos = T_gripper2base[:3, -1]
            cur_rot_mat = T_gripper2base[:3, :3]

            # T_gripper2base = self.trans_gripper_base(self.vec2mat(prev_action[arm]))
            T_gripper2base = self.vec2mat(prev_action[arm])
            prev_xpos = T_gripper2base[:3, -1]
            prev_rot_mat = T_gripper2base[:3, :3]

            xpos_delta = cur_xpos - prev_xpos
            quat_delta = R.from_matrix(cur_rot_mat @ np.linalg.inv(prev_rot_mat))
            if press_flag[arm] == False:
                xpos_delta = np.array([0, 0, 0])
                quat_delta = R.from_matrix(np.identity(3))

            target_xpos = cur_pose_dict[arm] + xpos_delta
            target_quat = quat_delta * R.from_quat(np.roll(cur_quat_dict[arm], -1))
            
            mujoco_action[arm] = \
                np.concatenate((target_xpos, 
                                target_quat.as_quat()))
        return mujoco_action

    def vec2mat(self, vec):
        # vec: 1x7 x, y, z, x, y, z, w,
        T = np.identity(4)
        T[:3, :3] = R.from_quat(vec[3:]).as_matrix()
        T[:3, -1] = vec[:3]
        return T

    def prepare_action(self, action, left_pose, right_pose, left_gripper, right_gripper):
        T_xfront2yfront = np.identity(4)
        R_xfront2yfront = R.from_euler("Z", np.array([np.pi/2]))
        T_xfront2yfront[:3, :3] = R_xfront2yfront.as_matrix()
        T_left_pose = self.vec2mat(left_pose)
        T_right_pose = self.vec2mat(right_pose)
        T_left_pose = T_xfront2yfront @ T_left_pose
        T_right_pose = T_xfront2yfront @ T_right_pose
        action['left'] = \
            np.concatenate((T_left_pose[:3, -1], R.from_matrix(T_left_pose[:3, :3]).as_quat()))
        action['right'] = \
            np.concatenate((T_right_pose[:3, -1], R.from_matrix(T_right_pose[:3, :3]).as_quat()))

        gripper_dist = {'left': left_gripper, 'right': right_gripper}
        for arm in self.arms:
            action.extra['buttons'][arm + 'Trig'] = (1 - min(1, gripper_dist[arm] / (self.gripper_limit[arm] * 2)),)
        return action
    
    def check_valid_action(self, action, current_pose, current_quat):
        # TODO: add more safty checks
        for arm in self.arms:
            if np.abs(np.linalg.norm(action[arm][3:]) - 1) > 0.1:
                action[arm][3:] = np.array([0, 0, 0, 1])
            # initialize pose from default position
            if np.linalg.norm(current_pose[arm] - action[arm][:3]) > 0.25:
                # print("target pose too far")
                # print("target pose: ", action[arm])
                # print("current pose: ", current_pose[arm])
                action[arm][:3] = current_pose[arm]
                action[arm][-1] = current_quat[arm][0]
                action[arm][3:-1] = current_quat[arm][1:]
        return action


    def set_ee_pose(self, model, data, target_xpos_list, target_quat_list, body_names):
        # mujoco quaternion is scalar first
        target_quat_list_ = \
            [np.concatenate(([quat[-1]], quat[:-1])) for quat in target_quat_list]
        body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name) 
                    for i, body_name in enumerate(body_names)]
        sol = self.ik_solver.calculate(
            target_xpos_list, 
            target_quat_list_, 
            data.qpos, 
            body_ids)

        # Apply control
        data.qpos = sol


    def step(self, head_rmat, left_pose, right_pose, left_gripper, right_gripper, left_pressed, right_pressed):

        if self.print_freq:
            start = time.time()
        
        # get current statesgripper_joint_names
        current_pose_dict, current_quat_dict = {}, {}
        for i, (body_key, body_name) in enumerate(self.ee_body_names.items()):
            current_pose = self.data.body(body_name).xpos
            current_quat = self.data.body(body_name).xquat
            current_pose_dict[body_key] = current_pose
            current_quat_dict[body_key] = current_quat


        #if self.print_freq:
        #print("get vr pose: ", left_pose, right_pose)


        # Initialize action
        action = TeleopAction()
        # if action.extra['buttons'].get('A', False):
        #     print("A button pressed")
        #     raise Exception("A button pressed")
        
        # action = oculus_pose2mujoco(action)
        action = self.prepare_action(action, left_pose, right_pose, left_gripper, right_gripper)
        press_flag = {'left': left_pressed, 'right': right_pressed}
        self.prev_action, action = self.process_action(self.prev_action, action, press_flag)
        mujoco_action = self.oculus_delta_pose2mujoco(action, self.prev_action, current_pose_dict, current_quat_dict, press_flag)
        mujoco_action = self.check_valid_action(mujoco_action, current_pose_dict, current_quat_dict)
        self.prev_action = action

        # control
        self.set_ee_pose(
            self.model, self.data,
            [mujoco_action[arm][:3] for arm in self.arms], 
            [mujoco_action[arm][3:] for arm in self.arms], 
            [self.ee_body_names[arm] for arm in self.arms])
        self.set_gripper_joint(
            self.model, self.data, 
            action.extra['buttons'],
            self.gripper_joint_names 
            )

        mj.mj_step(self.model, self.data)

        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # time.sleep(max(0, 0.05 - (data.time - simstart)))
        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)
        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        rgb_image = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb_image, None, viewport, self.context)
        rgb_image = np.flipud(rgb_image)
        # cv2.namedWindow('Mujoco', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Mujoco', rgb_image)
        # cv2.waitKey(1)
        return rgb_image, rgb_image

    def get_resolution(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        return (viewport_height, viewport_width)
    
    def end(self):
        glfw.terminate()


if __name__ == '__main__':
    record_playback_realtime = 2
    simulator = Sim(record_playback_realtime=record_playback_realtime)
    teleoperator = VuerTeleop('inspire_hand.yml', record_data_path="hand_records.pkl", record_playback_realtime=record_playback_realtime, resolution=simulator.get_resolution())
    while True:
        try:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos, left_gripper_dist, right_gripper_dist, left_pressed, right_pressed = teleoperator.step()
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_gripper_dist, right_gripper_dist, left_pressed, right_pressed)
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
        except Exception as e:
            print("Error: ", e)
            simulator.end()
            exit(0)
