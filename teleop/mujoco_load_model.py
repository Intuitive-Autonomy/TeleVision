import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib as mpl
import matplotlib.pyplot as plt

from lm_ik import LevenbegMarquardtIK


xml_path = '../assets/h1_2/h1_2.xml'


# For visualize callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        init_controller(model, data)
        mj.mj_forward(model, data)
    elif act == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.terminate()
        exit(0)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
data_sim = mj.MjData(model)            # data structure for forward/inverse kinematics
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window_size = (1200, 900)
window = glfw.create_window(window_size[0], window_size[1], "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Set camera configuration
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 3.0
cam.lookat = np.array([0.0, 0.0, 0.5])


def controller(model, data):
    # Apply control
    pass

def init_controller(model, data):
    # give the end effector a intial value
    # data.qpos[14:21] = [-0.64302, 0.16464, -0.17368, 0.15443, 1.07197, -0.00155, 1.12379]
    data.qpos[14:21] = [-0.76785, -0.24141,  0.73457,  0.44476, -0.67473,  0.10158,  1.17724]
    data.qpos[33:40]  = [-0.70458, -0.71602,  0.48448,  1.00907, -0.93045,  0.47,     0.86384]
    # data.qpos[33:40] = [-1.40312, -0.09321, 2.11381, 1.82825, -1.72764, 0.47, 1.26999]
    mj.mj_step(model, data)


simend = 20000
init_controller(model, data)

while not glfw.window_should_close(window):
    simstart = data.time
    mj.mj_step(model, data)

    if (data.time>=simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    time.sleep(max(0, 0.05 - (data.time - simstart)))

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
