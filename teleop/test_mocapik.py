import os
import numpy as np
import mujoco as mj
import time

scene_xml_path = '../assets/robot_and_scene_posctrl.xml'
scene_xml_path = os.path.join(os.path.dirname(__file__), scene_xml_path)
robot_xml_path = '../assets/robot_assemble_mocap.xml'
robot_xml_path = os.path.join(os.path.dirname(__file__), robot_xml_path)


model = mj.MjModel.from_xml_path(scene_xml_path)
data = mj.MjData(model)
data.qpos[15] = 0.5
mj.mj_step(model, data)
init_ee_pos = data.body("R_Link6").xpos
init_ee_quat = data.body("R_Link6").xquat
print("initial R_Link6 pos: ", data.body("R_Link6").xpos)
print("initial L_Link6 pos: ", data.body("L_Link6").xpos)



target_ee_pos_L = np.array([.26050162, 0.6385, 1.25])
target_ee_pos_R = np.array([-0.26050162, 0.56033647, 0.89388679])
model_ik = mj.MjModel.from_xml_path(robot_xml_path)
data_ik = mj.MjData(model_ik)
mj.mj_forward(model_ik, data_ik)
print("data ik init qpos: ", data_ik.qpos[:8])
data_ik.mocap_pos[0] = target_ee_pos_L.copy()
data_ik.mocap_pos[1] = target_ee_pos_R.copy()
time_0 = time.time()
for _ in range(100):
    mj.mj_step(model_ik, data_ik)
time_1 = time.time()
print("time used: ", time_1 - time_0)
print("final R_Link6 pos: ", data_ik.body("R_Link6").xpos)
print("final L_Link6 pos: ", data_ik.body("L_Link6").xpos)
print("data ik final qpos: ", data_ik.qpos[:8])

data.qpos[7:] = data_ik.qpos.copy()
mj.mj_step(model, data)

print("final R_Link6 pos: ", data.body("R_Link6").xpos)
print("final L_Link6 pos: ", data.body("L_Link6").xpos)
