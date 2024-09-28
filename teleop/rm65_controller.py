import threading
import numpy as np

from robotic_arm_package.robotic_arm import *

kNumJoints = 12
kLeftArmIP = '192.168.1.18'  # to be updated
kRightArmIP = '192.168.1.18'  # to be updated

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class RM65Controller():
    def __init__(self):
        try:
            self.robot_arm_left = Arm(RM65, kLeftArmIP)
            self.robot_arm_right = Arm(RM65, kLeftArmIP)
        except Exception as e:
            print("Arm Initialization Error: ", e)
            exit(0)
        self.joint_state_buffer = DataBuffer()
        self.joint_command_buffer = DataBuffer()
        
        # get and set initial state
        l_ret, r_ret = False, False
        while not (l_ret and r_ret):
            l_ret, l_joint, l_pose, l_arm_err, l_sys_err = self.robot_arm_left.Get_Current_Arm_State()
            r_ret, r_joint, r_pose, r_arm_err, r_sys_err = self.robot_arm_right.Get_Current_Arm_State()
            if l_ret and r_ret:
                joint = np.zeros(kNumJoints)
                joint[:int(kNumJoints/2)] = l_joint
                joint[int(kNumJoints/2):] = r_joint
                self.joint_state_buffer.SetData(joint)
                self.joint_command_buffer.SetData(joint)
            else:
                print("fail to get arm states")
        self.report_joint_state_thread = threading.Thread(target=self.SubscribeState)
        self.report_joint_state_thread.start()
        self.control_thread = threading.Thread(target=self.Control)
        self.control_thread.start()
    
    def get_joint_state(self):
        joint_state = self.joint_state_buffer.GetData()
        if joint_state:
            return joint_state
        else:
            return None
    
    def set_joint_state(self, joint):
        if len(joint) != kNumJoints:
            print("invalid joint dimension")
            return
        if joint is not None:
            self.joint_command_buffer.SetData(joint)

    def SubscribeState(self):
        while True:
            joint_state_data = np.zeros(kNumJoints)
            l_ret, l_joint, l_pose, l_arm_err, l_sys_err = self.robot_arm_left.Get_Current_Arm_State()
            r_ret, r_joint, r_pose, r_arm_err, r_sys_err = self.robot_arm_right.Get_Current_Arm_State()
            if l_ret and r_ret:
                joint = np.zeros(kNumJoints)
                joint[:int(kNumJoints/2)] = l_joint
                joint[int(kNumJoints/2):] = r_joint
                self.joint_state_buffer.SetData()
            else:
                print("fail to get arm states")
            time.sleep(0.01)

    def Control(self):
        while True:
            joint = self.joint_command_buffer.GetData()
            if joint is not None:
                l_ret = self.robot_arm_left.Movej_CANFD(joint[:int(kNumJoints/2)], True, 0)
                r_ret = self.robot_arm_right.Movej_CANFD(joint[int(kNumJoints/2):], True, 0)
                if not (l_ret and r_ret):
                    print("fail to set arm states")
            time.sleep(0.01)

if __name__ == '__main__':
    arm_controller = RM65Controller()
    joint_state = arm_controller.get_joint_state()
    joint_state[0] += 0.1
    arm_controller.set_joint_state(joint_state)
