from configs.base_config import teleop_config
from utils.camera_utils import RealSenseCamera

cam = RealSenseCamera()

# from telemoma.utils.camera_utils import Camera
# cam = Camera(img_topic="/camera/color/image_raw", depth_topic="/camera/aligned_depth_to_color/image_raw")

teleop_config.arm_left_controller = 'vision'
teleop_config.arm_right_controller = 'vision'
teleop_config.base_controller = 'vision'
# teleop_config.torso_controller = 'vision'
teleop_config.use_oculus = False
teleop_config.interface_kwargs.vision = dict(camera=cam, set_ref=True)