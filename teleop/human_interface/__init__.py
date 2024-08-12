from human_interface.oculus import OculusPolicy
from human_interface.vision import VisionTeleopPolicy
from human_interface.keyboard import KeyboardInterface
from human_interface.spacemouse import SpaceMouseInterface
from human_interface.mobile_phone import MobilePhonePolicy

INTERFACE_MAP = {
    'oculus': OculusPolicy,
    'vision': VisionTeleopPolicy,
    'keyboard': KeyboardInterface,
    'spacemouse': SpaceMouseInterface,
    'mobile_phone': MobilePhonePolicy
}
