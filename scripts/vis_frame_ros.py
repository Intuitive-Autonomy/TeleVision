from tf.transformations import quaternion_from_matrix
import rospy
import tf2_ros
import geometry_msgs.msg

from scipy.spatial.transform import Rotation as R
import numpy as np
import pickle

def publish_transform(transform, name, base_name):
    translation = transform[:3]

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = base_name
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = transform[3:]
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    br.sendTransform(t)

def redefine_transformation(T_hand2head):
    R_head2base = R.from_euler("X", np.array([np.pi]))
    T_head2base = np.identity(4)
    T_head2base[:3, :3] = R_head2base.as_matrix()

    R_gripper2hand = R.from_euler("XZ", np.array([np.pi/2, np.pi]))
    T_gripper2hand = np.identity(4)
    T_gripper2hand[:3, :3] = R_gripper2hand.as_matrix()

    T_gripper2base = T_head2base @ T_hand2head @ T_gripper2hand
    return T_gripper2base

def main():
    with open('pose.pkl', 'rb') as file:
        pose = pickle.load(file)
    pose_iter = 0

    rospy.init_node('frame_publisher')
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        head_rmat, left_pose, right_pose = pose[pose_iter]
        pose_iter += 1
        if pose_iter == len(pose):
            break
        publish_transform(left_pose, 'left', 'base')
        publish_transform(right_pose, 'right', 'base')
        head_quat = R.from_matrix(head_rmat)
        publish_transform(np.concatenate((np.zeros(3), head_quat.as_quat())), 'head', 'base')

        rate.sleep()

if __name__ == '__main__':
    main()
