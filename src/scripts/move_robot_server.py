#!/usr/bin/env python3

import roslib; roslib.load_manifest('kinova_rl')
import rospy
import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import math
from kinova_rl.srv import MoveRobot, MoveRobotResponse
import time

""" Global variable """
arm_joint_number = 0
finger_number = 0
prefix = 'j2n6s300_'
finger_maxDist = 18.9 / 2 / 1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentCartesianCommand = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072]  # default home in unit mq


def cartesian_pose_client(position, orientation):
    """Send a cartesian goal to the action server."""
    action_address = '/j2n6s300_driver/pose_action/tool_pose'
    rospy.loginfo(f"Waiting for action server: {action_address}")
    client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
    client.wait_for_server(rospy.Duration(10.0))
    
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("Action server not available")
        return None

    goal = kinova_msgs.msg.ArmPoseGoal()
    goal.pose.header = std_msgs.msg.Header(frame_id=(prefix + 'link_base'))
    goal.pose.pose.position = geometry_msgs.msg.Point(
        x=position[0], y=position[1], z=position[2])
    goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
        x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

    rospy.loginfo(f"Sending goal: {goal.pose.pose}")
    client.send_goal(goal)

    if client.wait_for_result(rospy.Duration(10.0)):
        result = client.get_result()
        # getcurrentCartesianCommand()
        return result
    else:
        client.cancel_all_goals()
        rospy.logerr('      The cartesian action timed-out')
        return None

def joint_angle_client(angle_set):
    """Send a joint angle goal to the action server."""
    action_address = '/' + prefix + 'driver/joints_action/joint_angles'
    client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.ArmJointAnglesAction)
    client.wait_for_server()

    goal = kinova_msgs.msg.ArmJointAnglesGoal()

    goal.angles.joint1 = angle_set[0]
    goal.angles.joint2 = angle_set[1]
    goal.angles.joint3 = angle_set[2]
    goal.angles.joint4 = angle_set[3]
    goal.angles.joint5 = angle_set[4]
    goal.angles.joint6 = angle_set[5]
    goal.angles.joint7 = angle_set[6]

    client.send_goal(goal)
    if client.wait_for_result(rospy.Duration(20.0)):
        return client.get_result()
    else:
        print('        the joint angle action timed-out')
        client.cancel_all_goals()
        return None
    
def QuaternionNorm(Q_raw):
    qx_temp, qy_temp, qz_temp, qw_temp = Q_raw[0:4]
    qnorm = math.sqrt(qx_temp * qx_temp + qy_temp * qy_temp + qz_temp * qz_temp + qw_temp * qw_temp)
    qx_ = qx_temp / qnorm
    qy_ = qy_temp / qnorm
    qz_ = qz_temp / qnorm
    qw_ = qw_temp / qnorm
    Q_normed_ = [qx_, qy_, qz_, qw_]
    return Q_normed_


def Quaternion2EulerXYZ(Q_raw):
    Q_normed = QuaternionNorm(Q_raw)
    qx_ = Q_normed[0]
    qy_ = Q_normed[1]
    qz_ = Q_normed[2]
    qw_ = Q_normed[3]

    tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
    ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
    tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
    EulerXYZ_ = [tx_, ty_, tz_]
    return EulerXYZ_


def EulerXYZ2Quaternion(EulerXYZ_):
    tx_, ty_, tz_ = EulerXYZ_[0:3]
    sx = math.sin(0.5 * tx_)
    cx = math.cos(0.5 * tx_)
    sy = math.sin(0.5 * ty_)
    cy = math.cos(0.5 * ty_)
    sz = math.sin(0.5 * tz_)
    cz = math.cos(0.5 * tz_)

    qx_ = sx * cy * cz + cx * sy * sz
    qy_ = -sx * cy * sz + cx * sy * cz
    qz_ = sx * sy * cz + cx * cy * sz
    qw_ = -sx * sy * sz + cx * cy * cz

    Q_ = [qx_, qy_, qz_, qw_]
    return Q_


def getcurrentCartesianCommand():
    # wait to get current position
    topic_address = '/j2n6s300_driver/out/cartesian_command'
    rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, setcurrentCartesianCommand)
    rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
    print('Position listener obtained message for Cartesian pose.')


def setcurrentCartesianCommand(feedback):
    global currentCartesianCommand

    currentCartesianCommand_str_list = str(feedback).split("\n")

    for index in range(len(currentCartesianCommand_str_list)):
        temp_str = currentCartesianCommand_str_list[index].split(": ")
        currentCartesianCommand[index] = float(temp_str[1])


def unitParser(unit_, pose_value_, relative_):
    """ Argument unit """
    global currentCartesianCommand

    position_ = list(pose_value_[:3])  # Convert tuple to list
    orientation_ = list(pose_value_[3:])  # Convert tuple to list

    for i in range(3):
        if relative_:
            position_[i] = pose_value_[i] + currentCartesianCommand[i]
        else:
            position_[i] = pose_value_[i]

    # Modify current pose
    # currentCartesianCommand = position_

    if unit_ == 'mq':
        if relative_:
            orientation_XYZ = Quaternion2EulerXYZ(orientation_)
            orientation_xyz_list = [orientation_XYZ[i] + currentCartesianCommand[3 + i] for i in range(3)]
            orientation_q = EulerXYZ2Quaternion(orientation_xyz_list)
        else:
            orientation_q = orientation_

        orientation_rad = Quaternion2EulerXYZ(orientation_q)
        orientation_deg = list(map(math.degrees, orientation_rad))

    elif unit_ == 'mdeg':
        if relative_:
            orientation_deg_list = list(map(math.degrees, currentCartesianCommand[3:]))
            orientation_deg = [orientation_[i] + orientation_deg_list[i] for i in range(3)]
        else:
            orientation_deg = orientation_

        orientation_rad = list(map(math.radians, orientation_deg))
        orientation_q = EulerXYZ2Quaternion(orientation_rad)

    elif unit_ == 'mrad':
        if relative_:
            orientation_rad_list = currentCartesianCommand[3:]
            orientation_rad = [orientation_[i] + orientation_rad_list[i] for i in range(3)]
        else:
            orientation_rad = orientation_

        orientation_deg = list(map(math.degrees, orientation_rad))
        orientation_q = EulerXYZ2Quaternion(orientation_rad)

    else:
        raise Exception("Cartesian value have to be in unit: mq, mdeg or mrad")

    pose_mq_ = position_ + orientation_q
    pose_mdeg_ = position_ + orientation_deg
    pose_mrad_ = position_ + orientation_rad

    return pose_mq_, pose_mdeg_, pose_mrad_

def move_to_initial_position():
    initial_position = [281.3, 213.7, 32.05, 155.08, 106.384, 136.60, 0.0]
    # initial_position = [-0.05, -0.35, 0.28]
    # initial_orientation_deg = [90, 0, 7.68]  # Orientation in degrees (Euler angles)

    # # Convert Euler angles from degrees to radians
    # initial_orientation_rad = [math.radians(angle) for angle in initial_orientation_deg]

    # # Convert Euler angles to quaternion
    # initial_orientation_quat = EulerXYZ2Quaternion(initial_orientation_rad)

    rospy.loginfo("Moving to initial position")
    # result = cartesian_pose_client(initial_position, initial_orientation_quat)
    result = joint_angle_client(initial_position)
    if result:
        rospy.loginfo("Moved to initial position successfully")
    else:
        rospy.logerr("Failed to move to initial position")

    # # Add a 3-second delay at the end
    # rospy.loginfo("Waiting for 4 seconds...")
    # time.sleep(8)
    # rospy.loginfo("Delay complete")

def handle_move_robot(req):

    # global currentCartesianCommand
    getcurrentCartesianCommand()
    pose_mq, pose_mdeg, pose_mrad = unitParser(req.unit, req.pose_value, req.relative)
    rospy.loginfo(f"pose_mq: {pose_mq}")  # Debug print
    try:
        poses = [float(n) for n in pose_mq]
        rospy.loginfo(f"poses: {poses}")  # Debug print

        if len(poses) < 7:
            rospy.logerr("Insufficient pose values")
            return MoveRobotResponse(success=False, message="Insufficient pose values")
        
        result = cartesian_pose_client(poses[:3], poses[3:])
        return MoveRobotResponse(success=True, message="Cartesian pose sent!")
    except rospy.ROSInterruptException:
        return MoveRobotResponse(success=False, message="Program interrupted before completion")


def move_robot_server():
    rospy.init_node('move_robot_server')
    s = rospy.Service('move_robot', MoveRobot, handle_move_robot)
    rospy.loginfo("Ready to move robot.")
    move_to_initial_position() 
    rospy.spin()


if __name__ == "__main__":
    move_robot_server()
