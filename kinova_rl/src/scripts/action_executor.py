#!/usr/bin/env python3

import rospy
from kinova_rl.srv import MoveRobot, MoveRobotRequest, MoveFingers, MoveFingersRequest
from kinova_msgs.msg import Action  # Assuming the custom message is in my_robot_msgs
import time

# Cooldown period in seconds
COOLDOWN_PERIOD = 0.5  # Adjust this value as needed

# State variables
last_command_time = time.time()

def call_move_robot_service(pose_value, unit='mdeg', relative=True):
    rospy.wait_for_service('move_robot')
    try:
        move_robot = rospy.ServiceProxy('move_robot', MoveRobot)
        request = MoveRobotRequest()
        request.pose_value = pose_value
        request.unit = unit
        request.relative = relative
        response = move_robot(request)
        rospy.loginfo(response.message)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def call_move_fingers_service(finger_positions):
    rospy.wait_for_service('move_fingers')
    try:
        move_fingers = rospy.ServiceProxy('move_fingers', MoveFingers)
        request = MoveFingersRequest()
        request.finger_positions = finger_positions
        response = move_fingers(request)
        rospy.loginfo(response.message)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def action_callback1(msg):
    #### Pick and place task - Journal ####
    global last_command_time
    grasp_executed = rospy.get_param('/grasp_executed', False)
    release_executed = rospy.get_param('/release_executed', False)

    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_PERIOD:
        return
    last_command_time = current_time

    action = msg.action
    if action == 0:  # Move left
        call_move_robot_service([0.04, 0, 0, 0, 0, 0])
        rospy.loginfo("Moving left")
    elif action == 1:  # Move right
        call_move_robot_service([-0.04, 0, 0, 0, 0, 0])
        rospy.loginfo("Moving right")
    elif action == 2:  # Hold (No action)
        rospy.loginfo("Holding position")
    elif action == 3:  # Move forward
        call_move_robot_service([0, -0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving forward")
    elif action == 4:  # Move backward
        call_move_robot_service([0, 0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving backward")
    elif action == 5: # Grasp
        # if not grasp_executed:
        call_move_fingers_service([6800, 6800, 6800])
        call_move_robot_service([0, 0, 0.1, 0, 0, 0])
        # rospy.set_param('/grasp_executed', True)
        rospy.loginfo("Grasping")
    elif action == 6: # Release
        # if not release_executed:
        call_move_robot_service([0, -0.15, 0, 0, 0, 0]) # forward
        # call_move_robot_service([0, 0, -0.06, 0, 0, 0]) # down
        call_move_fingers_service([0, 0, 0]) # open fingers
        call_move_robot_service([0, 0, 0.05, 0, 0, 0]) # open fingers
        rospy.loginfo("Releasing")
        rospy.set_param('/episode_done', True)


def action_callback2(msg):
    #### Reach-to-grasp task - ICRA ####
    global last_command_time
    grasp_executed = rospy.get_param('/grasp_executed', False)
    release_executed = rospy.get_param('/release_executed', False)

    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_PERIOD:
        return
    last_command_time = current_time

    action = msg.action
    if action == 0:  # Move left
        call_move_robot_service([0.03, -0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving left")
    elif action == 1:  # Move right
        call_move_robot_service([-0.03, -0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving right")
    elif action == 2:  # Hold (No action)
        rospy.loginfo("Holding position")
    elif action == 3:  # Move forward
        call_move_robot_service([0, -0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving forward")
    elif action == 4:  # Move backward
        call_move_robot_service([0, 0.02, 0, 0, 0, 0])
        rospy.loginfo("Moving backward")
    elif action == 5: # Grasp
        # if not grasp_executed:
        call_move_robot_service([0, -0.02, 0, 0, 0, 0]) # forward
        call_move_fingers_service([6800, 6800, 6800])
        call_move_robot_service([0, 0, 0.1, 0, 0, 0])
        # rospy.set_param('/grasp_executed', True)
        rospy.loginfo("Grasping")
        rospy.set_param('/episode_done', True)
    elif action == 6: # Release
        # if not release_executed:
        call_move_robot_service([0, -0.13, 0, 0, 0, 0]) # forward
        # call_move_robot_service([0, 0, -0.06, 0, 0, 0]) # down
        call_move_fingers_service([0, 0, 0]) # open fingers
        call_move_robot_service([0, 0, 0.05, 0, 0, 0]) # open fingers
        rospy.loginfo("Releasing")
        rospy.set_param('/episode_done', True)

def listener():
    rospy.init_node('action_executor', anonymous=True)
    rospy.loginfo("Starting action executor node...")

    rospy.Subscriber('/robot_action', Action, action_callback2) 

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

