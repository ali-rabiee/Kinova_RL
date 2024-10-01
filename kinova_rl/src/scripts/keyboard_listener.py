#!/usr/bin/env python3

import rospy
from pynput import keyboard
from kinova_rl.srv import MoveRobot, MoveRobotRequest, MoveFingers, MoveFingersRequest
import threading
import time

# Cooldown period in seconds
COOLDOWN_PERIOD = 0.5  # Adjust this value as needed

# State variables
key_state = {}
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

def on_press(key):
    global last_command_time
    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_PERIOD:
        return
    last_command_time = current_time

    try:
        if key.char not in key_state or not key_state[key.char]:
            key_state[key.char] = True
            if key.char == 'w':  # Move forward
                call_move_robot_service([0, -0.02, 0, 0, 0, 0])
                rospy.loginfo("Moving forward")
            elif key.char == 's':  # Move backward
                call_move_robot_service([0, 0.02, 0, 0, 0, 0])
                rospy.loginfo("Moving backward")
            elif key.char == 'a':  # Move left
                call_move_robot_service([0.02, 0, 0, 0, 0, 0])
                rospy.loginfo("Moving left")
            elif key.char == 'd':  # Move right
                call_move_robot_service([-0.02, 0, 0, 0, 0, 0])
                rospy.loginfo("Moving right")
            elif key.char == 'q':  # Move up
                call_move_robot_service([0, 0, 0.02, 0, 0, 0])
                rospy.loginfo("Moving up")
            elif key.char == 'e':  # Move down
                call_move_robot_service([0, 0, -0.02, 0, 0, 0])
                rospy.loginfo("Moving down")
            elif key.char == 'z':  # Close fingers
                call_move_fingers_service([6800, 6800, 6800])  
                rospy.loginfo("Closing fingers")
            elif key.char == 'x':  # Open fingers
                call_move_fingers_service([0, 0, 0])  # Assuming 0 is fully open
                rospy.loginfo("Opening fingers")
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        # Stop listener
        return False

def listener():
    rospy.init_node('keyboard_listener', anonymous=True)
    rospy.loginfo("Starting keyboard listener node...")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
