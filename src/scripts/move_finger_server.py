#!/usr/bin/env python3

import rospy
import actionlib
import kinova_msgs.msg
from kinova_rl.srv import MoveFingers, MoveFingersResponse

""" Global variable """
prefix = 'j2n6s300_'
finger_maxDist = 18.9 / 2 / 1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentFingerPosition = [0.0, 0.0, 0.0]

def gripper_client(finger_positions):
    action_address = '/' + prefix + 'driver/fingers_action/finger_positions'
    rospy.loginfo(f"Waiting for action server: {action_address}")
    client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.SetFingersPositionAction)

    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("Action server not available at " + action_address)
        return None

    goal = kinova_msgs.msg.SetFingersPositionGoal()
    goal.fingers.finger1 = float(finger_positions[0])
    goal.fingers.finger2 = float(finger_positions[1])
    goal.fingers.finger3 = float(finger_positions[2]) if len(finger_positions) > 2 else 0.0

    rospy.loginfo(f"Sending goal: {goal.fingers}")
    client.send_goal(goal)

    if client.wait_for_result(rospy.Duration(5.0)):
        result = client.get_result()
        rospy.loginfo(f"Action completed with result: {result}")
        getCurrentFingerPosition(prefix)
        return result
    else:
        client.cancel_all_goals()
        rospy.logerr('The gripper action timed-out')
        return None

def getCurrentFingerPosition(prefix_):
    topic_address = '/' + prefix_ + 'driver/out/finger_position'
    rospy.Subscriber(topic_address, kinova_msgs.msg.FingerPosition, setCurrentFingerPosition)
    rospy.wait_for_message(topic_address, kinova_msgs.msg.FingerPosition)
    rospy.loginfo('Obtained current finger position.')

def setCurrentFingerPosition(feedback):
    global currentFingerPosition
    currentFingerPosition[0] = feedback.finger1
    currentFingerPosition[1] = feedback.finger2
    currentFingerPosition[2] = feedback.finger3

def handle_move_fingers(req):
    rospy.loginfo(f"Received finger positions: {req.finger_positions}")
    result = gripper_client(req.finger_positions)
    if result:
        return MoveFingersResponse(success=True, message="Finger positions set successfully")
    else:
        return MoveFingersResponse(success=False, message="Failed to set finger positions")

def move_fingers_server():
    rospy.init_node('move_fingers_server')
    s = rospy.Service('move_fingers', MoveFingers, handle_move_fingers)
    rospy.loginfo("Ready to move fingers.")
    rospy.spin()

if __name__ == "__main__":
    move_fingers_server()
