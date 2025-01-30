#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from kinova_msgs.msg import Action

class ManualControlNode:
    def __init__(self):
        rospy.init_node('manual_control_node')
        
        # Parameters
        self.frequency = rospy.get_param('~frequency', 1)  # 1 Hz
        
        # Subscriber
        self.keyboard_state_sub = rospy.Subscriber('/keyboard_state', String, self.keyboard_state_callback)
        
        # Publisher
        self.action_pub = rospy.Publisher('/robot_action', Action, queue_size=10)
        
        # Timer to control the frequency of action publication
        rospy.Timer(rospy.Duration(1.0 / self.frequency), self.publish_action)

        # Store the last received keyboard state
        self.last_keyboard_state = "neutral"

        # Initialize grasp/release flags from the parameter server
        if not rospy.has_param('/grasp_executed'):
            rospy.set_param('/grasp_executed', False)
        if not rospy.has_param('/release_executed'):
            rospy.set_param('/release_executed', False)

    def keyboard_state_callback(self, msg):
        self.last_keyboard_state = msg.data

    def publish_action(self, event):
        action_msg = Action()
        
        # Retrieve the current state of grasp and release flags
        grasp_executed = rospy.get_param('/grasp_executed', False)
        release_executed = rospy.get_param('/release_executed', False)

        if self.last_keyboard_state == 'a':  # left
            action_msg.action = 0
        elif self.last_keyboard_state == 'd':  # right
            action_msg.action = 1
        elif self.last_keyboard_state == 'w':  # forward
            action_msg.action = 3
        elif self.last_keyboard_state == 's':  # backward
            action_msg.action = 4
        elif self.last_keyboard_state == 'g' and not grasp_executed:  # grasp (only if not executed before)
            action_msg.action = 5
            rospy.set_param('/grasp_executed', True)  # Set the grasp flag to True
            rospy.set_param('/release_executed', False)  # Reset release flag
            rospy.loginfo("Grasp action executed.")
        elif self.last_keyboard_state == 'r' and not release_executed:  # release (only if not executed before)
            action_msg.action = 6
            rospy.set_param('/release_executed', True)  # Set the release flag to True
            rospy.set_param('/grasp_executed', False)  # Reset grasp flag
            rospy.loginfo("Release action executed.")
        else:  # neutral or any other state
            action_msg.action = 2  # hold

        # Publish the action message
        self.action_pub.publish(action_msg)
        rospy.loginfo(f"Published action: {action_msg.action}")

if __name__ == '__main__':
    try:
        node = ManualControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
