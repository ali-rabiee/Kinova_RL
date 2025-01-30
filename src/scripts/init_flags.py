#!/usr/bin/env python3

import rospy

def initialize_flags():
    rospy.set_param('/grasp_executed', False)
    rospy.set_param('/release_executed', False)
    rospy.set_param('/episode_done', False)

if __name__ == '__main__':
    rospy.init_node('initialize_flags_node', anonymous=True)
    initialize_flags()
    rospy.loginfo("Initialized flags on the parameter server")
