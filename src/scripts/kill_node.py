#!/usr/bin/env python3

import rospy
import subprocess

def monitor_params():
    rospy.init_node('kill_node', anonymous=True)

    rate = rospy.Rate(1)  # Check the parameters every second

    while not rospy.is_shutdown():
        episode_done = rospy.get_param('/episode_done', False)

        if episode_done:
            rospy.loginfo("Both actions executed. Killing all ROS nodes...")
            
            # Call the shell script to kill all ROS nodes
            subprocess.call(['/home/tnlab/catkin_ws/src/kinova-ros/kinova_rl/kill_ros_nodes.sh'])
            break

        rate.sleep()

if __name__ == '__main__':
    try:
        monitor_params()
    except rospy.ROSInterruptException:
        pass
