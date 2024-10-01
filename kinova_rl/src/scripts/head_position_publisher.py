#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial

def talker():
    pub = rospy.Publisher('head_position', String, queue_size=10)
    rospy.init_node('head_position_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    ser = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust the port name if needed

    while not rospy.is_shutdown():
        if ser.in_waiting > 0:
            head_angel = float(ser.readline().decode('utf-8').strip())
            head_angel += 12 # Offset
            if head_angel < -20:
                head_position = "Right"
            elif head_angel > 20:
                head_position = "Left"
            else:
                head_position = "Neutral"

            # print(head_position)

            # rospy.loginfo(head_position)
            pub.publish(head_position)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
