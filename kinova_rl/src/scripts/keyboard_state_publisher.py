#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from pynput import keyboard
import threading

class KeyboardStatePublisher:
    def __init__(self):
        rospy.init_node('keyboard_state_publisher', anonymous=True)
        self.pub = rospy.Publisher('keyboard_state', String, queue_size=10)
        self.rate = rospy.Rate(10)  
        self.key_state = {}
        self.publish_thread = threading.Thread(target=self.publish_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def on_press(self, key):
        try:
            self.key_state[key.char] = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.key_state[key.char] = False
        except AttributeError:
            pass

        try:
            if key.char == 'p':
                print("Letter 'p' pressed")
                return False
            
        except AttributeError:
        # Handle special keys that do not have a 'char' attribute
            pass


    def get_state_string(self):
        if not self.key_state:
            return "Neutral"
        
        active_keys = [k for k, v in self.key_state.items() if v]
        return ','.join(active_keys) if active_keys else "Neutral"

    def publish_loop(self):
        while not rospy.is_shutdown():
            state = self.get_state_string()
            self.pub.publish(state)
            self.rate.sleep()

    def run(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            rospy.loginfo("Keyboard state publisher is running. Press ESC to exit.")
            listener.join()

if __name__ == '__main__':
    try:
        publisher = KeyboardStatePublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass