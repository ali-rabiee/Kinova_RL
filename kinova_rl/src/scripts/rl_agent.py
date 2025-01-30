#!/usr/bin/env python3

#!/usr/bin/env python3

import rospy
import torch
import collections
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge, CvBridgeError
from kinova_msgs.msg import Action  # Import your custom Action message
from DQN_net import DQN  # Assuming your DQN model is defined in DQN_net
import cv2
import numpy as np

class RLAgentNode:
    def __init__(self):
        rospy.init_node('rl_agent_node')
        
        # Parameters
        model_path = '/home/tnlab/Projects/github/sharedcontrol_DQN_Kinova/models/pickplace_seg_v4_bs64_ss4_rb30000_gamma0.99_decaylf120000_lr0.001.pt'
        self.stack_size = 4
        self.screen_height = 64 
        self.screen_width = 64
        self.n_actions = 5
        self.model_path = rospy.get_param('~model_path', model_path)
        self.frequency = rospy.get_param('~frequency', 0.5)  # 0.5 Hz for 2 seconds interval
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = self.load_model(self.model_path)
        
        # Initialize data containers
        self.bridge = CvBridge()
        self.stacked_frames = collections.deque(self.stack_size * [torch.zeros(self.screen_height, self.screen_width)], maxlen=self.stack_size)
        self.stacked_head_movements = collections.deque(self.stack_size * [0], maxlen=self.stack_size)
        
        # Subscribers
        self.frame_sub = rospy.Subscriber('/yolo/segmentation_mask_2d', Image, self.frame_callback)
        self.head_movement_sub = rospy.Subscriber('/head_position', String, self.head_movement_callback)
        
        # Publisher
        self.action_pub = rospy.Publisher('/robot_action', Action, queue_size=10)
        
        # Timer to control the frequency of action publication
        rospy.Timer(rospy.Duration(2), self.publish_action)

    def load_model(self, path):
        model = DQN(self.screen_height, self.screen_width, self.n_actions, stack_size=self.stack_size).to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        return model

    def frame_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            # Resize the image to 64x64
            frame_resized = cv2.resize(frame, (self.screen_height , self.screen_width))
            # print(frame_resized.shape)
            # Normalize the image
            
            # Convert the image to a tensor
            frame_tensor = torch.from_numpy(frame_resized).unsqueeze(0).unsqueeze(0)
            # print(frame_tensor.shape)
            
            # Add the frame to the deque
            self.stacked_frames.append(frame_tensor)
            
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert image: {e}")

    def head_movement_callback(self, msg):
        data = msg.data
        head_pose = 0
        if data == 'Left':
            head_pose = 1 
        elif data == 'Right':
            head_pose = -1

        head_pose = torch.tensor([head_pose], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.stacked_head_movements.append(head_pose)

    def publish_action(self, event):
        if len(self.stacked_frames) < self.stack_size:
            rospy.loginfo("Not enough frames yet")
            return
        
        if len(self.stacked_head_movements) < self.stack_size:
            rospy.loginfo("Not enough head movement data yet")
            return
        
        stacked_frames_t = torch.cat(tuple(self.stacked_frames), dim=1).to(self.device)
        stacked_head_movements_t = torch.cat(tuple(self.stacked_head_movements), dim=1).to(self.device)

        # Select action using the policy network
        # print(stacked_frames_t.shape)
        action = self.policy_net(stacked_frames_t, stacked_head_movements_t).max(1)[1].view(1, 1).item()

        # Publish the action
        action_msg = Action()
        action_msg.action = int(action)
        # print(action_msg.action) 
        self.action_pub.publish(action_msg)

if __name__ == '__main__':
    try:
        node = RLAgentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
