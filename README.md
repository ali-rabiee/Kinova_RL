# Kinova RL: Shared Control Deep Reinforcement Learning for Robotic Manipulation

## Overview

This ROS package implements a shared control Deep Reinforcement Learning (DRL) system for robotic manipulation using a Kinova robotic arm. The system integrates visual input from a camera, head motion data from an IMU sensor, and a trained DQN (Deep Q-Network) agent to control the robot's actions.

## Key Components

1. **RL Agent** (`rl_agent.py`): 
   - Loads a pre-trained DQN model
   - Processes visual input and head motion data
   - Publishes robot actions based on the DQN policy

2. **Action Executor** (`action_executor.py`):
   - Subscribes to robot actions
   - Executes corresponding movements on the Kinova arm
   - Implements both pick-and-place and reach-to-grasp tasks

3. **Head Position Publisher** (`head_position_publisher.py`):
   - Reads data from an IMU sensor attached to the human's head
   - Publishes head position (Left, Right, or Neutral)

4. **Move Robot Server** (`move_robot_server.py`):
   - Provides a ROS service for moving the robot to specific positions
   - Handles cartesian pose commands

5. **Move Finger Server** (`move_finger_server.py`):
   - Provides a ROS service for controlling the robot's fingers
   - Manages gripper actions

## Prerequisites

- ROS (tested on ROS Noetic)
- Python 3
- PyTorch
- OpenCV
- Kinova ROS packages
- YOLO (for segmentation)

## Installation

1. Clone this repository into your catkin workspace:
   ```
   cd ~/catkin_ws/src
   git clone [your-repo-url] kinova_rl
   ```

2. Install dependencies:
   ```
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Build the package:
   ```
   cd ~/catkin_ws
   catkin_make
   ```

## Usage

1. Launch the Kinova arm and necessary drivers.

2. Launch the RL agent:
   ```
   roslaunch kinova_rl kinova_run.launch
   ```

## Configuration

- Adjust parameters in the launch files or directly in the node scripts as needed.
- Ensure the correct model path is set for the RL agent in `rl_agent.py`.

## Customization

- Modify the action space in `action_executor.py` to change the robot's movement patterns.
- Adjust the DQN architecture in `DQN_net.py` (not provided in the given files, but assumed to exist).


## Acknowledgements

This project uses the Kinova ROS stack and incorporates Deep Q-Learning for robotic control. You can access the Kinova ROS package in this repo: https://github.com/Kinovarobotics/kinova-ros
