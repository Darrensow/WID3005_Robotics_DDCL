#!/bin/bash

roslaunch usb_cam usb_cam-test.launch &
sleep 3
roslaunch robot_vision_openvino yolo_ros.launch &
sleep 3
rosrun ddcl_project people_detection.py
