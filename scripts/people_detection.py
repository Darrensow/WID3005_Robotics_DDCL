#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
import os



def callback(data):

    if data.data:
        rospy.loginfo("Found person")
        os.system("mpg123 /home/mustar/catkin_ws/src/ddcl_project/scripts/assets/sounds-effect.mp3")
        # os._exit(0)

if __name__ == '__main__':
    rospy.init_node("listener", anonymous=True)
    rospy.loginfo("Successfully launch")
    # Subscribes to the topic 'yolo_launcher' and specifies that the callback
    # function callback should be called when a message is received
    rospy.Subscriber("yolo_launcher", Bool,  callback=callback)
    rospy.spin()


