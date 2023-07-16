#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, String
from sound_play.libsoundplay import SoundClient


def callback(data):

    if data.data:
        rospy.loginfo("Executing welcome message")
        play_sound()
        # os._exit(0)


def play_sound():
    sound_handle = SoundClient()
    rospy.sleep(2) # allow time for initialization.
    sound_handle.stopAll()
    sound_handle.say('welcome to DDCL')
    rospy.sleep(3)

if __name__ == '__main__':
    rospy.init_node("listener_welcome_node", anonymous=True)
    rospy.loginfo("Successfully launch")
    rospy.Subscriber("yolo_launcher", Bool,  callback=callback)
    rospy.spin()


