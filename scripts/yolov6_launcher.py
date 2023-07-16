#!/usr/bin/env python3

import cv2 as cv
import rospy
from yolov6_model import Custom_Inferer
from std_msgs.msg import Bool



img_size = 448
inferer = Custom_Inferer(img_size=img_size, weights='./assets/yolov6lite_m.pt')

counter = 0
# Creates a publisher object that publish messages of type Bool on the topic named 'yolo_launcher'
pub = rospy.Publisher('yolo_launcher', Bool, queue_size=1)
# Initializes the ROS node with the name 'human_detector'
rospy.init_node('human_detector')
r = rospy.Rate(1)

cap = cv.VideoCapture(0)

while not rospy.is_shutdown():
    # Get video frame
    ret, frame = cap.read()
    img, pred = inferer.infer(frame, plot_box_and_label=True)

    print(f"Object(s) detected: {pred}")
    cv.imshow("Object Detection", frame)

    if "Person" in pred:
        if counter % 5 == 0 and counter != 0: 
            res = Bool()
            res.data = True
            # Publishes result to subscribers listening to yolo_launcher
            pub.publish(res)
            rospy.loginfo("Person is detected")

            counter = 0
        else:
            counter += 1

    r.sleep()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
