#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import cv2
import numpy as np
import mediapipe as mp


def detector():
    pub = rospy.Publisher('object_detector', String, queue_size=10)
    pub2 = rospy.Publisher('detection', String, queue_size=10)
    rospy.init_node('detector', anonymous = True)
    rate = rospy.Rate(10)
    #while not rospy.is_shutdown():
    #image_path = "test_img/classroom.jpg"
    image_path = "/home/jiamun/catkin_ws/src/robocup_pkg/launch/muggle_bot_object_detection/test_img/standing.jpeg"
    detected = inference(image_path)
    #print(detected)
    rospy.loginfo(detected) 
    pub.publish(detected)
    pub2.publish(detected)
    #image = Image.open('temp.jpeg')
    #image.show()
    rate.sleep()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_from_hip(knee_coor, hip_coor):
    # down_vector = np.array([0, 1])
    down_vector = np.array([0, 1, 0])
    knee_vector_from_hip = unit_vector(np.array(knee_coor) - np.array(hip_coor))
    return np.arccos(np.clip(np.dot(down_vector, knee_vector_from_hip), -1.0, 1.0))*180/np.pi

def gesture_detection(pose_result):
    def coordinate_vector(landmark):
        # return (landmark.x, landmark.y)
        return (landmark.x, landmark.y, landmark.z)

    left_hip = pose_result.pose_landmarks.landmark[23]
    right_hip = pose_result.pose_landmarks.landmark[24]
    left_knee = pose_result.pose_landmarks.landmark[25]
    right_knee = pose_result.pose_landmarks.landmark[26]
    left_hip_vector = coordinate_vector(left_hip)
    right_hip_vector = coordinate_vector(right_hip)
    left_knee_vector = coordinate_vector(left_knee)
    right_knee_vector = coordinate_vector(right_knee)

    left_thigh_angle = angle_from_hip(left_knee_vector, left_hip_vector)
    right_thigh_angle = angle_from_hip(right_knee_vector, right_hip_vector)

    """ Assume standing if both thighs' angles from -ve y-axis are < 45 """ 
    return 'standing' if left_thigh_angle < 45 and right_thigh_angle < 45 else 'sitting'

def inference(image_path):
    image = cv2.imread(image_path)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils 
    #mp_drawing_styles = mp.solutions.drawing_styles
    font = cv2.FONT_HERSHEY_SIMPLEX
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # End if no pose detected
        if not results.pose_landmarks:
            return print("No human pose detected")
        
        # Annotate image with pose landmarks
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
            #landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()

        # Annotate if standing or sitting
        gesture = gesture_detection(results)
        cv2.putText(annotated_image, f'Gesture: {gesture}', (5, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture detection", annotated_image)
        print(f"The person is {gesture}")
        cv2.waitKey(10000)  # wait for any key
        cv2.destroyAllWindows()  # close the image window

        return gesture 

if __name__ == "__main__":
    try:
        detector()
    except rospy.ROSInterruptException:
        pass