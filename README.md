# WID3005_Robotics_DDCL

### Instructions to run the code.
<I>P.S. Steps 1 and 2 are not needed if you are deploying the code on JUNO/IO/Jupiter Robot</I>
<ol>
<li>Install ROS Melodic according to http://wiki.ros.org/melodic/Installation/Ubuntu</li>
<li>Git clone github repo</li>

`git clone https://github.com/robocupathomeedu/rc-home-edu-learn-ros.git`

<li>Create a new ROS package, following steps in http://wiki.ros.org/ROS/Tutorials/CreatingPackage</li>

`cd ~/catkin_ws/src`

`catkin_create_pkg task1_2 rospy roscpp std_msgs sensor_msgs cv_bridge`

`cd ..`

`catkin_make`

<li>Create a folder named scripts in the package</li>

`mkdir scripts`

<li>Download and paste all the files from scripts into the scripts folder that you have created</li>

## Techniques 
<ol>
<li><b>For Welcoming Message</b></li>

- Execute sound_play.py then only run play_message.py 

`rosrun sound_play soundplay_node.py`

`rosrun ddcl_project play_message.py`

<li><b>For People Detection</b></li>

- Run the people_detection.py file

`rosrun ddcl_project people_detection.py`

<li><b>For object recognition with Yolob></li>

- Execute yolo6_launcher.py to initiate camera capture for object recognition.

`rosrun ddcl_project yolo6_launcher.py`

</ol>




