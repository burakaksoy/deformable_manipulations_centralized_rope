#!/usr/bin/env python3

import sys

import rospy

import numpy as np
import time

import PyQt5.QtWidgets as qt_widgets
import PyQt5.QtCore as qt_core
from PyQt5.QtCore import Qt

from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from deformable_manipulations_centralized_rope.srv import ControllerSetParticleState, ControllerSetParticleStateRequest
from deformable_manipulations_centralized_rope.srv import ResetParticlePosition, ResetParticlePositionRequest
from std_srvs.srv import SetBool, SetBoolRequest
from std_srvs.srv import Empty, EmptyResponse

import math
import tf.transformations as transformations

"""
Author: Burak Aksoy

The fake_odom_publisher_gui_node simulates the publishing of odometry data for 
a set of particles within a ROS network. Utilizing the PyQt5 library, the node
provides a GUI interface which displays a series of buttons representing each 
particle, allowing the user to manually toggle the publishing state of 
individual particles.

Odometry data, which includes both positional and orientational data, is 
published based on the current Twist message received from the 
/spacenav/twist topic and the marker data received from the /dlo_points topic. 

A simple time-step integration technique is employed to calculate the new 
position of the particle in the YZ plane based on the twist linear velocities. 
The orientation, represented as a quaternion, is determined from the direction
vector formed by two points in a line list marker message.

The node also implements a rospy Timer to publish odometry messages at a 
predefined rate (pub_rate_odom) for all the particles which are currently 
selected in the GUI. The Timer callback function checks the state of each 
button in the GUI and publishes odometry data if the button (i.e. the particle)
is selected.

The node continuously checks whether the ROS network is still running. If the 
ROS network is shut down, the node also shuts down its GUI interface, ensuring
a clean exit. 

Lastly, the node maintains a dictionary of the last time-stamp 
requests to ensure that velocity commands are spaced appropriately, and any 
time-step greater than the maximum permissible time-step is ignored.
"""


# Velocity commands will only be considered if they are spaced closer than MAX_TIMESTEP
MAX_TIMESTEP = 0.04 # Set it to ~ twice of pub rate odom

class FakeOdomPublisherGUI(qt_widgets.QWidget):
    def __init__(self):
        super(FakeOdomPublisherGUI, self).__init__()
        self.shutdown_timer = qt_core.QTimer()

        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.initial_values_set = False  # Initialization state variable

        self.particles = [] # All particles that is union of controllable and uncontrollable particles
        self.follower_particles = [] # Particles that are controllable by the robots 
        self.binded_particles = [] # particles that are uncontrollable and binded to the leader frame,
        # (e.g human hand held points when the neck joint is the leader)

        self.odom_topic_prefix = None
        self.odom_topic_leader = None
        while (not self.particles):
            try:
                self.particles = rospy.get_param("/custom_static_particles")
                self.follower_particles = rospy.get_param("/follower_particles") 
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix")
                self.odom_topic_leader = rospy.get_param("/odom_topic_leader")
            except:
                rospy.logwarn("No particles obtained from ROS parameters. GUI will be empty.")
                time.sleep(0.5)

        self.binded_particles = list(set(self.particles) - set(self.follower_particles))


        self.odom_publishers = {}
        self.info_binded_pose_publishers = {}

        self.particle_positions = {}
        self.particle_orientations = {}

        self.initialize_leader_position()

        self.binded_relative_poses = {} # stores Pose() msg of ROS geometry_msgs (Pose.position and Pose.orientation)

        self.createUI()
        
        self.spacenav_twist = Twist() # set it to an empty twist message
        self.last_spacenav_twist_time = rospy.Time.now() # for timestamping of the last twist msg
        # self.spacenav_twist_wait_timeout = rospy.Duration(2.0*(1.0/self.pub_rate_odom)) # timeout duration to wait twist msg before zeroing in seconds
        self.spacenav_twist_wait_timeout = rospy.Duration(1.0) # timeout duration to wait twist msg before zeroing in seconds


        self.sub_twist = rospy.Subscriber("/spacenav/twist", Twist, self.spacenav_twist_callback, queue_size=1)
        self.sub_marker = rospy.Subscriber("/dlo_points", Marker, self.marker_callback, queue_size=1)

        self.last_timestep_requests = {}

        self.odom_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)
        

    def createUI(self):
        self.layout = qt_widgets.QVBoxLayout(self)

        self.buttons_manual = {} # To enable/disable manual control
        self.pause_controller_buttons = {} # for follower particles
        self.start_controller_buttons = {} # for leader particle only, (Yes, there is only one leader always but for the sake of generality this is a dictionary)
        self.bind_to_leader_buttons = {} # for binded particles

        self.text_inputs_pos = {} # to manually set x y z positions of the particles
        self.text_inputs_ori = {} # to manually set x y z orientations of the particles (Euler RPY, degree input)
        
        # Combine the lists with boundary markers
        combined_particles = (["leader"] + ["_boundary_marker_"] +
                            self.binded_particles + ["_boundary_marker_"] +
                            self.follower_particles)

        # Create rows for the (custom static) particles and the leader
        for particle in combined_particles:
            # insert a horizontal line between each list (["leader"], self.binded_particles, and self.follower_particles)
            if particle == "_boundary_marker_":
                # Add a horizontal line here
                h_line = qt_widgets.QFrame()
                h_line.setFrameShape(qt_widgets.QFrame.HLine)
                h_line.setFrameShadow(qt_widgets.QFrame.Sunken)
                self.layout.addWidget(h_line)  # Assuming 'self.layout' is your main layout
                continue

            # Create QHBoxLayout for each row
            row_layout = qt_widgets.QHBoxLayout()

            manual_control_button = qt_widgets.QPushButton()
            manual_control_button.setText("Manually Control " + str(particle))
            manual_control_button.setCheckable(True)  # Enables toggle behavior
            manual_control_button.setChecked(False)
            manual_control_button.clicked.connect(lambda _, p=particle: self.manual_control_button_pressed_cb(p))
            row_layout.addWidget(manual_control_button) # Add button to row layout

            self.buttons_manual[particle] = manual_control_button

            # ------------------------------------------------
            # Add a separator vertical line here
            separator = qt_widgets.QFrame()
            separator.setFrameShape(qt_widgets.QFrame.VLine)
            separator.setFrameShadow(qt_widgets.QFrame.Sunken)
            row_layout.addWidget(separator)
            # ------------------------------------------------

            # Add button to get current pose to GUI for easy set position and orientation operations
            get_pose_button = qt_widgets.QPushButton()
            get_pose_button.setText("Get Pose")
            get_pose_button.clicked.connect(lambda _, p=particle: self.get_pose_button_pressed_cb(p))
            row_layout.addWidget(get_pose_button)

            # ------------------------------------------------
            # Add a separator vertical line here
            separator = qt_widgets.QFrame()
            separator.setFrameShape(qt_widgets.QFrame.VLine)
            separator.setFrameShadow(qt_widgets.QFrame.Sunken)
            row_layout.addWidget(separator)
            # ------------------------------------------------

            # Create LineEdits and Add to row layout
            self.text_inputs_pos[particle] = {}
            for axis in ['x', 'y', 'z']:
                label = qt_widgets.QLabel(axis + ':')
                line_edit = qt_widgets.QLineEdit()

                if axis == 'x':
                    line_edit.setText(str(0.0))
                elif axis == 'y':
                    line_edit.setText(str(0.0))
                elif axis == 'z':
                    line_edit.setText(str(0.0))
                
                row_layout.addWidget(label)
                row_layout.addWidget(line_edit)
                self.text_inputs_pos[particle][axis] = line_edit
            
            
            # Create Set Position button
            set_pos_button = qt_widgets.QPushButton()
            set_pos_button.setText("Set Position")
            set_pos_button.clicked.connect(lambda _, p=particle: self.set_position_cb(p))
            row_layout.addWidget(set_pos_button)

            # ------------------------------------------------
            # Add a separator vertical line here
            separator = qt_widgets.QFrame()
            separator.setFrameShape(qt_widgets.QFrame.VLine)
            separator.setFrameShadow(qt_widgets.QFrame.Sunken)
            row_layout.addWidget(separator)
            # ------------------------------------------------

            # Add Set orientation button text inputs
            self.text_inputs_ori[particle] = {}
            for axis in ['x', 'y', 'z']:
                label = qt_widgets.QLabel(axis + ':')
                line_edit = qt_widgets.QLineEdit()

                if axis == 'x':
                    line_edit.setText(str(0.0))
                elif axis == 'y':
                    line_edit.setText(str(0.0))
                elif axis == 'z':
                    line_edit.setText(str(0.0))
                
                row_layout.addWidget(label)
                row_layout.addWidget(line_edit)
                self.text_inputs_ori[particle][axis] = line_edit

            # Create Set Orientation button
            set_ori_button = qt_widgets.QPushButton()
            set_ori_button.setText("Set Orientation")
            set_ori_button.clicked.connect(lambda _, p=particle: self.set_orientation_cb(p))
            row_layout.addWidget(set_ori_button)

            # ------------------------------------------------
            # Add a separator vertical line here
            separator = qt_widgets.QFrame()
            separator.setFrameShape(qt_widgets.QFrame.VLine)
            separator.setFrameShadow(qt_widgets.QFrame.Sunken)
            row_layout.addWidget(separator)
            # ------------------------------------------------

            # Add button to reset all trails
            clear_trails_button = qt_widgets.QPushButton()
            clear_trails_button.setText("Clear Trail")
            clear_trails_button.clicked.connect(lambda _, p=particle: self.clear_trails_cb(p))
            row_layout.addWidget(clear_trails_button)
            
            # ------------------------------------------------
            # Add a separator vertical line here
            separator = qt_widgets.QFrame()
            separator.setFrameShape(qt_widgets.QFrame.VLine)
            separator.setFrameShadow(qt_widgets.QFrame.Sunken)
            row_layout.addWidget(separator)
            # ------------------------------------------------

            # Create Pause Controller Button or Enable Disable Particle Button
            if particle == "leader":
                start_controller_button = qt_widgets.QPushButton()
                start_controller_button.setText("Start Controller")
                start_controller_button.clicked.connect(lambda _, p=particle: self.start_controller_button_cb(p))
                start_controller_button.setCheckable(True) # Set the pause button as checkable

                row_layout.addWidget(start_controller_button)

                self.start_controller_buttons[particle] = start_controller_button            
            elif particle in self.binded_particles:
                bind_to_leader_button = qt_widgets.QPushButton()
                bind_to_leader_button.setText("Bind to Leader")
                bind_to_leader_button.clicked.connect(lambda _, p=particle: self.bind_to_leader_button_cb(p))
                bind_to_leader_button.setCheckable(True) # Set the pause button as checkable

                row_layout.addWidget(bind_to_leader_button)

                self.bind_to_leader_buttons[particle] = bind_to_leader_button            
            else: # particle in self.follower_particles
                pause_button = qt_widgets.QPushButton()
                pause_button.setText("Disable Control")
                pause_button.clicked.connect(lambda _, p=particle: self.pause_controller_cb(p))
                pause_button.setCheckable(True) # Set the pause button as checkable
                
                row_layout.addWidget(pause_button)

                self.pause_controller_buttons[particle] = pause_button


            # Create Reset DLO positions button
            if particle in self.follower_particles:
                reset_button = qt_widgets.QPushButton()
                reset_button.setText("Reset Controller Position")
                reset_button.clicked.connect(lambda _, p=particle: self.reset_position_cb(p))
                row_layout.addWidget(reset_button)
            elif particle in self.binded_particles: 
                reset_button = qt_widgets.QPushButton()
                reset_button.setText("Reset Controller Position")
                reset_button.setEnabled(False)  # Disable the button
                row_layout.addWidget(reset_button)
            else: # Leader particle
                # Send leader particle to the centroid of the particles
                send_to_centroid_button = qt_widgets.QPushButton()
                send_to_centroid_button.setText("Center Leader Frame")
                send_to_centroid_button.clicked.connect(lambda _, p=particle: self.send_to_centroid_cb(p))
                row_layout.addWidget(send_to_centroid_button)

            # Add row layout to the main layout
            self.layout.addLayout(row_layout)
            
            if particle == "leader":
                self.odom_publishers[particle] = rospy.Publisher(self.odom_topic_leader, Odometry, queue_size=1)
            else:
                self.odom_publishers[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle), Odometry, queue_size=1)

            if particle in self.binded_particles:
                self.info_binded_pose_publishers[particle] = rospy.Publisher( "fake_odom_publisher_gui_info_"+str(particle)+"_target_pose", Marker, queue_size=1)

        self.setLayout(self.layout)

        self.shutdown_timer.timeout.connect(self.check_shutdown)
        self.shutdown_timer.start(1000)  # Timer triggers every 1000 ms (1 second)

    def format_number(self,num, digits=4):
        # When digits = 4, look at the affect of the function
        # print(format_number(5))         # Output: '5.0'
        # print(format_number(5.0))       # Output: '5.0'
        # print(format_number(5.12345))   # Output: '5.1235'
        # print(format_number(5.1000))    # Output: '5.1'
        # print(format_number(123.456789))# Output: '123.4568'

        rounded_num = round(float(num), digits)  # Ensure num is treated as a float
        # Check if the rounded number is an integer
        if rounded_num.is_integer():
            return f'{int(rounded_num)}.0'
        else:
            return f'{rounded_num:.4f}'.rstrip('0').rstrip('.')

    def get_pose_button_pressed_cb(self, particle):
        """
        Gets the current pose of the particle 
        and fills the text_inputs for pos and ori accordingly
        """

        # Check if the particle pose is set
        if (particle in self.particle_positions) and (particle in self.particle_orientations):
            # Get Current Pose of the particle in world frame
            pos = self.particle_positions[particle] # Point() msg of ROS geometry_msgs
            ori = self.particle_orientations[particle] # Quaternion() msg of ROS geometry_msgs

            # Fill the position text inputs with the current position
            self.text_inputs_pos[particle]['x'].setText(self.format_number(pos.x,digits=3)) 
            self.text_inputs_pos[particle]['y'].setText(self.format_number(pos.y,digits=3)) 
            self.text_inputs_pos[particle]['z'].setText(self.format_number(pos.z,digits=3)) 

            # Convert quaternion orientation to RPY (Roll-pitch-yaw) Euler Angles (degrees)
            rpy = np.rad2deg(transformations.euler_from_quaternion([ori.x,ori.y,ori.z,ori.w]))

            # Fill the orientation text  inputs with the current RPY orientation
            self.text_inputs_ori[particle]['x'].setText(self.format_number(rpy[0],digits=1))
            self.text_inputs_ori[particle]['y'].setText(self.format_number(rpy[1],digits=1))
            self.text_inputs_ori[particle]['z'].setText(self.format_number(rpy[2],digits=1))
        else:
            rospy.logwarn(f"Key '{particle}' not found in the particle_positions and particle_orientations dictionaries.")

    def set_position_cb(self, particle):
        pose = Pose()
        pose.position.x = float(self.text_inputs_pos[particle]['x'].text())
        pose.position.y = float(self.text_inputs_pos[particle]['y'].text())
        pose.position.z = float(self.text_inputs_pos[particle]['z'].text())

        # Keep the same orientation
        pose.orientation = self.particle_orientations[particle]

        # if particle == "leader":
        self.particle_positions[particle] = pose.position
        self.particle_orientations[particle] = pose.orientation

        # Prepare Odometry message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map" 
        odom.pose.pose = pose

        self.odom_publishers[particle].publish(odom)

    def set_orientation_cb(self,particle):
        pose = Pose()

        # Keep the same position
        pose.position = self.particle_positions[particle]

        # Update the orientation with RPY degree input
        th_x = np.deg2rad(float(self.text_inputs_ori[particle]['x'].text())) # rad
        th_y = np.deg2rad(float(self.text_inputs_ori[particle]['y'].text())) # rad
        th_z = np.deg2rad(float(self.text_inputs_ori[particle]['z'].text())) # rad

        q = transformations.quaternion_from_euler(th_x, th_y,th_z)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        # if particle == "leader":
        self.particle_positions[particle] = pose.position
        self.particle_orientations[particle] = pose.orientation

        # Prepare Odometry message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map" 
        odom.pose.pose = pose

        self.odom_publishers[particle].publish(odom)

    def start_controller_button_cb(self, particle):
        # Call pause/resume service
        service_name = '/dlo_velocity_controller/set_enable_controller'
        
        rospy.wait_for_service(service_name, timeout=2.0)
        try:
            set_enable_service = rospy.ServiceProxy(service_name, SetBool)
            
            # Prepare the request   
            request = SetBoolRequest()
            request.data = self.start_controller_buttons[particle].isChecked() 
            
            # Call the service with the request
            response = set_enable_service(request)
            
            if not response.success:  # If the response for success is false
                # Make the button return to its previous state because it has failed
                self.start_controller_buttons[particle].setChecked(not self.start_controller_buttons[particle].isChecked())

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            # Make the button return to its previous state because it has failed
            self.start_controller_buttons[particle].setChecked(not self.start_controller_buttons[particle].isChecked())

    def pause_controller_cb(self, particle):
        # Call pause/resume service
        service_name = '/dlo_velocity_controller/set_disable_particle_in_controller'
        rospy.wait_for_service(service_name, timeout=2.0)
        try:
            set_disable_particle_service = rospy.ServiceProxy(service_name, ControllerSetParticleState)

            # Prepare the request
            request = ControllerSetParticleStateRequest()
            request.particle_id = particle  # 'particle' is the ID
            request.enable = not self.pause_controller_buttons[particle].isChecked()

            # Call the service with the request
            response = set_disable_particle_service(request)

            if not response.success:  # If the response for success is false
                # Make the button return to its previous state because it has failed
                self.pause_controller_buttons[particle].setChecked(not self.pause_controller_buttons[particle].isChecked())

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            # Make the button return to its previous state because it has failed
            self.pause_controller_buttons[particle].setChecked(not self.pause_controller_buttons[particle].isChecked())

    def reset_position_cb(self, particle):
        # Call reset positions service
        service_name = '/dlo_velocity_controller/reset_positions' 
        rospy.wait_for_service(service_name, timeout=2.0)
        try:
            reset_positions_service = rospy.ServiceProxy(service_name, ResetParticlePosition)
            # Create a request to the service
            request = ResetParticlePositionRequest()
            request.particle_id = particle

            reset_positions_service(request)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def bind_to_leader_button_cb(self, particle):
        if self.bind_to_leader_buttons[particle].isChecked():
            # Button is currently pressed, no need to set it to False
            print(f"Button for binding particle {particle} to leader is now pressed.")

            # Store binded particle's current pose relative to the leader frame
            if (particle in self.particle_positions) and ("leader" in self.particle_positions):                
                # Current Pose of the leader in world frame
                pos_leader = self.particle_positions["leader"] # Point() msg of ROS geometry_msgs
                ori_leader = self.particle_orientations["leader"] # Quaternion() msg of ROS geometry_msgs

                # Current Pose of the binded particle in world frame
                pos_binded_world = self.particle_positions[particle] # Point() msg of ROS geometry_msgs
                ori_binded_world = self.particle_orientations[particle] # Quaternion() msg of ROS geometry_msgs

                # Convert quaternions to rotation matrices
                rotation_matrix_leader = transformations.quaternion_matrix([ori_leader.x, ori_leader.y, ori_leader.z, ori_leader.w])

                # Calculate the relative position of the binded particle in the leader's frame
                relative_pos = np.dot(rotation_matrix_leader.T, np.array([pos_binded_world.x - pos_leader.x, pos_binded_world.y - pos_leader.y, pos_binded_world.z - pos_leader.z, 1]))[:3]

                # Calculate the relative orientation of the binded particle in the leader's frame
                inv_ori_leader = transformations.quaternion_inverse([ori_leader.x, ori_leader.y, ori_leader.z, ori_leader.w])
                relative_ori = transformations.quaternion_multiply(inv_ori_leader, [ori_binded_world.x, ori_binded_world.y, ori_binded_world.z, ori_binded_world.w])

                # Store the relative position and orientation
                pose = Pose()
                pose.position = Point(*relative_pos)
                pose.orientation = Quaternion(*relative_ori)
                self.binded_relative_poses[particle] = pose # stores Pose() msg of ROS geometry_msgs (Pose.position and Pose.orientation)

                # Make the manual control button unpressed
                self.buttons_manual[particle].setChecked(False)
            else:
                rospy.logwarn("Initial pose values of the object particles or leader is not yet set")
                self.bind_to_leader_buttons[particle].setChecked(False)

        else:
            # Button is currently not pressed, no need to set it to True
            print(f"Button for binding particle {particle} to leader is now NOT pressed.")   

            # Unbind particle from leader by deleting the relative pose from the dictionary
            if particle in self.binded_relative_poses:
                del self.binded_relative_poses[particle]
            else:
                rospy.logerr(f"Key '{particle}' not found in the binded_relative_poses dictionary.")

    def send_to_centroid_cb(self,particle_to_send):
        # Calculate the centroid of the other particles
        centroid_pos = np.zeros(3)
        n_particle = 0 # number of particles to calculate the centroid
        for particle in ["leader"] + self.particles:
            if particle != particle_to_send:
                # Check if the particle pose is set
                if (particle in self.particle_positions) and (particle in self.particle_orientations):
                    # Get Current Pose of the particle in world frame
                    pos = self.particle_positions[particle] # Point() msg of ROS geometry_msgs
                    # ori = self.particle_orientations[particle] # Quaternion() msg of ROS geometry_msgs

                    centroid_pos = centroid_pos + np.array([pos.x,pos.y,pos.z])
                    n_particle = n_particle + 1
                else:
                    rospy.logwarn(f"Key '{particle}' not found in the particle_positions and particle_orientations dictionaries.")

        # Send to centrod
        if n_particle > 0:
            centroid_pos = centroid_pos/n_particle # Take the mean

            pose = Pose()
            pose.position.x = centroid_pos[0]
            pose.position.y = centroid_pos[1]
            pose.position.z = centroid_pos[2]

            # Keep the same orientation
            pose.orientation = self.particle_orientations[particle_to_send]

            if particle_to_send == "leader":
                self.particle_positions[particle_to_send] = pose.position
                self.particle_orientations[particle_to_send] = pose.orientation

            # Prepare Odometry message
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map" 
            odom.pose.pose = pose

            self.odom_publishers[particle_to_send].publish(odom)
        else:
            rospy.logwarn("There is no particle to calculate the centroid.")

    def clear_trails_cb(self, particle):
        # Call reset positions service
        service_name = '/reset_trail_service_' + str(particle)
        rospy.wait_for_service(service_name, timeout=2.0)
        try:
            reset_trail_service = rospy.ServiceProxy(service_name, Empty)
            reset_trail_service()
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def odom_pub_timer_callback(self,event):
        # Reset spacenav_twist to zero if it's been long time since the last arrived
        self.check_spacenav_twist_wait_timeout()

        # Handle manual control of each particle and the leader
        for particle in ["leader"] + self.particles:
            # Do not proceed with the "non-leader" particles until the initial values have been set
            if (particle != "leader") and not((particle in self.particle_positions) and ("leader" in self.particle_positions)):
                continue

            if self.buttons_manual[particle].isChecked():
                dt = self.get_timestep(particle)   
                # dt = 0.01
                # dt = 1. / self.pub_rate_odom

                # simple time step integration using Twist data
                pose = Pose()
                pose.position.x = self.particle_positions[particle].x + dt*self.spacenav_twist.linear.x
                pose.position.y = self.particle_positions[particle].y + dt*self.spacenav_twist.linear.y
                pose.position.z = self.particle_positions[particle].z + dt*self.spacenav_twist.linear.z

                # # --------------------------------------------------------------
                # # To update the orientation with the twist message
                # Calculate the magnitude of the angular velocity vector
                omega_magnitude = math.sqrt(self.spacenav_twist.angular.x**2 + 
                                            self.spacenav_twist.angular.y**2 + 
                                            self.spacenav_twist.angular.z**2)
                
                # print("omega_magnitude: " + str(omega_magnitude))

                pose.orientation = self.particle_orientations[particle]

                if omega_magnitude > 1e-5:  # Avoid division by zero
                    # Create the delta quaternion based on world frame twist
                    delta_quat = transformations.quaternion_about_axis(omega_magnitude * dt, [
                        self.spacenav_twist.angular.x / omega_magnitude,
                        self.spacenav_twist.angular.y / omega_magnitude,
                        self.spacenav_twist.angular.z / omega_magnitude
                    ])
                    
                    # Update the pose's orientation by multiplying delta quaternion with current orientation 
                    # Note the order here. This applies the world frame rotation directly.
                    current_quaternion = (
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w
                    )
                    
                    new_quaternion = transformations.quaternion_multiply(delta_quat, current_quaternion)
                    
                    pose.orientation.x = new_quaternion[0]
                    pose.orientation.y = new_quaternion[1]
                    pose.orientation.z = new_quaternion[2]
                    pose.orientation.w = new_quaternion[3]
                # # --------------------------------------------------------------

                # if particle == "leader":
                self.particle_positions[particle] = pose.position
                self.particle_orientations[particle] = pose.orientation

                
                # Prepare Odometry message
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = "map" 
                odom.pose.pose = pose
                self.odom_publishers[particle].publish(odom)

        # Handle the control of binded particles
        for particle in self.binded_particles:
            # Do not proceed with the "binded" particles until the initial values have been set
            if not (particle in self.particle_positions) and ("leader" in self.particle_positions):
                continue

            if self.bind_to_leader_buttons[particle].isChecked():
                if particle in self.binded_relative_poses:
                    # calculate the pose of the binded particle in world using the relative pose to the leader
                    
                    # Current Pose of the leader in world frame
                    pos_leader = self.particle_positions["leader"] # Point() msg of ROS geometry_msgs
                    ori_leader = self.particle_orientations["leader"] # Quaternion() msg of ROS geometry_msgs

                    # Relative Pose of the binded particle in leader's frame
                    pos_binded_in_leader = self.binded_relative_poses[particle].position # Point() msg of ROS geometry_msgs
                    ori_binded_in_leader = self.binded_relative_poses[particle].orientation # Quaternion() msg of ROS geometry_msgs

                    # Convert quaternions to rotation matrices
                    rotation_matrix_leader = transformations.quaternion_matrix([ori_leader.x, ori_leader.y, ori_leader.z, ori_leader.w])

                    # Calculate the position of the binded particle in the world frame
                    pos_binded_in_world = np.dot(rotation_matrix_leader, np.array([pos_binded_in_leader.x, pos_binded_in_leader.y, pos_binded_in_leader.z, 1]))[:3] + np.array([pos_leader.x, pos_leader.y, pos_leader.z])

                    # Calculate the orientation of the binded particle in the world frame
                    ori_binded_in_world = transformations.quaternion_multiply([ori_leader.x, ori_leader.y, ori_leader.z, ori_leader.w], [ori_binded_in_leader.x, ori_binded_in_leader.y, ori_binded_in_leader.z, ori_binded_in_leader.w])

                    # Now publish the binded particle's pose as an odom msg
                    pose = Pose()
                    pose.position = Point(*pos_binded_in_world) # pos_binded_world
                    pose.orientation = Quaternion(*ori_binded_in_world) # ori_binded_world

                    # Prepare Odometry message
                    odom = Odometry()
                    odom.header.stamp = rospy.Time.now()
                    odom.header.frame_id = "map" 
                    odom.pose.pose = pose
                    self.odom_publishers[particle].publish(odom)

                    # Also publish an arrow to distinguish the binded particles
                    self.publish_arrow_marker(pos_leader,pose,particle)
                else:
                    rospy.logwarn(f"Key '{particle}' not found in the binded_relative_poses dictionary.")



    def publish_arrow_marker(self, leader_position, target_pose, particle):
        # Create a marker for the arrow
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_arrow"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set the scale of the arrow
        marker.scale.x = 0.015  # Shaft diameter
        marker.scale.y = 0.05  # Head diameter
        marker.scale.z = 0.3  # Head length
        
        # Set the color
        marker.color.a = 0.3
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.7
        
        # Set the pose (position and orientation) for the marker
        marker.pose.orientation.w = 1.0  # Orientation (quaternion)
        
        # Set the start and end points of the arrow
        marker.points = []
        start_point = leader_position  # Should be a Point message
        end_point = target_pose.position  # Assuming target_pose is a Pose message
        marker.points.append(start_point)
        marker.points.append(end_point)
        
        # Publish the marker
        self.info_binded_pose_publishers[particle].publish(marker)


    def spacenav_twist_callback(self, twist):
        # self.spacenav_twist.linear.x = twist.linear.x # because we are in YZ plane
        # self.spacenav_twist.linear.y = twist.linear.y
        # self.spacenav_twist.linear.z = twist.linear.z
        # self.spacenav_twist.angular.x = twist.angular.x
        # self.spacenav_twist.angular.y = twist.angular.y  # twist.angular.y because we are in YZ plane
        # self.spacenav_twist.angular.z = twist.angular.z  # twist.angular.z because we are in YZ plane
        
        self.spacenav_twist = twist

        self.last_spacenav_twist_time = rospy.Time.now()

    def check_spacenav_twist_wait_timeout(self):
        if (rospy.Time.now() - self.last_spacenav_twist_time) > self.spacenav_twist_wait_timeout:
            # Reset spacenav_twist to zero after timeout
            self.spacenav_twist = Twist()

            rospy.loginfo("spacenav_twist is zeroed because it's been long time since the last msg arrived..")

    def marker_callback(self, marker):
        # Positions
        if marker.type == Marker.POINTS:
            for particle in self.particles:
                if not self.buttons_manual[particle].isChecked(): # Needed To prevent conflicts in pose updates when applying manual control
                    self.particle_positions[particle] = marker.points[particle]

        # Orientations (TODO: Make sure the logic below is also valid for 3D not just in 2D, later)
        if marker.type == Marker.LINE_LIST: 
            for particle in self.particles:
                if not self.buttons_manual[particle].isChecked(): # Needed To prevent conflicts in pose updates when applying manual control
                    p1 = marker.points[2*particle]
                    p2 = marker.points[2*particle+1]

                    # Calculate the direction vector from p1 to p2
                    direction = -np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

                    # Normalize the direction vector
                    magnitude = np.linalg.norm(direction)
                    direction = direction/magnitude;

                    # Convert direction vector to quaternion
                    # Assuming direction vector is along z axis
                    z_axis = np.array([0, 0, 1])

                    # cross and dot products
                    v = np.cross(z_axis, direction)
                    c = np.dot(z_axis, direction)

                    quaternion = Quaternion()

                    # If the vectors are not aligned
                    if np.any(v):
                        # Compute the quaternion
                        w = np.sqrt((1 + c) / 2)
                        quaternion.x = v[0] / (2 * w)
                        quaternion.y = v[1] / (2 * w)
                        quaternion.z = v[2] / (2 * w)
                        quaternion.w = w    
                    else:
                        quaternion.x = 0
                        quaternion.y = 0
                        quaternion.z = 0
                        quaternion.w = 1

                    self.particle_orientations[particle] = quaternion


    def initialize_leader_position(self):
        # Initialize to zero vector
        position = Point()
        position.x = 0
        position.y = 0
        position.z = 0

        # Initialize to identity rotation
        quaternion = Quaternion()
        quaternion.x = 0
        quaternion.y = 0
        quaternion.z = 0
        quaternion.w = 1

        self.particle_positions["leader"] = position
        self.particle_orientations["leader"] = quaternion

    def manual_control_button_pressed_cb(self, particle):
        if self.buttons_manual[particle].isChecked():
            # Button is currently pressed, no need to set it to False
            print(f"Button for manual control of particle {particle} is now pressed.")

            # Do not proceed with the "non-leader" particles until the initial values have been set
            if (particle != "leader") and not ((particle in self.particle_positions) and ("leader" in self.particle_positions)):
                # Make the manual control button unpressed
                self.buttons_manual[particle].setChecked(False)

                rospy.logwarn("Initial pose values of the object particles or the leader is not yet set")
            else:
                if particle in self.binded_particles:
                    # Unbind particle from leader by deleting the relative pose from the dictionary
                    if particle in self.binded_relative_poses:
                        del self.binded_relative_poses[particle]
                    # else:
                    #     rospy.loginfo(f"Key '{particle}' not found in the binded_relative_poses dictionary.")

                    # Make the bind to leader button unpressed
                    self.bind_to_leader_buttons[particle].setChecked(False)

        else:
            # Button is currently not pressed, no need to set it to True
            print(f"Button for manual control of particle {particle} is now NOT pressed.")    

    def get_timestep(self, integrator_name):
        current_time = rospy.Time.now().to_time()
        if integrator_name in self.last_timestep_requests:
            dt = current_time - self.last_timestep_requests[integrator_name]
            self.last_timestep_requests[integrator_name] = current_time
            if dt > MAX_TIMESTEP:
                dt = 0.0
            return dt
        else:
            self.last_timestep_requests[integrator_name] = current_time
            return 0.0

    def check_shutdown(self):
        if rospy.is_shutdown():
            qt_widgets.QApplication.quit()

    

if __name__ == "__main__":
    rospy.init_node('fake_odom_publisher_gui_node', anonymous=False)

    app = qt_widgets.QApplication(sys.argv)

    gui = FakeOdomPublisherGUI()
    gui.show()

    sys.exit(app.exec_())
