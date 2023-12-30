#!/usr/bin/env python3

import sys

import rospy

import numpy as np
import time


from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, Float32MultiArray


from deformable_manipulations_centralized_rope.srv import ControllerSetParticleState, ControllerSetParticleStateResponse
from deformable_manipulations_centralized_rope.srv import ResetParticlePosition, ResetParticlePositionResponse

from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty, EmptyResponse

import cvxpy as cp

import time

class NominalController:
    def __init__(self, Kp=1.0, Kd=0.0, MAX_TIMESTEP = 0.1):
        # PD gains
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)

        self.last_time = None # time.time()

        # Velocity commands will only be considered if they are spaced closer than MAX_TIMESTEP
        self.MAX_TIMESTEP = MAX_TIMESTEP

    def output(self, error, vel):
        # Calculate the output
        output = self.Kp * error - self.Kd * vel

        return output
    
    def get_dt(self):
        current_time = time.time()
        if self.last_time:
            dt = current_time - self.last_time
            self.last_time = current_time
            if dt > self.MAX_TIMESTEP:
                dt = 0.0
                rospy.logwarn("Controller took more time than specified MAX_TIMESTEP duration, resetting to 0.")
            return dt
        else:
            self.last_time = current_time
            return 0.0
        
class VelocityControllerNode:
    def __init__(self):
        # rospy.sleep(2)

        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.initial_values_set = False  # Initialization state variable
        
        self.delta_x = rospy.get_param("/delta_x", 0.0)
        self.delta_y = rospy.get_param("/delta_y", 0.1)
        self.delta_z = rospy.get_param("/delta_z", 0.1)

        # Parameters needed to figure out the max and min allowed distances automatically
        self.dlo_l = rospy.get_param("/dlo_l")
        self.dlo_num_segments = rospy.get_param("/dlo_num_segments")

        self.particles = [] # All particles that is union of controllable and uncontrollable particles
        self.follower_particles = [] # Particles that are controllable by the robots 
        self.binded_particles = [] # particles that are uncontrollable and binded to the leader frame,
        # (e.g human hand held points when the neck joint is the leader)

        self.odom_topic_prefix = None
        self.odom_topic_leader = None
        while not self.particles:
            try:
                self.particles = rospy.get_param("/custom_static_particles")
                self.follower_particles = rospy.get_param("/follower_particles") 
                self.odom_topic_prefix = rospy.get_param("/custom_static_particles_odom_topic_prefix")
                self.odom_topic_leader = rospy.get_param("/odom_topic_leader")
            except:
                rospy.logwarn("No particles obtained from ROS parameters in Controller node.")
                time.sleep(0.5)

        self.binded_particles = list(set(self.particles) - set(self.follower_particles))

        # Find maximum allowed distances between particles (Keys are tuple pairs as particle (id1,id2))
        self.d_maxs = {}
        self.d_mins = {}

        self.find_max_n_min_allowed_distances()

        # SERVICES
        # Create the service server for enabling the centralized controller
        self.enabled = False  # Flag to enable/disable controller
        self.set_enable_controller_server = rospy.Service('~set_enable_controller', SetBool, self.set_enable_controller)

        # Create the service server for dynamically adjusting active particles considered in the controller
        self.enabled_particles = {} # Dict to store which follower particles are enabled in the controller as True/False
        # Initialize each particle in the controller as enabled
        for particle in self.follower_particles:
            self.enabled_particles[particle] = True

        # Create the service server for resetting particle positions in the controller
        self.set_disable_particle_in_controller_server = rospy.Service('~set_disable_particle_in_controller',
                                                                        ControllerSetParticleState,
                                                                        self.set_disable_particle)
        
        # Create the service server for resetting particle positions
        self.reset_positions_server = rospy.Service('~reset_positions', ResetParticlePosition, self.reset_positions)

        
        # Subscriber for dlo points to figure out the current particle positions
        self.particle_positions = {}
        self.particle_orientations = {}

        self.particle_positions_prev = {} # dicts to store the prev. particle positions, 
        self.particle_orientations_prev = {} # will be used to calculate the velocities of the particles

        self.initial_relative_positions = {} 
        self.initial_relative_orientations = {}
        
        self.dlo_points_topic_name    = rospy.get_param("~dlo_points_topic_name", "/dlo_points") 
        self.sub_marker = rospy.Subscriber(self.dlo_points_topic_name, Marker, self.marker_callback, queue_size=10)

        # Subscriber for the leader frame to figure out the current pose of the leader
        self.sub_odom_leader = rospy.Subscriber(self.odom_topic_leader, Odometry, self.odom_leader_cb, queue_size=1)

        # Subscriber for the current minimim distance to the obstacle TODO: LATER SOMEHOW CONSIDER MULTIPLE OBSTACLES
        

        # Subscribers to the minimum distances with perturbations
        self.min_distance_topic_prefix    = rospy.get_param("~min_distance_topic_prefix",   "/distance_to_obj") 
        self.min_distance_topic_name    = self.min_distance_topic_prefix
        self.min_distance_dx_topic_name = self.min_distance_topic_name + "_" + str(self.id) + "_x" 
        self.min_distance_dy_topic_name = self.min_distance_topic_name + "_" + str(self.id) + "_y" 
        self.min_distance_dz_topic_name = self.min_distance_topic_name + "_" + str(self.id) + "_z" 

        self.min_distance    = None
        self.min_distance_dx = None
        self.min_distance_dy = None
        self.min_distance_dz = None

        self.sub_min_distance    = rospy.Subscriber(self.min_distance_topic_name,    Float32, self.min_distance_callback,    queue_size=10)
        self.sub_min_distance_dx = rospy.Subscriber(self.min_distance_dx_topic_name, Float32, self.min_distance_dx_callback, queue_size=10)
        self.sub_min_distance_dy = rospy.Subscriber(self.min_distance_dy_topic_name, Float32, self.min_distance_dy_callback, queue_size=10)
        self.sub_min_distance_dz = rospy.Subscriber(self.min_distance_dz_topic_name, Float32, self.min_distance_dz_callback, queue_size=10)


        # Controller gains
        self.kp = np.array(rospy.get_param("~kp", [10.0,10.0,10.0]))
        self.kd = np.array(rospy.get_param("~kd", [0.0,0.0,0.0]))
        self.control_output_safe = np.zeros(3) # initialization for the velocity command

        # Create the nominal controller
        self.controller = NominalController(self.kp, self.kd, self.pub_rate_odom*2.0)

        # Parameters for the auxiliary control (NOTE: Work in progress, keep it disabled )
        self.aux_enabled = rospy.get_param("~aux_enabled", False) 
        self.aux_bias_direction = rospy.get_param("~aux_bias_direction", "up")  # "up" or "down", anything else removes bias and follows the paper convention
        
        # create a publisher for the controlled particle
        self.odom_publisher = rospy.Publisher(self.odom_topic_prefix + str(self.id), Odometry, queue_size=1)
        
        # Create the necessary publishers for information topics, 
        # topic names are prefixed with "controller_info_" + str(self.id) 
        # followed by a description of the information topic.
        self.info_error_norm_publisher       = rospy.Publisher("controller_info_"+str(self.id)+"_error_norm",                Float32,           queue_size=1)
        self.info_target_pose_publisher      = rospy.Publisher("controller_info_"+str(self.id)+"_target_pose",               Marker,            queue_size=1)
        self.info_J_publisher                = rospy.Publisher("controller_info_"+str(self.id)+"_J_matrix",                  Float32MultiArray, queue_size=1)
        self.info_h_collision_publisher      = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_collision",      Float32,           queue_size=1)
        self.info_h_overstretching_publisher = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_overstretching", Float32MultiArray, queue_size=1)
        self.info_h_too_close_publisher      = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_too_close",      Float32MultiArray, queue_size=1)

        # Start the control
        self.odom_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)

        

    def find_max_n_min_allowed_distances(self):
        """Find max and min allowed distances between particles"""
        l_i = self.dlo_l / self.dlo_num_segments  # Each particle segment length
        delta_norm = np.linalg.norm([self.delta_x, self.delta_y, self.delta_z])

        # Using a set to store visited particle pairs to avoid duplicates
        visited_pairs = set()

        for particle in self.follower_particles:
            for other_particle in self.particles:
                if particle != other_particle:
                    # Always treating the particle with the smaller identifier as the first
                    particle_pair = tuple(sorted([particle, other_particle]))

                    # Skip if this pair has already been processed
                    if particle_pair in visited_pairs:
                        continue

                    l_0_ij = l_i * abs(particle - other_particle)  # Default distance between particle i and j
                    d_max = l_0_ij - l_i - 1.0 * delta_norm
                    d_min = l_i + 1.0 * delta_norm

                    self.d_maxs[particle_pair] = d_max
                    self.d_mins[particle_pair] = d_min
                    visited_pairs.add(particle_pair)

        # For Info print all min and max distances between controlled particles (self.follower_particles) and custom static particles (self.particles)
        self.print_allowed_distances()

    def print_allowed_distances(self):
        """Prints all min and max distances between controlled and static particles."""
        for particle_pair, d_max in self.d_maxs.items(): # Assuming self.d_maxs and self.d_mins have the same keys and maintened accordingly
            if all(particle in self.follower_particles for particle in particle_pair):
                d_min = self.d_mins[particle_pair]
                rospy.loginfo(f"Particles {particle_pair}: Max distance = {d_max}, Min distance = {d_min}")

    def get_max_allowed_distance(self, particle1, particle2):
        """Get the max distance between two particles, regardless of order."""
        particle_pair = tuple(sorted([particle1, particle2]))
        if particle_pair in self.d_maxs:
            return self.d_maxs[particle_pair]
        else:
            rospy.logerr(f"Max distance not found for particle pair: {particle_pair}")
            return None

    def get_min_allowed_distance(self, particle1, particle2):
        """Get the min distance between two particles, regardless of order."""
        particle_pair = tuple(sorted([particle1, particle2]))
        if particle_pair in self.d_mins:
            return self.d_mins[particle_pair]
        else:
            rospy.logerr(f"Min distance not found for particle pair: {particle_pair}")
            return None

    # SERVICES
    def set_enable_controller(self, request):
        self.enabled = request.data
        return SetBoolResponse(True, 'Successfully set enabled state to {}'.format(self.enabled))

    def set_disable_particle(self, request):
        particle_id = request.particle_id
        enable_state = request.enable

        # Check if the particle ID exists in the controller
        if particle_id in self.enabled_particles:
            self.enabled_particles[particle_id] = enable_state
            return ControllerSetParticleStateResponse(success=True)
        else:
            rospy.logerr(f"Particle ID {particle_id} not found.")
            return ControllerSetParticleStateResponse(success=False)

    def reset_positions(self, request):
        particle = request.particle_id

        if (particle in self.particle_positions) and ("leader" in self.particle_positions):
            # Reset the initial relative positions and orientations
            initial_relative_position = Point()
            initial_relative_position.x = self.particle_positions[particle].x - self.particle_positions["leader"].x
            initial_relative_position.y = self.particle_positions[particle].y - self.particle_positions["leader"].y
            initial_relative_position.z = self.particle_positions[particle].z - self.particle_positions["leader"].z

            self.initial_relative_positions[particle] = initial_relative_position
            
            leader_orientation_inverse = self.inverse_quaternion(self.particle_orientations["leader"])
            initial_relative_orientation = self.multiply_quaternions(leader_orientation_inverse, self.particle_orientations[self.id])

            self.initial_relative_orientations[particle] = initial_relative_orientation

            return ResetParticlePositionResponse()
        else:
            rospy.logwarn(f"Leader position or {particle} position is not set yet.")

    # SUBSCRIBER CALLBACKS
    def marker_callback(self, marker):
        # Positions
        if marker.type == Marker.POINTS:
            for particle in self.particles:
                # Update the particle pose only if that particle is not currently actively controlled at that moment.
                if not (self.enabled and self.enabled_particles[particle]): # Needed To prevent conflicts in pose updates when applying control from this controller. NOTE: This is valid only when we can assume that the commanded odom pose direclty affects the detected fabric state without delay. 
                    # Update the record of the previous particle position to find the velocity later with numerical derivation
                    if particle in self.particle_positions: 
                        self.particle_positions_prev[particle] = self.particle_positions[particle]
                    # Update the current particle position
                    self.particle_positions[particle] = marker.points[particle]

                # Calculate initial relative positions
                if (not particle in self.initial_relative_positions):
                    if (particle in self.particle_positions) and ("leader" in self.particle_positions):
                        initial_relative_position = Point()
                        initial_relative_position.x = self.particle_positions[particle].x - self.particle_positions["leader"].x
                        initial_relative_position.y = self.particle_positions[particle].y - self.particle_positions["leader"].y
                        initial_relative_position.z = self.particle_positions[particle].z - self.particle_positions["leader"].z

                        self.initial_relative_positions[particle] = initial_relative_position
                    else:
                        rospy.logwarn(f"Leader position or {particle} position is not set yet.")

        # Orientations (TODO: Make sure the logic below is also valid for 3D not just in 2D, later)
        if marker.type == Marker.LINE_LIST:
            for particle in self.particles:
                # Update the particle pose only if that particle is not currently actively controlled at that moment.
                if not (self.enabled and self.enabled_particles[particle]): # Needed To prevent conflicts in pose updates when applying control from this controller. NOTE: This is valid only when we can assume that the commanded odom pose direclty affects the detected fabric state without delay. 
                    # Process to get orientation from points is similar to fake_odom_publisher_gui.py script
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

                    # Update the record of the previous particle orientation to find the angular velocity later with numerical derivation
                    if particle in self.particle_orientations:
                        self.particle_orientations_prev[particle] = self.particle_orientations[particle] 
                    # Update the current particle orientation
                    self.particle_orientations[particle] = quaternion
                
                # Calculate initial relative orientations
                if (not particle in self.initial_relative_orientations):
                    if (particle in self.particle_orientations) and ("leader" in self.particle_orientations):
    
                        leader_orientation_inverse = self.inverse_quaternion(self.particle_orientations["leader"])
                        self.initial_relative_orientations[particle] = self.multiply_quaternions(leader_orientation_inverse, self.particle_orientations[particle])
                    else:
                        rospy.logwarn(f"Leader orientation or {particle} orientation is not set yet.")

        # After all initial relative positions and orientations have been calculated, set the initialization state variable to True
        self.initial_values_set = True

    def odom_leader_cb(self, odom):
        """ Update leader's current pose"""
        particle = "leader"

        # Update the record of the previous particle position to find the velocity later with numerical derivation
        if particle in self.particle_positions: 
            self.particle_positions_prev[particle] = self.particle_positions[particle]
        # Update the current particle position
        self.particle_positions[particle] = odom.pose.pose.position # Point()

        # Update the record of the previous particle position to find the angular velocity later with numerical derivation
        if particle in self.particle_orientations:
            self.particle_orientations_prev[particle] = self.particle_orientations[particle] 
        # Update the current particle orientation
        self.particle_orientations[particle] = odom.pose.pose.orientation # Quaternion()


    def odom_pub_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            # Do not proceed until the initial values have been set
            if (not self.initial_values_set) or (not self.is_min_distances_set()):
                return
            
            target_pose = self.calculate_target_pose()

            # target_pose is geometry_msgs.msg.Pose, publish it for information to visualize in RVIZ 
            # as a line segment and a sphere in the end, and starting from the leader_position = self.particle_positions["leader"]
            leader_position = self.particle_positions["leader"]
            self.publish_arrow_marker(leader_position, target_pose)

            # Get current position and orientation
            current_pose = Pose()
            current_pose.position = self.particle_positions[self.id]
            current_pose.orientation = self.particle_orientations[self.id]

            # Calculate (relative) error
            error = self.calculate_error(current_pose, target_pose)

            # error is 3D np.array, publish its norm for information
            error_norm = np.linalg.norm(error)
            self.info_error_norm_publisher.publish(Float32(data=error_norm))

            # Update the controller terms with the current error and the last executed command
            # Get control output from the controller
            control_output = self.controller.output(error,self.control_output_safe) # nominal
            # control_output = np.zeros(3) # disable nominal controller to test only the safe controller

            # init_t = time.time()

            # Get safe control output using control barrier functions
            if not self.aux_enabled:
                self.control_output_safe = self.calculate_safe_control_output(control_output, 1./self.pub_rate_odom) # safe
            # else: # TODO
            #     self.control_output_safe = self.calculate_safe_control_output_w_aux(control_output, 1./self.pub_rate_odom, error_norm) # safe
            
            # rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.")

            if self.control_output_safe is not None:
                # Scale down the calculated output if its norm is higher than the specified norm max_u
                self.control_output_safe = self.scale_down_vector(self.control_output_safe, max_u=0.15)

                print("u + u_aux*r: ", str(self.control_output_safe))

                # print(np.linalg.norm(self.control_output_safe - control_output))

                # print(self.control_output_safe - control_output)
                # rospy.logwarn("control_output:" + str(control_output))
                # rospy.logwarn("control_output_safe: "+ str(self.control_output_safe))

                # self.control_output_safe = control_output

                # Prepare Odometry message
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = "map"

                dt_check = self.controller.get_dt() # 
                dt = 1./self.pub_rate_odom

                # Control output is the new position
                odom.pose.pose.position.x =  current_pose.position.x + self.control_output_safe[0]*dt
                odom.pose.pose.position.y =  current_pose.position.y + self.control_output_safe[1]*dt
                odom.pose.pose.position.z =  current_pose.position.z + self.control_output_safe[2]*dt

                odom.pose.pose.orientation = target_pose.orientation # 

                self.odom_publisher.publish(odom)
            else:
                self.control_output_safe = np.zeros(3)


    def calculate_target_pose(self,particle):
        # Calculate the target position and orientation relative to the leader
        target_pose = Pose()
        leader_position = self.particle_positions["leader"]
        leader_orientation = self.particle_orientations["leader"]

        initial_relative_position = self.initial_relative_positions[particle]

        target_pose.position.x = leader_position.x + initial_relative_position.x
        target_pose.position.y = leader_position.y + initial_relative_position.y
        target_pose.position.z = leader_position.z + initial_relative_position.z

        target_pose.orientation = self.multiply_quaternions(leader_orientation, self.initial_relative_orientations[particle]) # TODO

        return target_pose
    
    def calculate_error(self, current_pose, target_pose):
        
        err_x = target_pose.position.x - current_pose.position.x
        err_y = target_pose.position.y - current_pose.position.y
        err_z = target_pose.position.z - current_pose.position.z

        # calculate error for orientations

        return np.array([err_x,err_y,err_z])
    
    def calculate_current_relative_position_vec(self,particle1,particle2):
        p_x = self.particle_positions[particle1].x - self.particle_positions[particle2].x
        p_y = self.particle_positions[particle1].y - self.particle_positions[particle2].y
        p_z = self.particle_positions[particle1].z - self.particle_positions[particle2].z

        return np.array([p_x,p_y,p_z])



    

    
    
    
    def min_distance_callback(self, distance):
        t = 0.5
        if self.min_distance is None:
            self.min_distance = distance.data
        else:
            self.min_distance = t*self.min_distance + (1-t)*distance.data

    def min_distance_dx_callback(self, distance):
        t = 0.5
        if self.min_distance_dx is None:
            self.min_distance_dx = distance.data
        else:
            self.min_distance_dx = t*self.min_distance_dx + (1-t)*distance.data

    def min_distance_dy_callback(self, distance):
        t = 0.5
        if self.min_distance_dy is None:
            self.min_distance_dy = distance.data
        else:
            self.min_distance_dy = t*self.min_distance_dy + (1-t)*distance.data

    def min_distance_dz_callback(self, distance):
        t = 0.5
        if self.min_distance_dz is None:
            self.min_distance_dz = distance.data
        else:
            self.min_distance_dz = t*self.min_distance_dz + (1-t)*distance.data

    def is_min_distances_set(self):
        return ((self.min_distance    is not None) and
                (self.min_distance_dx is not None) and  
                (self.min_distance_dy is not None) and 
                (self.min_distance_dz is not None))



    def calculate_safe_control_output(self, nominal_u, dt):
        if not self.is_min_distances_set():
            return

        # ---------------------------------------------------
        # Define optimization variables
        u = cp.Variable(3)

        # Define weights for each control input
        weights = np.array([1.0, 1.0, 0.1])  # Less weight on z control input
        
        # Define cost function with weights
        cost = cp.sum_squares(cp.multiply(weights, u - nominal_u)) / 2.0

        # Initialize the constraints
        constraints = []
        # ---------------------------------------------------

        # ---------------------------------------------------
        # DEFINE COLLISION AVOIDANCE CONTROL BARRRIER CONSTRAINT
        d_offset = 0.05 # offset distance from the obstacles
        h = self.min_distance - d_offset  # Control Barrier Function (CBF)
        alpha_h = self.alpha_collision_avoidance(h)

        # publish h that is the distance to collision for information
        # print("h distance to collision: ",str(h))
        self.info_h_collision_publisher.publish(Float32(data=h))

        Jx = ((self.min_distance_dx - self.min_distance) / self.delta_x) if self.delta_x != 0 else 0
        Jy = ((self.min_distance_dy - self.min_distance) / self.delta_y) if self.delta_y != 0 else 0
        Jz = ((self.min_distance_dz - self.min_distance) / self.delta_z) if self.delta_z != 0 else 0

        J = np.array([[Jx,Jy,Jz]]) # 1x3

        # publish J for information
        # print("J",str(J))
        J_msg = Float32MultiArray(data=np.ravel(J))
        self.info_J_publisher.publish(J_msg)

        J_tolerance = 0.05
        if not np.all(np.abs(J) < J_tolerance):
            # Add collision avoidance to the constraints
            constraints += [J @ u >= -alpha_h]
        else:
            rospy.loginfo("Follower: "+str(self.id)+", ignored J")
        # ---------------------------------------------------

        # ---------------------------------------------------
        h_overstretchs = []
        h_too_closes = []
        for particle in self.particles:
            if particle != self.id:
                # DEFINE OVERSTRETCHING AVOIDANCE CONTROL BARRRIER CONSTRAINT
                P_cur = self.calculate_current_relative_position_vec(self.id, particle)
                P_cur_norm = np.linalg.norm(P_cur) 

                min_norm_tolerance = 1.0e-9
                if P_cur_norm >= min_norm_tolerance:
                    P_cur_unit = P_cur/P_cur_norm

                    d_max = self.d_maxs[particle] # meters # maximum distance between the human held point and the robot controlled point

                    h_overstretch = d_max - P_cur_norm  # Control Barrier Function (CBF)
                    alpha_h = self.alpha_overstretch_avoidance(h_overstretch)

                    # print("h distance to overstretching: ",str(h_overstretch))
                    h_overstretchs.append(h_overstretch)

                    # calculate the velocity of the other particle
                    v_other_x = (self.particle_positions[particle].x - self.particle_positions_prev[particle].x)/dt if dt >= 1.0e-5 else 0.0
                    v_other_y = (self.particle_positions[particle].y - self.particle_positions_prev[particle].y)/dt if dt >= 1.0e-5 else 0.0
                    v_other_z = (self.particle_positions[particle].z - self.particle_positions_prev[particle].z)/dt if dt >= 1.0e-5 else 0.0
                    v_other = np.array([v_other_x,v_other_y,v_other_z])
                    # rospy.loginfo("Follower: "+str(self.id)+", v_other: "+str(v_other)+ " (for particle: " + str(particle) + ")")

                    # Add overstretching avoidance to the constraints
                    constraints += [-P_cur_unit.T @ u + P_cur_unit.T @ v_other >= -alpha_h]
                # ---------------------------------------------------

                    # ---------------------------------------------------
                    # DEFINE GETTING TOO CLOSE AVOIDANCE CONTROL BARRRIER CONSTRAINT
                    d_min = self.d_mins[particle] # meters # maximum distance between the human held point and the robot controlled point

                    h_too_close = P_cur_norm - d_min   # Control Barrier Function (CBF)
                    alpha_h = self.alpha_too_close_avoidance(h_too_close)

                    # print("h distance to agent getting too close: ",str(h))
                    h_too_closes.append(h_too_close)
                    
                    # Add getting too close avoidance to the constraints
                    constraints += [P_cur_unit @ u - P_cur_unit.T @ v_other >= -alpha_h]
                    # ---------------------------------------------------
        
        # publish h that is the distance to overstretching for information
        self.info_h_overstretching_publisher.publish(Float32MultiArray(data=h_overstretchs))

        # publish h that is the distance to agent getting too close for information
        self.info_h_too_close_publisher.publish(Float32MultiArray(data=h_too_closes))


        # Add also limit to the feasible u
        u_max = 0.15
        constraints += [cp.norm(u,'inf') <= u_max] # If inf-norm used, the constraint is linear, use OSQP solver (still QP)
        # constraints += [cp.norm(u,2)     <= u_max] # If 2-norm is used, the constraint is Quadratic, use CLARABEL or ECOS solver (Not QP anymore, a conic solver is needed)

        # ---------------------------------------------------
        # Define and solve problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # # For warm-start
        # if hasattr(self, 'prev_optimal_u'):
        #     u.value = self.prev_optimal_u

        init_t = time.time() # For timing
        try:
            # problem.solve() # Selects automatically
            problem.solve(solver=cp.CLARABEL) # ~8ms 
            # problem.solve(solver=cp.CVXOPT) # ~10ms (warm start capable)
            # problem.solve(solver=cp.ECOS) # ~8ms
            # problem.solve(solver=cp.ECOS_BB) # ~8ms
            # problem.solve(solver=cp.GLOP) # NOT SUITABLE
            # problem.solve(solver=cp.GLPK) # NOT SUITABLE
            # problem.solve(solver=cp.GUROBI) # ~8ms
            # problem.solve(solver=cp.MOSEK) # Encountered unexpected exception importing solver CBC
            # problem.solve(solver=cp.OSQP) # ~7ms (default) (warm start capable)
            # problem.solve(solver=cp.PDLP) # NOT SUITABLE
            # problem.solve(solver=cp.SCIPY) # NOT SUITABLE 
            # problem.solve(solver=cp.SCS) # ~8ms (warm start capable)

        except cp.error.SolverError as e:
            rospy.logwarn("Follower: "+str(self.id)+",Could not solve the problem: {}".format(e))
            # self.prev_optimal_u = None # to reset warm-start
            return None

        rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.") # For timing

        # Print the available qp solvers
        # e.g. ['CLARABEL', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLOP', 'GLPK', 'GLPK_MI', 'GUROBI', 'MOSEK', 'OSQP', 'PDLP', 'SCIPY', 'SCS']
        rospy.loginfo_once(str(cp.installed_solvers()))
        # Print the solver used to solve the problem (once)
        rospy.loginfo_once("Solver used: " + str(problem.solver_stats.solver_name))

        # check if problem is infeasible or unbounded
        if problem.status in ["infeasible", "unbounded"]:
            rospy.logwarn("Follower: "+str(self.id)+", The problem is {}.".format(problem.status))
            # self.prev_optimal_u = None # to reset warm-start
            return None
        
        # # For warm-start in the next iteration
        # self.prev_optimal_u = u.value
        
        # Return optimal u
        return u.value
  

    # CONTROLLER RELATED HELPER FUNCTIONS    
    def alpha_collision_avoidance(self,h):
        # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
        # Piecewise Linear function is used

        c1 = 0.4 # 0.8 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
        c2 = 5.0 # 3.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
        alpha_h = c1*(h) if (h) >= 0 else c2*(h)
        return alpha_h
    
    def alpha_overstretch_avoidance(self,h):
        # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
        # Piecewise Linear function is used
        
        c1 = 0.4 # 2.0 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
        c2 = 2.0 # 2.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
        alpha_h = c1*(h) if (h) >= 0 else c2*(h)
        return alpha_h
    
    def alpha_too_close_avoidance(self,h):
        # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
        # Piecewise Linear function is used
        
        c1 = 0.4 # 2.0 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
        c2 = 2.0 # 2.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
        alpha_h = c1*(h) if (h) >= 0 else c2*(h)
        return alpha_h

    def scale_down_vector(self, u, max_u=0.005):
        norm_u = np.linalg.norm(u)

        if norm_u > max_u:
            u = u / norm_u * max_u

        return u


    def publish_arrow_marker(self, leader_position, target_pose):
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
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        # Set the pose (position and orientation) for the marker
        marker.pose.orientation.w = 1.0  # Orientation (quaternion)
        
        # Set the start and end points of the arrow
        marker.points = []
        start_point = leader_position  # Should be a Point message
        end_point = target_pose.position  # Assuming target_pose is a Pose message
        marker.points.append(start_point)
        marker.points.append(end_point)
        
        # Publish the marker
        self.info_target_pose_publisher.publish(marker)

    # SOME HELPER QUATERNION OPERATIONS
    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions.
        """
        a1, b1, c1, d1 = q1.w, q1.x, q1.y, q1.z
        a2, b2, c2, d2 = q2.w, q2.x, q2.y, q2.z

        w = a1*a2 - b1*b2 - c1*c2 - d1*d2
        x = a1*b2 + b1*a2 + c1*d2 - d1*c2
        y = a1*c2 - b1*d2 + c1*a2 + d1*b2
        z = a1*d2 + b1*c2 - c1*b2 + d1*a2

        return Quaternion(x, y, z, w)

    def inverse_quaternion(self, q):
        """
        Calculate the inverse of a quaternion.
        """
        a, b, c, d = q.w, q.x, q.y, q.z
        norm = a**2 + b**2 + c**2 + d**2

        return Quaternion(-b/norm, -c/norm, -d/norm, a/norm)

if __name__ == "__main__":
    rospy.init_node('dlo_velocity_controller_node', anonymous=False)

    node = VelocityControllerNode()

    rospy.spin()
