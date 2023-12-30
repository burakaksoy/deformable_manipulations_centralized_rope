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
        
        self.delta_x = rospy.get_param("/delta_x", 0.1)
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
        for particle in self.particles:
            if particle in self.binded_particles:
                self.enabled_particles[particle] = False
            else: # ie. if particle in self.follower_particles:
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

        self.initial_relative_positions = {} # Point(), relative position vector from leader to the particle in LEADER frame
        self.initial_relative_orientations = {} # Quaternion(), rotation from leader frame to the particle frame 
        # (note that if the particle does not have orientation it is identity)
        
        self.dlo_points_topic_name    = rospy.get_param("~dlo_points_topic_name", "/dlo_points") 
        self.sub_marker = rospy.Subscriber(self.dlo_points_topic_name, Marker, self.marker_callback, queue_size=10)

        # Subscriber for the leader frame to figure out the current pose of the leader
        self.sub_odom_leader = rospy.Subscriber(self.odom_topic_leader, Odometry, self.odom_leader_cb, queue_size=10)

        # Subscriber for the current minimim distance to the obstacle TODO: LATER SOMEHOW CONSIDER MULTIPLE OBSTACLES
        self.min_distance = None

        self.min_distance_topic_prefix    = rospy.get_param("~min_distance_topic_prefix",   "/distance_to_obj") 
        self.min_distance_topic_name    = self.min_distance_topic_prefix
        self.sub_min_distance    = rospy.Subscriber(self.min_distance_topic_name,    Float32, self.min_distance_callback,    queue_size=10)

        # Subscribers to the minimum distances with perturbations      
        ## We create 3 subscriber for each follower particle
        ## so each of the following dictionaries will have elements as many as the follower particles  
        self.subs_min_distance_dx = {}
        self.subs_min_distance_dy = {}
        self.subs_min_distance_dz = {}

        ## Dictionaries to store the minimum distances of each follower particle
        self.min_distances_dx = {}
        self.min_distances_dy = {}
        self.min_distances_dz = {}

        self.min_distances_set_particles = [] # For bookkeeping of which follower particles are obtained their all perturbed minimum distance readings at least once.

        ## Create the subscribers to minimum distances with perturbations
        for particle in self.follower_particles:
            # Prepare the topic names of that particle
            min_distance_dx_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_x" 
            min_distance_dy_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_y" 
            min_distance_dz_topic_name = self.min_distance_topic_name + "_" + str(particle) + "_z" 

            # Create the subscribers (also takes the particle argument)
            self.subs_min_distance_dx[particle] = rospy.Subscriber(min_distance_dx_topic_name, Float32, self.min_distance_dx_callback, particle, queue_size=10)
            self.subs_min_distance_dy[particle] = rospy.Subscriber(min_distance_dy_topic_name, Float32, self.min_distance_dy_callback, particle, queue_size=10)
            self.subs_min_distance_dz[particle] = rospy.Subscriber(min_distance_dz_topic_name, Float32, self.min_distance_dz_callback, particle, queue_size=10)
        
        # Create an (odom) Publisher for each follower particle as a control output to the particles
        self.odom_publishers = {}

        for particle in self.follower_particles:
            self.odom_publishers[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle), Odometry, queue_size=10)

        # Finally, create the (centralized) controller that will publish odom to each follower particle properly

        # Controller gains (assumed to be common for all follower particles)
        self.kp = np.array(rospy.get_param("~kp", [1.0,1.0,1.0]))
        self.kd = np.array(rospy.get_param("~kd", [0.0,0.0,0.0]))
        
        # Create the nominal controllers and control outputs
        self.target_poses = {}
        self.nominal_controllers = {}
        self.control_outputs_safe = {} 

        for particle in self.follower_particles:
            self.target_poses[particle] = Pose()
            self.nominal_controllers[particle] = NominalController(self.kp, self.kd, self.pub_rate_odom*2.0)
            self.control_outputs_safe[particle] = np.zeros(3) # initialization for the velocity command

        # Parameters for the auxiliary control (NOTE: Work in progress, keep it disabled )
        self.aux_enabled = rospy.get_param("~aux_enabled", False) 
        self.aux_bias_direction = rospy.get_param("~aux_bias_direction", "up")  # "up" or "down", anything else removes bias and follows the paper convention
        
        
        # # Create the necessary publishers for information topics, 
        # # topic names are prefixed with "controller_info_" + str(particle) 
        # # followed by a description of the information topic.
        self.info_target_position_publishers = {} # To publish arrow from leader to the particle
        self.info_pos_error_norm_publishers = {} # To publish the norm of position errors

        for particle in self.follower_particles:
            self.info_target_position_publishers[particle]      = rospy.Publisher("controller_info_"+str(particle)+"_target_pose",               Marker,            queue_size=5)
            self.info_pos_error_norm_publishers[particle]       = rospy.Publisher("controller_info_"+str(particle)+"_pos_error_norm",            Float32,           queue_size=1)
        # self.info_J_publisher                = rospy.Publisher("controller_info_"+str(self.id)+"_J_matrix",                  Float32MultiArray, queue_size=1)
        # self.info_h_collision_publisher      = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_collision",      Float32,           queue_size=1)
        # self.info_h_overstretching_publisher = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_overstretching", Float32MultiArray, queue_size=1)
        # self.info_h_too_close_publisher      = rospy.Publisher("controller_info_"+str(self.id)+"_h_distance_too_close",      Float32MultiArray, queue_size=1)

        # Start the control
        self.calculate_control_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.calculate_control_outputs_timer_callback)
        self.odom_pub_timer          = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)

        

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

        rospy.loginfo("[Set Disable Particle Server] For Particle: "+str(particle_id)+", request: "+str(enable_state))

        # Check if the particle ID exists in the controller
        if particle_id in self.enabled_particles:
            self.enabled_particles[particle_id] = enable_state
            return ControllerSetParticleStateResponse(success=True)
        else:
            rospy.logerr(f"Particle ID {particle_id} not found.")
            return ControllerSetParticleStateResponse(success=False)

    def reset_positions(self, request):
        particle = request.particle_id

        if ( (particle in self.particle_positions) and ("leader" in self.particle_positions) and \
        (particle in self.particle_orientations) and ("leader" in self.particle_orientations) ):
            ##  Reset the initial relative position
            # Current Pose of the leader in world frame
            pos_leader = self.particle_positions["leader"] # Point() msg of ROS geometry_msgs
            ori_leader = self.particle_orientations["leader"] # Quaternion() msg of ROS geometry_msgs

            # Current Pose of the controlled particle in world frame
            pos_controlled_world = self.particle_positions[particle] # Point() msg of ROS geometry_msgs
            # ori_controlled_world = self.particle_orientations[particle] # Quaternion() msg of ROS geometry_msgs

            # Convert quaternion of leader orientation to rotation matrix from world frame
            rot_mat_leader = self.quaternion_to_matrix(ori_leader)

            # Calculate the relative position of the controlled particle in the leader's frame
            relative_pos = np.dot(rot_mat_leader.T, np.array([pos_controlled_world.x - pos_leader.x, pos_controlled_world.y - pos_leader.y, pos_controlled_world.z - pos_leader.z]))
            
            self.initial_relative_positions[particle] = Point(*relative_pos)
            
            ##  Reset initial relative orientation
            leader_orientation_inverse = self.inverse_quaternion(self.particle_orientations["leader"])
            self.initial_relative_orientations[particle] = self.multiply_quaternions(leader_orientation_inverse, self.particle_orientations[particle])

            return ResetParticlePositionResponse()
        else:
            rospy.logwarn(f"Leader position or Particle {particle} pose is not set yet.")

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
                    if (particle in self.particle_positions) and ("leader" in self.particle_positions) and ("leader" in self.particle_orientations):

                        # Current Pose of the leader in world frame
                        pos_leader = self.particle_positions["leader"] # Point() msg of ROS geometry_msgs
                        ori_leader = self.particle_orientations["leader"] # Quaternion() msg of ROS geometry_msgs

                        # Current Pose of the controlled particle in world frame
                        pos_controlled_world = self.particle_positions[particle] # Point() msg of ROS geometry_msgs
                        # ori_controlled_world = self.particle_orientations[particle] # Quaternion() msg of ROS geometry_msgs

                        # Convert quaternion of leader orientation to rotation matrix from world frame
                        rot_mat_leader = self.quaternion_to_matrix(ori_leader)

                        # Calculate the relative position of the controlled particle in the leader's frame
                        relative_pos = np.dot(rot_mat_leader.T, np.array([pos_controlled_world.x - pos_leader.x, pos_controlled_world.y - pos_leader.y, pos_controlled_world.z - pos_leader.z]))
                        
                        self.initial_relative_positions[particle] = Point(*relative_pos)
                    else:
                        rospy.logwarn(f"Leader position/orientation or Particle {particle} position is not set yet.")
                        time.sleep(1.0)

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
                        rospy.logwarn(f"Leader orientation or Particle {particle} orientation is not set yet.")
                        time.sleep(1.0)

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

        # Update the record of the previous particle orientation to find the angular velocity later with numerical derivation
        if particle in self.particle_orientations:
            self.particle_orientations_prev[particle] = self.particle_orientations[particle] 
        # Update the current particle orientation
        self.particle_orientations[particle] = odom.pose.pose.orientation # Quaternion()

    def min_distance_callback(self, distance):
        t = 0.5
        if self.min_distance is None:
            self.min_distance = distance.data
        else:
            self.min_distance = t*self.min_distance + (1-t)*distance.data

    def min_distance_dx_callback(self, distance, particle):
        t = 0.5
        if not particle in self.min_distances_dx:
            self.min_distances_dx[particle] = distance.data
        else:
            self.min_distances_dx[particle] = t*self.min_distances_dx[particle] + (1-t)*distance.data

    def min_distance_dy_callback(self, distance, particle):
        t = 0.5
        if not particle in self.min_distances_dy:
            self.min_distances_dy[particle] = distance.data
        else:
            self.min_distances_dy[particle] = t*self.min_distances_dy[particle] + (1-t)*distance.data

    def min_distance_dz_callback(self, distance, particle):
        t = 0.5
        if not particle in self.min_distances_dz:
            self.min_distances_dz[particle] = distance.data
        else:
            self.min_distances_dz[particle] = t*self.min_distances_dz[particle] + (1-t)*distance.data

    def is_min_distance_set_for_particle(self,particle):
        """
        Checks if the minimum distance parameters (dx, dy, dz) are set for a specified particle.
        This function determines if the follower particle has obtained minimum distances along the x, y, and z axes at least once.
        If these distances are set, the particle is added to the 'min_distances_set_particles' list,
        and the function returns True if the 'min_distance' attribute is not None. Otherwise, it returns False.
        """
        if particle in self.min_distances_set_particles:
            return (self.min_distance is not None)
        else:
            check = ((particle in self.min_distances_dx) and  
                    (particle in self.min_distances_dy) and 
                    (particle in self.min_distances_dz))
            if check:
                self.min_distances_set_particles.append(particle)
                return (self.min_distance is not None)
            else:
                return False

    def calculate_control_outputs_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            for particle in self.follower_particles:
                # Do not proceed until the initial values have been set
                if ((not self.enabled_particles[particle]) or \
                    (not (particle in self.particle_positions)) or \
                    (not ("leader" in self.particle_positions)) or \
                    (not self.is_min_distance_set_for_particle(particle))):
                    continue
            
                self.target_poses[particle] = self.calculate_target_pose(particle)

                # target_pose is geometry_msgs.msg.Pose, publish it for information to visualize in RVIZ 
                # as a line segment and a sphere in the end, and starting from the leader_position = self.particle_positions["leader"]
                leader_position = self.particle_positions["leader"]
                self.publish_arrow_marker(leader_position, self.target_poses[particle], particle)

                # Get current position and orientation
                current_pose = Pose()
                current_pose.position = self.particle_positions[particle]
                current_pose.orientation = self.particle_orientations[particle]

                # Calculate (relative) error
                pos_error, ori_error = self.calculate_error(current_pose, self.target_poses[particle]) # pos_error and ori_error are 3D np.arrays

                # error is 3D np.array, publish its norm for information
                pos_error_norm = np.linalg.norm(pos_error)
                self.info_pos_error_norm_publishers[particle].publish(Float32(data=pos_error_norm))

                # Update the controller terms with the current error and the last executed command
                # Get control output from the nominal controller
                control_output = self.nominal_controllers[particle].output(pos_error, self.control_outputs_safe[particle]) # nominal
                # control_output = np.zeros(3) # disable nominal controller to test only the safe controller

                # # init_t = time.time()

                # # Get safe control output using control barrier functions
                # if not self.aux_enabled:
                #     self.control_output_safe = self.calculate_safe_control_output(control_output, 1./self.pub_rate_odom) # safe
                # # else: # TODO
                # #     self.control_output_safe = self.calculate_safe_control_output_w_aux(control_output, 1./self.pub_rate_odom, error_norm) # safe
                
                # # rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.")
                self.control_outputs_safe[particle] = control_output # to test nominal controller only

                if self.control_outputs_safe[particle] is not None:
                    # Scale down the calculated output if its norm is higher than the specified norm max_u
                    self.control_outputs_safe[particle] = self.scale_down_vector(self.control_outputs_safe[particle], max_u=0.15)

                    # print("Particle " + str(particle) + " u: " + str(self.control_outputs_safe[particle]))

                    # print(np.linalg.norm(self.control_output_safe - control_output))

                    # print(self.control_output_safe - control_output)
                    # rospy.logwarn("control_output:" + str(control_output))
                    # rospy.logwarn("control_output_safe: "+ str(self.control_output_safe))

                    # self.control_output_safe = control_output
            
            

    def odom_pub_timer_callback(self,event):
        # Only publish if enabled
        if self.enabled:
            for particle in self.follower_particles:
                # Do not proceed until the initial values have been set
                if ((not self.enabled_particles[particle]) or \
                    (not (particle in self.particle_positions)) or \
                    (not ("leader" in self.particle_positions)) or \
                    (not self.is_min_distance_set_for_particle(particle))):

                    # print("------ PARTICLE: " + str(particle) +" ----------")
                    # print("(not self.enabled_particles[particle])" + str((not self.enabled_particles[particle])))
                    # print("(not (particle in self.particle_positions))" + str((not (particle in self.particle_positions))))
                    # print("(not (\"leader\" in self.particle_positions))" + str((not ("leader" in self.particle_positions))))
                    # print("(not self.is_min_distance_set_for_particle(particle)))" + str((not self.is_min_distance_set_for_particle(particle))))
                    
                    continue
                    
                if self.control_outputs_safe[particle] is not None:
                    # Prepare Odometry message
                    odom = Odometry()
                    odom.header.stamp = rospy.Time.now()
                    odom.header.frame_id = "map"

                    dt_check = self.nominal_controllers[particle].get_dt() # 
                    dt = 1./self.pub_rate_odom

                    # Control output is the new position
                    odom.pose.pose.position.x =  self.particle_positions[particle].x + self.control_outputs_safe[particle][0]*dt
                    odom.pose.pose.position.y =  self.particle_positions[particle].y + self.control_outputs_safe[particle][1]*dt
                    odom.pose.pose.position.z =  self.particle_positions[particle].z + self.control_outputs_safe[particle][2]*dt

                    # The new orientation
                    odom.pose.pose.orientation = self.target_poses[particle].orientation # TODO: We assume the orientation is direclty applied
                    # TODO: Later involve the orientation into the controller

                    # Direclty update the pose of the particle since internal pose update through marker callback is disabled  when the particle is controlled
                    self.particle_positions[particle] = odom.pose.pose.position
                    self.particle_orientations[particle] = odom.pose.pose.orientation

                    # Publish
                    self.odom_publishers[particle].publish(odom)
                else:
                    self.control_outputs_safe[particle] = np.zeros(3)



    # def calculate_safe_control_output(self, nominal_u, dt):
    #     if not self.is_min_distance_set_for_particle():
    #         return

    #     # ---------------------------------------------------
    #     # Define optimization variables
    #     u = cp.Variable(3)

    #     # Define weights for each control input
    #     weights = np.array([1.0, 1.0, 0.1])  # Less weight on z control input
        
    #     # Define cost function with weights
    #     cost = cp.sum_squares(cp.multiply(weights, u - nominal_u)) / 2.0

    #     # Initialize the constraints
    #     constraints = []
    #     # ---------------------------------------------------

    #     # ---------------------------------------------------
    #     # DEFINE COLLISION AVOIDANCE CONTROL BARRRIER CONSTRAINT
    #     d_offset = 0.05 # offset distance from the obstacles
    #     h = self.min_distance - d_offset  # Control Barrier Function (CBF)
    #     alpha_h = self.alpha_collision_avoidance(h)

    #     # publish h that is the distance to collision for information
    #     # print("h distance to collision: ",str(h))
    #     self.info_h_collision_publisher.publish(Float32(data=h))

    #     Jx = ((self.min_distance_dx - self.min_distance) / self.delta_x) if self.delta_x != 0 else 0
    #     Jy = ((self.min_distance_dy - self.min_distance) / self.delta_y) if self.delta_y != 0 else 0
    #     Jz = ((self.min_distance_dz - self.min_distance) / self.delta_z) if self.delta_z != 0 else 0

    #     J = np.array([[Jx,Jy,Jz]]) # 1x3

    #     # publish J for information
    #     # print("J",str(J))
    #     J_msg = Float32MultiArray(data=np.ravel(J))
    #     self.info_J_publisher.publish(J_msg)

    #     J_tolerance = 0.05
    #     if not np.all(np.abs(J) < J_tolerance):
    #         # Add collision avoidance to the constraints
    #         constraints += [J @ u >= -alpha_h]
    #     else:
    #         rospy.loginfo("Follower: "+str(self.id)+", ignored J")
    #     # ---------------------------------------------------

    #     # ---------------------------------------------------
    #     h_overstretchs = []
    #     h_too_closes = []
    #     for particle in self.particles:
    #         if particle != self.id:
    #             # DEFINE OVERSTRETCHING AVOIDANCE CONTROL BARRRIER CONSTRAINT
    #             P_cur = self.calculate_current_relative_position_vec(self.id, particle)
    #             P_cur_norm = np.linalg.norm(P_cur) 

    #             min_norm_tolerance = 1.0e-9
    #             if P_cur_norm >= min_norm_tolerance:
    #                 P_cur_unit = P_cur/P_cur_norm

    #                 d_max = self.d_maxs[particle] # meters # maximum distance between the human held point and the robot controlled point

    #                 h_overstretch = d_max - P_cur_norm  # Control Barrier Function (CBF)
    #                 alpha_h = self.alpha_overstretch_avoidance(h_overstretch)

    #                 # print("h distance to overstretching: ",str(h_overstretch))
    #                 h_overstretchs.append(h_overstretch)

    #                 # calculate the velocity of the other particle
    #                 v_other_x = (self.particle_positions[particle].x - self.particle_positions_prev[particle].x)/dt if dt >= 1.0e-5 else 0.0
    #                 v_other_y = (self.particle_positions[particle].y - self.particle_positions_prev[particle].y)/dt if dt >= 1.0e-5 else 0.0
    #                 v_other_z = (self.particle_positions[particle].z - self.particle_positions_prev[particle].z)/dt if dt >= 1.0e-5 else 0.0
    #                 v_other = np.array([v_other_x,v_other_y,v_other_z])
    #                 # rospy.loginfo("Follower: "+str(self.id)+", v_other: "+str(v_other)+ " (for particle: " + str(particle) + ")")

    #                 # Add overstretching avoidance to the constraints
    #                 constraints += [-P_cur_unit.T @ u + P_cur_unit.T @ v_other >= -alpha_h]
    #             # ---------------------------------------------------

    #                 # ---------------------------------------------------
    #                 # DEFINE GETTING TOO CLOSE AVOIDANCE CONTROL BARRRIER CONSTRAINT
    #                 d_min = self.d_mins[particle] # meters # maximum distance between the human held point and the robot controlled point

    #                 h_too_close = P_cur_norm - d_min   # Control Barrier Function (CBF)
    #                 alpha_h = self.alpha_too_close_avoidance(h_too_close)

    #                 # print("h distance to agent getting too close: ",str(h))
    #                 h_too_closes.append(h_too_close)
                    
    #                 # Add getting too close avoidance to the constraints
    #                 constraints += [P_cur_unit @ u - P_cur_unit.T @ v_other >= -alpha_h]
    #                 # ---------------------------------------------------
        
    #     # publish h that is the distance to overstretching for information
    #     self.info_h_overstretching_publisher.publish(Float32MultiArray(data=h_overstretchs))

    #     # publish h that is the distance to agent getting too close for information
    #     self.info_h_too_close_publisher.publish(Float32MultiArray(data=h_too_closes))


    #     # Add also limit to the feasible u
    #     u_max = 0.15
    #     constraints += [cp.norm(u,'inf') <= u_max] # If inf-norm used, the constraint is linear, use OSQP solver (still QP)
    #     # constraints += [cp.norm(u,2)     <= u_max] # If 2-norm is used, the constraint is Quadratic, use CLARABEL or ECOS solver (Not QP anymore, a conic solver is needed)

    #     # ---------------------------------------------------
    #     # Define and solve problem
    #     problem = cp.Problem(cp.Minimize(cost), constraints)

    #     # # For warm-start
    #     # if hasattr(self, 'prev_optimal_u'):
    #     #     u.value = self.prev_optimal_u

    #     init_t = time.time() # For timing
    #     try:
    #         # problem.solve() # Selects automatically
    #         problem.solve(solver=cp.CLARABEL) # ~8ms 
    #         # problem.solve(solver=cp.CVXOPT) # ~10ms (warm start capable)
    #         # problem.solve(solver=cp.ECOS) # ~8ms
    #         # problem.solve(solver=cp.ECOS_BB) # ~8ms
    #         # problem.solve(solver=cp.GLOP) # NOT SUITABLE
    #         # problem.solve(solver=cp.GLPK) # NOT SUITABLE
    #         # problem.solve(solver=cp.GUROBI) # ~8ms
    #         # problem.solve(solver=cp.MOSEK) # Encountered unexpected exception importing solver CBC
    #         # problem.solve(solver=cp.OSQP) # ~7ms (default) (warm start capable)
    #         # problem.solve(solver=cp.PDLP) # NOT SUITABLE
    #         # problem.solve(solver=cp.SCIPY) # NOT SUITABLE 
    #         # problem.solve(solver=cp.SCS) # ~8ms (warm start capable)

    #     except cp.error.SolverError as e:
    #         rospy.logwarn("Follower: "+str(self.id)+",Could not solve the problem: {}".format(e))
    #         # self.prev_optimal_u = None # to reset warm-start
    #         return None

    #     rospy.logwarn("QP solver calculation time: " + str(1000*(time.time() - init_t)) + " ms.") # For timing

    #     # Print the available qp solvers
    #     # e.g. ['CLARABEL', 'CVXOPT', 'ECOS', 'ECOS_BB', 'GLOP', 'GLPK', 'GLPK_MI', 'GUROBI', 'MOSEK', 'OSQP', 'PDLP', 'SCIPY', 'SCS']
    #     rospy.loginfo_once(str(cp.installed_solvers()))
    #     # Print the solver used to solve the problem (once)
    #     rospy.loginfo_once("Solver used: " + str(problem.solver_stats.solver_name))

    #     # check if problem is infeasible or unbounded
    #     if problem.status in ["infeasible", "unbounded"]:
    #         rospy.logwarn("Follower: "+str(self.id)+", The problem is {}.".format(problem.status))
    #         # self.prev_optimal_u = None # to reset warm-start
    #         return None
        
    #     # # For warm-start in the next iteration
    #     # self.prev_optimal_u = u.value
        
    #     # Return optimal u
    #     return u.value
  
    # NOMINAL CONTROLLER RELATED HELPER FUNCTIONS
    def calculate_target_pose(self,particle):
        """ 
        Calculate the target position and orientation of the particle in world frame
        assuming that the particle is rigidly attached to the leader frame.
        """

        # Get current positions and orientations of the leader in world frame.
        leader_position = self.particle_positions["leader"] # Point()
        P_wl_in_w_cur = np.array([leader_position.x,leader_position.y,leader_position.z]) # current position of the leader in world frame
        leader_orientation = self.particle_orientations["leader"] # Quaternion()

        # Recall the initial position vector from leader to particle in world frame
        initial_relative_position = self.initial_relative_positions[particle]  # Point()
        P_lc_in_w_init = np.array([initial_relative_position.x,initial_relative_position.y,initial_relative_position.z])  # initial position vector from leader to controlled particle in world frame as np.array

        # Recall the initial relative orientation from leader to the particle
        initial_relative_orientation = self.initial_relative_orientations[particle] # Quaternion()
        # initial_relative_orientation = self.particle_orientations[particle] # Quaternion() Assumes initial relative orientation is the same as the particle's initial orientation

        # Convert leader's orientation to rotation matrix
        rotation_matrix = self.quaternion_to_matrix(leader_orientation)

        # Target pos as np.array
        target_pos = P_wl_in_w_cur + rotation_matrix.dot(P_lc_in_w_init)

        # Target orientation as Quaternion()
        target_ori = self.multiply_quaternions(leader_orientation, initial_relative_orientation)

        # Prepare the pose msg
        target_pose = Pose()
        target_pose.position = Point(*target_pos)
        target_pose.orientation = target_ori

        return target_pose

    
    def calculate_error(self, current_pose, target_pose):
        
        err_x = target_pose.position.x - current_pose.position.x
        err_y = target_pose.position.y - current_pose.position.y
        err_z = target_pose.position.z - current_pose.position.z

        # TODO: calculate error for orientations

        return np.array([err_x,err_y,err_z]),np.zeros(3)

    def scale_down_vector(self, u, max_u=0.005):
        norm_u = np.linalg.norm(u)

        if norm_u > max_u:
            u = u / norm_u * max_u

        return u

    # # SAFE CONTROLLER RELATED HELPER FUNCTIONS    
    # def calculate_current_relative_position_vec(self,particle1,particle2):
    #     p_x = self.particle_positions[particle1].x - self.particle_positions[particle2].x
    #     p_y = self.particle_positions[particle1].y - self.particle_positions[particle2].y
    #     p_z = self.particle_positions[particle1].z - self.particle_positions[particle2].z

    #     return np.array([p_x,p_y,p_z])

    # def alpha_collision_avoidance(self,h):
    #     # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
    #     # Piecewise Linear function is used

    #     c1 = 0.4 # 0.8 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
    #     c2 = 5.0 # 3.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
    #     alpha_h = c1*(h) if (h) >= 0 else c2*(h)
    #     return alpha_h
    
    # def alpha_overstretch_avoidance(self,h):
    #     # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
    #     # Piecewise Linear function is used
        
    #     c1 = 0.4 # 2.0 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
    #     c2 = 2.0 # 2.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
    #     alpha_h = c1*(h) if (h) >= 0 else c2*(h)
    #     return alpha_h
    
    # def alpha_too_close_avoidance(self,h):
    #     # calculates the value of extended_class_K function \alpha(h) for COLLISION AVOIDANCE
    #     # Piecewise Linear function is used
        
    #     c1 = 0.4 # 2.0 # 0.01 # 0.01 # 0.00335 # decrease this if you want to start reacting early to be more safe, (but causes less nominal controller following)
    #     c2 = 2.0 # 2.0 # 0.04 # 0.02 # 0.01 # increase this if you want to remove the offset violation more agressively, (but may cause instability due to the discretization if too agressive)
    #     alpha_h = c1*(h) if (h) >= 0 else c2*(h)
    #     return alpha_h


    # INFO RELATED HELPER FUNCTIONS
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
        self.info_target_position_publishers[particle].publish(marker)

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions.
        """
        w0, x0, y0, z0 = q1.w, q1.x, q1.y, q1.z
        w1, x1, y1, z1 = q2.w, q2.x, q2.y, q2.z

        w = w0*w1 - x0*x1 - y0*y1 - z0*z1
        x = w0*x1 + x0*w1 + y0*z1 - z0*y1
        y = w0*y1 - x0*z1 + y0*w1 + z0*x1
        z = w0*z1 + x0*y1 - y0*x1 + z0*w1

        return Quaternion(x, y, z, w)

    def inverse_quaternion(self, q):
        """
        Calculate the inverse of a quaternion.
        """
        w, x, y, z = q.w, q.x, q.y, q.z
        norm = w**2 + x**2 + y**2 + z**2

        return Quaternion(-x/norm, -y/norm, -z/norm, w/norm)
    
    def quaternion_to_matrix(self, quat):
        """
        Return 3x3 rotation matrix from quaternion geometry msg.
        """
        # Extract the quaternion components
        q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64, copy=True)

        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return np.identity(3)
        
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)

        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
            ), dtype=np.float64)


        

if __name__ == "__main__":
    rospy.init_node('dlo_velocity_controller_node', anonymous=False)

    node = VelocityControllerNode()

    rospy.spin()
