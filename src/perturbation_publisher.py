#!/usr/bin/env python3

import sys
import rospy

import numpy as np
import time

from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32


class PerturbationPublisherNode:
    def __init__(self):
        self.pub_rate_odom = rospy.get_param("~pub_rate_odom", 50)

        self.dlo_points_topic_name    = rospy.get_param("~dlo_points_topic_name", "/dlo_points") # subscribed
        self.odom_topic_prefix = rospy.get_param("~pub_odom_topic_prefix", "odom_follower_") # published

        self.follower_particles = None

        while (not self.follower_particles):
            try:
                self.follower_particles = rospy.get_param("/follower_particles") # Default follower particles 
            except:
                rospy.logwarn("No particles obtained from ROS parameters!.")
                time.sleep(0.5)

        self.delta_x = rospy.get_param("/delta_x", 0.0)
        self.delta_y = rospy.get_param("/delta_y", 0.1)
        self.delta_z = rospy.get_param("/delta_z", 0.1)

        # We will hold each follower current position and orientation in the following dictionaries
        # so each dict will have as many as the follower particles
        self.particle_positions = {}
        self.particle_orientations = {} 
        self.initial_values_set = False  # Initialization state variable

        self.sub_marker = rospy.Subscriber(self.dlo_points_topic_name, Marker, self.marker_callback, queue_size=10)

        # we will create a 3 odom publisher for each follower particle
        # so each of the following dictionaries will have elements as many as the follower particles
        self.odom_publishers_delta_x = {}
        self.odom_publishers_delta_y = {}
        self.odom_publishers_delta_z = {}

        for particle in self.follower_particles:
            self.odom_publishers_delta_x[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_x", Odometry, queue_size=10)
            self.odom_publishers_delta_y[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_y", Odometry, queue_size=10)
            self.odom_publishers_delta_z[particle] = rospy.Publisher(self.odom_topic_prefix + str(particle) + "_z", Odometry, queue_size=10)

        self.odom_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_odom), self.odom_pub_timer_callback)


    def odom_pub_timer_callback(self,event):
        # Do not proceed until the initial values have been set
        if (not self.initial_values_set):
            return
        
        for particle in self.follower_particles:
            t_now = rospy.Time.now()
            frame_id = "map"

            # Prepare Odometry message
            odom = Odometry()
            odom.header.stamp = t_now
            odom.header.frame_id = frame_id 
            
            # ----------------------------------------------------------------------------------------
            pose_delta_x = Pose()
            pose_delta_x.position.x = self.particle_positions[particle].x + self.delta_x
            pose_delta_x.position.y = self.particle_positions[particle].y 
            pose_delta_x.position.z = self.particle_positions[particle].z 

            # Keep the same orientation
            pose_delta_x.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_x
            self.odom_publishers_delta_x[particle].publish(odom)
            # ----------------------------------------------------------------------------------------
            pose_delta_y = Pose()
            pose_delta_y.position.x = self.particle_positions[particle].x # + self.delta_x
            pose_delta_y.position.y = self.particle_positions[particle].y + self.delta_y
            pose_delta_y.position.z = self.particle_positions[particle].z # + self.delta_z

            # Keep the same orientation
            pose_delta_y.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_y
            self.odom_publishers_delta_y[particle].publish(odom)
            # ----------------------------------------------------------------------------------------
            pose_delta_z = Pose()
            pose_delta_z.position.x = self.particle_positions[particle].x # + self.delta_x
            pose_delta_z.position.y = self.particle_positions[particle].y # + self.delta_y
            pose_delta_z.position.z = self.particle_positions[particle].z + self.delta_z

            # Keep the same orientation
            pose_delta_z.orientation = self.particle_orientations[particle]

            odom.pose.pose = pose_delta_z
            self.odom_publishers_delta_z[particle].publish(odom)

    def marker_callback(self, marker):
        if marker.type == Marker.POINTS:
            for particle in self.follower_particles:
                self.particle_positions[particle] = marker.points[particle]

        if marker.type == Marker.LINE_LIST:
            for particle in self.follower_particles:
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

                self.particle_orientations[particle] = quaternion
                
        if not self.initial_values_set:
            # After all initial relative positions and orientations have been calculated, set the initialization state variable to True
            self.initial_values_set = True


if __name__ == "__main__":
    rospy.init_node('perturbation_publisher_node', anonymous=False)

    node = PerturbationPublisherNode()

    rospy.spin()
