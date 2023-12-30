#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty, EmptyResponse

import time
import math

MAX_POINTS = 4000    # Choose a reasonable limit
MIN_DISTANCE = 0.001  # Minimum distance to update, in meters 

class TrailMaker:
    def __init__(self):
        rospy.init_node('trail_maker_node')

        # Fetching ROS parameters
        try:
            self.id = int(rospy.get_param("~id")) # follower particle id of this controller node controls
        except:
            rospy.logerr("NO PARTICLE ID IS SPECIFIED FOR THE TRAIL MAKER NODE")

        self.particles = None
        self.follower_particles = None
        self.leader_particle = None
        while not self.particles:
            try:
                self.particles = rospy.get_param("/custom_static_particles")
                self.follower_particles = rospy.get_param("/follower_particles") 
                self.leader_particle = rospy.get_param("/leader_particle") 
            except:
                rospy.logerr("No particles obtained from ROS parameters in Trail Maker node.")
                time.sleep(0.5)

        marker_array_topic = rospy.get_param('~trail_marker_array_topic', 'trail_marker_array')
        reset_trail_service = rospy.get_param('~reset_trail_service', 'reset_trail')
        odom_topic = rospy.get_param('~odom_topic', 'odom')
        header_frame_id = rospy.get_param('~header_frame_id', 'map')
        
        marker_scale = rospy.get_param('~marker_scale', 0.1)
        marker_color = rospy.get_param('~marker_color', {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0})

        text_scale = rospy.get_param('~text_scale', 0.1)
        text_color = rospy.get_param('~text_color', {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0})

        self.text_offset_xyz = rospy.get_param("~text_offset_xyz", {'x': 0.0, 'y': 0.0, 'z': 0.1})

        sphere_scale = rospy.get_param('~sphere_scale', 0.07) # diameter
        sphere_color = rospy.get_param('~sphere_color', {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0})

        self.last_point = None

        # Publisher for the trail marker
        self.marker_pub = rospy.Publisher(marker_array_topic, MarkerArray, queue_size=10)

        # Subscriber for the controlled point's odometry
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

        # Advertise the reset service
        self.reset_service = rospy.Service(reset_trail_service, Empty, self.reset_trail)

        # Setting up the markers
        self.trail_marker = Marker()
        self.trail_marker.header.frame_id = header_frame_id
        self.trail_marker.ns = "trail"
        self.trail_marker.id = 0
        self.trail_marker.type = Marker.LINE_STRIP
        self.trail_marker.action = Marker.ADD
        self.trail_marker.scale.x = marker_scale  # Width of the line
        self.trail_marker.color.r = marker_color['r']
        self.trail_marker.color.g = marker_color['g']
        self.trail_marker.color.b = marker_color['b']
        self.trail_marker.color.a = marker_color['a']
        self.trail_marker.pose.orientation.w = 1.0

        self.text_marker = Marker()
        self.text_marker.header.frame_id = header_frame_id
        self.text_marker.ns = "text"
        self.text_marker.id = 1
        self.text_marker.type = Marker.TEXT_VIEW_FACING
        self.text_marker.action = Marker.ADD
        self.text_marker.text = self.find_text_marker_text()
        self.text_marker.scale.z = text_scale  
        self.text_marker.color.r = text_color['r']
        self.text_marker.color.g = text_color['g']
        self.text_marker.color.b = text_color['b']
        self.text_marker.color.a = text_color['a']
        self.text_marker.pose.orientation.w = 1.0

        self.sphere_marker = Marker()
        self.sphere_marker.header.frame_id = header_frame_id
        self.sphere_marker.ns = "sphere"
        self.sphere_marker.id = 2
        self.sphere_marker.type = Marker.SPHERE
        self.sphere_marker.action = Marker.ADD
        self.sphere_marker.scale.x = sphere_scale 
        self.sphere_marker.scale.y = sphere_scale 
        self.sphere_marker.scale.z = sphere_scale 
        self.sphere_marker.color.r = sphere_color['r']
        self.sphere_marker.color.g = sphere_color['g']
        self.sphere_marker.color.b = sphere_color['b']
        self.sphere_marker.color.a = sphere_color['a']

        

    def odom_callback(self, odom):
        new_point = odom.pose.pose.position

        # Check the distance to the last point
        if self.last_point is None or self.distance(new_point, self.last_point) >= MIN_DISTANCE:
            self.last_point = new_point  # Update last point

            if len(self.trail_marker.points) >= MAX_POINTS:
                self.trail_marker.points.pop(0)
            
            self.trail_marker.points.append(new_point)
            self.trail_marker.header.stamp = rospy.Time.now()

            self.text_marker.header.stamp = rospy.Time.now()
            self.text_marker.pose.position.x = odom.pose.pose.position.x + self.text_offset_xyz['x']
            self.text_marker.pose.position.y = odom.pose.pose.position.y + self.text_offset_xyz['y']
            self.text_marker.pose.position.z = odom.pose.pose.position.z + self.text_offset_xyz['z']

            self.sphere_marker.header.stamp = rospy.Time.now()
            self.sphere_marker.pose = odom.pose.pose
            self.sphere_marker.pose.orientation.w = 1.0

            # Append the markers to the marker array and publish
            marker_array = MarkerArray()
            marker_array.markers.append(self.trail_marker)
            marker_array.markers.append(self.text_marker)
            marker_array.markers.append(self.sphere_marker)
            self.marker_pub.publish(marker_array)

    def reset_trail(self, request):
        self.trail_marker.points = []
        marker_array = MarkerArray()
        marker_array.markers.append(self.trail_marker)
        self.marker_pub.publish(marker_array)
        return EmptyResponse()
    
    def find_text_marker_text(self):
        if self.id in self.follower_particles:
            return "Follower:" + str(self.id)
        elif self.id == self.leader_particle:
            return "Leader:" + str(self.id)
        else:
            return "Binded:" + str(self.id)

    def distance(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    trail_maker = TrailMaker()
    trail_maker.run()