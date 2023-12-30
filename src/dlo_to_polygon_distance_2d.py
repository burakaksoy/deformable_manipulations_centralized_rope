#!/usr/bin/env python3

import rospy

import visualization_msgs.msg # Marker
import geometry_msgs.msg # PolygonStamped, Point32
import std_msgs.msg  # Float32

import shapely.geometry # Polygon, LineString
import shapely.ops # nearest_points

import csv
import numpy as np
import time

import meshpy.triangle

"""
dlo_to_polygon_distance_2d.py

Author: Burak Aksoy

Reads a csv file which has 3D point coordinates of a polygon.
Creates the polygon with shapely library.
(This polygon is published with a separate timer for rviz visualization)
Subscribes to the DLO's visualization_msgs.Marker, extracts the line segment center coordinates.
Creates/updates a polyline with LineString object of shapely library for each DLO Marker message.

Publishes the minimum distance between the polyline and the polygon using Shapely distance() function.
If some portions of the polyline is inside the polygon, the most inner point distance is published as negative distance (TODO)

IMPORTANT NOTE: This is a 2D application, all points are assumed to be on x=0 plane i.e. YZ plane. 
"""

class DloToPolygonDistance2d:
    def __init__(self):
        rospy.init_node('dlo_to_polygon_distance_2d', anonymous=False)

        self.dlo_marker_topic_name = rospy.get_param("~dlo_marker_topic_name", "/dlo_points")

        self.pub_distance_topic_name = rospy.get_param("~pub_distance_topic_name", "/distance_to_obj")

        self.pub_rate_polygon = rospy.get_param("~pub_rate_polygon", 1)
        
        self.viz_polygon_topic_name = rospy.get_param("~viz_polygon_topic_name", "/polygon")
        self.viz_polygon_mesh_topic_name = self.viz_polygon_topic_name + "_mesh"
        self.viz_min_distance_line_topic_name = rospy.get_param("~viz_min_distance_line_topic_name", "/min_distance_line_marker")
        
        self.polygon_pts_csv_file_path = rospy.get_param("~polygon_pts_csv_file_path")

        self.polygon_publish = rospy.get_param("~polygon_publish", True)

        self.viz_min_distance_line_color_rgba = rospy.get_param("~viz_min_distance_line_color_rgba", [1.0, 1.0, 1.0, 1.0])
        
        self.polygon_translate = rospy.get_param("~polygon_translate", [0.0, 0.0, 0.0])
        self.polygon_translate = np.array(self.polygon_translate)
        

        # Load polygon from csv
        with open(self.polygon_pts_csv_file_path, 'r') as file:
            reader = csv.reader(file)            
            self.polygon_pts = np.array(list(reader), dtype=np.float32)
            self.polygon_pts = self.polygon_pts + self.polygon_translate # translate
            # print(polygon_pts.tolist())
            self.polygon = shapely.geometry.Polygon(self.polygon_pts[:,[1,2]]) # get only YZ plane

            self.polygon = self.polygon.buffer(0.1)

            rospy.loginfo("Polygon number of vertices: " + str(len(self.polygon.exterior.coords) - 1))

            # Triangulate the polygon to obtain a simple mesh
            self.polygon_mesh = self.create_mesh(self.polygon, max_area=0.5)

        self.polyline = shapely.geometry.LineString()

        rospy.Subscriber(self.dlo_marker_topic_name, 
                         visualization_msgs.msg.Marker, 
                         self.dlo_marker_sub_callback, 
                         queue_size=10)
        self.pub_distance = rospy.Publisher(self.pub_distance_topic_name, std_msgs.msg.Float32, queue_size=10)
        self.pub_min_distance_line = rospy.Publisher(self.viz_min_distance_line_topic_name, visualization_msgs.msg.Marker, queue_size=10)

        if self.polygon_publish:
            self.pub_polygon = rospy.Publisher(self.viz_polygon_topic_name, geometry_msgs.msg.PolygonStamped, queue_size=1)
            self.pub_polygon_mesh = rospy.Publisher(self.viz_polygon_mesh_topic_name, visualization_msgs.msg.Marker, queue_size=1)

            self.polygon_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_polygon), self.polygon_pub_timer_callback)
        

    def dlo_marker_sub_callback(self, msg):
        # check if the marker is a line list (id 6)
        if msg.type == visualization_msgs.msg.Marker.LINE_LIST:
            # init_t = time.time()
            # points = [(p.x, p.y, p.z) for p in msg.points]
            points = [(p.y, p.z) for p in msg.points] # get only YZ plane
            # print(points)
            self.polyline = shapely.geometry.LineString(points)

            # Find the nearest points on the polygon and the polyline
            point_on_polyline, point_on_polygon = shapely.ops.nearest_points(self.polyline, self.polygon)

            self.publish_min_distance_line_marker(point_on_polyline, point_on_polygon)

            # Find the minimum distance and publish
            # distance = self.polygon.distance(self.polyline)
            distance = point_on_polygon.distance(point_on_polyline)

            # rospy.logwarn("DLO distance calculatio time: " + str(1000*(time.time() - init_t)) + " ms.")
            # If the polyline is inside the polygon, find the negative distance as penetration amount (TODO)

            self.pub_distance.publish(std_msgs.msg.Float32(data=distance))



    def polygon_pub_timer_callback(self, event):
        # Extend the polygon with the distance offset that we will use
        polygon_new = self.polygon.buffer(0.05)

        # Create PolygonStamped message
        polygon_stamped = geometry_msgs.msg.PolygonStamped()
        polygon_stamped.header.stamp = rospy.Time.now()
        polygon_stamped.header.frame_id = "map" # or whatever frame_id you are working with
        polygon_stamped.polygon.points = [geometry_msgs.msg.Point32(x=0.0, 
                                                  y=point[0], 
                                                  z=point[1]) for point in list(polygon_new.exterior.coords)[:-1]]
        
        # Publish the PolygonStamped message
        self.pub_polygon.publish(polygon_stamped)

        # ---------------------------------------------------------
        # init_t = time.time()
        # publish the triangles here
        # Create and publish the triangles
        marker = visualization_msgs.msg.Marker()

        marker.header.frame_id = "map"
        marker.type = marker.TRIANGLE_LIST
        marker.action = marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.color.r = self.viz_min_distance_line_color_rgba[0]
        marker.color.g = self.viz_min_distance_line_color_rgba[1]
        marker.color.b = self.viz_min_distance_line_color_rgba[2]
        marker.color.a = self.viz_min_distance_line_color_rgba[3]

        # Loop through the triangle elements of the mesh
        for triangle in self.polygon_mesh.elements:
            for index in triangle:  # Each triangle has 3 indices
                point = self.polygon_mesh.points[index]
                p = geometry_msgs.msg.Point()
                p.x, p.y, p.z = 0.0, point[0], point[1]  # Assuming flat triangles on z = 0 plane
                marker.points.append(p)

        # Publish the triangles
        self.pub_polygon_mesh.publish(marker)
        # rospy.logwarn("DLO polygon publish time: " + str(1000*(time.time() - init_t)) + " ms.")
        # ---------------------------------------------------------

    def publish_min_distance_line_marker(self,point_on_polyline, point_on_polygon):
        # Prepare a Marker message for the shortest distance
        min_distance_marker = visualization_msgs.msg.Marker()
        min_distance_marker.header.frame_id = "map"
        
        min_distance_marker.type = visualization_msgs.msg.Marker.LINE_STRIP
        min_distance_marker.action = visualization_msgs.msg.Marker.ADD

        min_distance_marker.pose.orientation.w = 1.0;

        min_distance_marker.scale.x = 0.01 # specify a suitable size
        r = self.viz_min_distance_line_color_rgba[0]
        g = self.viz_min_distance_line_color_rgba[1]
        b = self.viz_min_distance_line_color_rgba[2]
        a = self.viz_min_distance_line_color_rgba[3]
        min_distance_marker.color = std_msgs.msg.ColorRGBA(r=r, g=g, b=b, a=a) # Red color
        min_distance_marker.points.append(geometry_msgs.msg.Point(x=0, y=point_on_polyline.x, z=point_on_polyline.y))
        min_distance_marker.points.append(geometry_msgs.msg.Point(x=0, y=point_on_polygon.x, z=point_on_polygon.y))

        self.pub_min_distance_line.publish(min_distance_marker)

    def create_mesh(self, polygon, max_area=0.01):
        poly_points = list(polygon.exterior.coords)[:-1]

        def round_trip_connect(start, end):
            """
            creates a set of pairs where each pair contains an index 
            and the index of the next point, it helps to define the 
            edges of the polygon.
            """
            return [(i, i+1) for i in range(start, end)] + [(end, start)]
    
        info = meshpy.triangle.MeshInfo()
        info.set_points(poly_points)
        info.set_facets(round_trip_connect(0, len(poly_points)-1))

        mesh = meshpy.triangle.build(info, max_volume=max_area)

        # mesh_points = np.array(mesh.points) # N by 2 [x y] points array, N: Number of vertices
        # mesh_tris = np.array(mesh.elements) # M by 3 [v1 v2 v3] vertex indices array that forms the triangles, M: Number of triangles
        # mesh_facets = np.array(mesh.facets) # K by 2 [v1 v2] vertex indices array that defines the boundary edges of the mesh, K: Number of edges, (not necessarily ordered)

        return mesh

if __name__ == "__main__":
    dloToPolygonDistance2d = DloToPolygonDistance2d()
    rospy.spin()
