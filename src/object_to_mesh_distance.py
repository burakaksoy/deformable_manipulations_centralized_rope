#!/usr/bin/env python3

import rospy

import visualization_msgs.msg # Marker
import geometry_msgs.msg # PolygonStamped, Point32
import std_msgs.msg  # Float32

# import shapely.geometry # Polygon, LineString
# import shapely.ops # nearest_points

import csv
import numpy as np
import time

import meshpy.triangle

import trimesh # To read obj and find the distance between the obj file and the fabric mesh

"""
dlo_to_mesh_distance.py

Author: Burak Aksoy

Reads a obj file which has the mesh of a polygon or a polyhedron as obstacle.
(This mesh is published with a separate timer for rviz visualization)

Subscribes to the deformable object's visualization_msgs.Marker, extracts the particle center coordinates.
The positions of centroids of the particles are updated for each Deformable Object Simulation Marker message.

Publishes the minimum distance between the centeroid coordinates and the mesh polygon/polyhedron using trimesh mesh.nearest.on_surface(self.vertices) function.
If some portions of the deformable object is inside the polygon/polyhedron, the most inner point distance is published as negative distance (TODO)

NOTE: This is completely a 3D application. The minimum distances are in 3d.
"""

class ObjectToMeshDistance:
    """
    Object: The deformable object
    Mesh: 3D Obstacle representation as a model mesh file with a .obj file
    """
    def __init__(self):
        rospy.init_node('object_to_mesh_distance', anonymous=False)

        self.object_marker_topic_name = rospy.get_param("~object_marker_topic_name", "/dlo_points")

        self.pub_distance_topic_name = rospy.get_param("~pub_distance_topic_name", "/distance_to_obj")

        self.pub_rate_mesh = rospy.get_param("~pub_rate_mesh", 1)
        self.pub_rate_min_distance = rospy.get_param("~pub_rate_min_distance", 50)
        
        self.viz_mesh_topic_name = rospy.get_param("~viz_mesh_topic_name", "/mesh")
        self.viz_min_distance_line_topic_name = rospy.get_param("~viz_min_distance_line_topic_name", "/min_distance_line_marker")
        
        self.mesh_model_obj_file_path = rospy.get_param("~mesh_model_obj_file_path")

        self.mesh_publish = rospy.get_param("~mesh_publish", True)

        self.viz_min_distance_line_color_rgba = rospy.get_param("~viz_min_distance_line_color_rgba", [1.0, 1.0, 1.0, 1.0])
        self.viz_mesh_color_rgba              = rospy.get_param("~viz_mesh_color_rgba",              [1.0, 1.0, 1.0, 1.0])
        
        self.mesh_translation   = np.array(rospy.get_param("~mesh_translation",   [0.0, 0.0, 0.0]))
        self.mesh_rotationAxis  = np.array(rospy.get_param("~mesh_rotationAxis",  [0, 0, 1]))
        self.mesh_rotationAngle = np.deg2rad(np.array(rospy.get_param("~mesh_rotationAngle", 0.0)))
        self.mesh_scale         = np.array(rospy.get_param("~mesh_scale",         [1, 1, 1]))
        

        # Load mesh from obj file
        self.mesh = trimesh.load_mesh(self.mesh_model_obj_file_path)

        self.max_mesh_faces = rospy.get_param("~max_mesh_faces", 300)
        rospy.loginfo("Loaded mesh has: " + str(len(self.mesh.faces)) + " faces.")

        if len(self.mesh.faces) > self.max_mesh_faces:
            # Check the number of faces in the original mesh
            rospy.logwarn("The loaded mesh has faces more than specified max faces =" + str(self.max_mesh_faces) + ".")
            rospy.logwarn("The loaded mesh will be simplified accordingly.")

            # Perform quadric mesh simplification
            self.mesh = self.mesh.simplify_quadric_decimation(self.max_mesh_faces)
            rospy.logwarn("Loaded mesh has: " + str(len(self.mesh.faces)) + " faces after simplification.")

        if not self.mesh.is_watertight:
            rospy.logwarn('Warning: mesh is not closed, signed_distance may not be accurate')

        # Translate rotate scale the mesh
        self.mesh.apply_scale(self.mesh_scale)

        rotation_matrix = trimesh.transformations.rotation_matrix(self.mesh_rotationAngle, self.mesh_rotationAxis)
        self.mesh.apply_transform(rotation_matrix)

        self.mesh.apply_translation(self.mesh_translation)

        # Create a proximity query object for each mesh
        self.proximity_query_mesh = trimesh.proximity.ProximityQuery(self.mesh)


        # initialize fabric data structure
        self.vertices = None
        
        rospy.Subscriber(self.object_marker_topic_name, 
                         visualization_msgs.msg.Marker, 
                         self.object_marker_sub_callback, 
                         queue_size=10)
        
        
        self.pub_distance          = rospy.Publisher(self.pub_distance_topic_name, std_msgs.msg.Float32, queue_size=10)
        self.pub_min_distance_line = rospy.Publisher(self.viz_min_distance_line_topic_name, visualization_msgs.msg.Marker, queue_size=10)

        if self.mesh_publish:
            self.pub_mesh           = rospy.Publisher(self.viz_mesh_topic_name, visualization_msgs.msg.Marker, queue_size=1)
            self.pub_mesh_wireframe = rospy.Publisher(self.viz_mesh_topic_name + "_wireframe", visualization_msgs.msg.Marker, queue_size=1)

            self.mesh_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_mesh), self.mesh_pub_timer_callback)

        self.min_distance_pub_timer = rospy.Timer(rospy.Duration(1. / self.pub_rate_min_distance), self.min_distance_pub_timer_callback)
        

    def object_marker_sub_callback(self, msg):
        # check if the marker is a point list 
        if msg.type == visualization_msgs.msg.Marker.POINTS:
            self.vertices = np.array([(p.x, p.y, p.z) for p in msg.points])

    def min_distance_pub_timer_callback(self,event):
        # # if self.face_tri_ids is not None:
        if self.vertices is not None:
            # init_t = time.time()
            
            closest_points, distances, _ = self.mesh.nearest.on_surface(self.vertices)

            # Find the minimum element in the array
            min_distance = np.min(distances)

            # rospy.logwarn("Deformable object to obstacle distance calculation time: " + str(1000*(time.time() - init_t)) + " ms.")

            # Find the index of the minimum element in the array
            min_distance_index = np.argmin(distances)

            point_on_object = self.vertices[min_distance_index]
            point_on_mesh = closest_points[min_distance_index].squeeze()

            # ------------------------ RESULTS ------------------------
            # # print("min_distance: ",min_distance)
            # # print("closest_point on deformable object: ",     point_on_object)
            # # print("closest_point on obstacle mesh: ", point_on_mesh)    
            
            self.publish_min_distance_line_marker(point_on_object, point_on_mesh)
            self.pub_distance.publish(std_msgs.msg.Float32(data=min_distance))

    def mesh_pub_timer_callback(self, event):
        # Create Mesh Marker message
        marker = visualization_msgs.msg.Marker()
        
        marker.header.frame_id = "map"
        marker.type = marker.TRIANGLE_LIST
        marker.action = marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

        # Set the marker orientation
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the marker color
        marker.color.r = self.viz_mesh_color_rgba[0]
        marker.color.g = self.viz_mesh_color_rgba[1]
        marker.color.b = self.viz_mesh_color_rgba[2]
        marker.color.a = self.viz_mesh_color_rgba[3]

        for face in self.mesh.faces:
            triangle_points = self.mesh.vertices[face]
            for point in triangle_points:
                p = geometry_msgs.msg.Point()
                p.x, p.y, p.z = point
                marker.points.append(p)

        # Publish the Mesh Marker message
        self.pub_mesh.publish(marker)

        # # Create a new Marker message for the wireframe
        wireframe_marker = visualization_msgs.msg.Marker()
        wireframe_marker.header.frame_id = "map"
        wireframe_marker.type = wireframe_marker.LINE_LIST
        wireframe_marker.action = wireframe_marker.ADD
        wireframe_marker.scale.x = 0.01 # Choose a suitable line width
        wireframe_marker.pose.orientation.w = 1.0
        
        # Set color for wireframe
        wireframe_marker.color.r = 0.5
        wireframe_marker.color.g = 0.5
        wireframe_marker.color.b = 0.5
        wireframe_marker.color.a = 0.5
        
        for face in self.mesh.faces:
            triangle_points = self.mesh.vertices[face]
            for i in range(3):
                p1 = geometry_msgs.msg.Point()
                p1.x, p1.y, p1.z = triangle_points[i]
                
                p2 = geometry_msgs.msg.Point()
                p2.x, p2.y, p2.z = triangle_points[(i+1) % 3]
                
                wireframe_marker.points.append(p1)
                wireframe_marker.points.append(p2)
        
        # Publish the wireframe Marker message
        self.pub_mesh_wireframe.publish(wireframe_marker)

    def publish_min_distance_line_marker(self,point_on_object, point_on_mesh):
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
        min_distance_marker.points.append(geometry_msgs.msg.Point(x=point_on_object[0], y=point_on_object[1], z=point_on_object[2]))
        min_distance_marker.points.append(geometry_msgs.msg.Point(x=point_on_mesh[0], y=point_on_mesh[1], z=point_on_mesh[2]))

        self.pub_min_distance_line.publish(min_distance_marker)

if __name__ == "__main__":
    objectToMeshDistance = ObjectToMeshDistance()
    rospy.spin()
