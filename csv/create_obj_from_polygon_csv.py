#!/usr/bin/env python3

import shapely.geometry # Polygon, LineString
import shapely.ops # nearest_points, triangulate

import os 
import csv
import numpy as np

import meshpy.triangle

import matplotlib.pyplot as plt # To visualize polygons


# Set the default DPI for all images
plt.rcParams['figure.dpi'] = 100  # e.g. 300 dpi
# Set the default figure size
plt.rcParams['figure.figsize'] = [12.8, 9.6]  # e.g. 6x4 inches
# plt.rcParams['figure.figsize'] = [6.4, 4.8]  # e.g. 6x4 inches


"""
create_obj_from_polygon_csv.py

Author: Burak Aksoy

This script is designed as a utility script to create .obj file from a given csv file which holds the vertices of a polygon. 

In the implementation, the coordinates of the polygon is taken in x=0 plane i.e. YZ plane from that csv file. 
"""


def create_mesh_with_meshpy(polygon, max_area=0.01):
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

def plot_mesh_meshpy(mesh, save=False):
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)

    print("Mesh number of triangles (Meshpy): " + str(mesh_tris.shape[0]))

    # Show the created mesh
    plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)

    # Highlight the first and last points
    plt.scatter(*mesh_points[0], color='red')
    plt.scatter(*mesh_points[-1], color='blue')

    # Annotate each point with its index
    for i, (x, y) in enumerate(mesh_points):
        plt.annotate(str(i), (x, y), fontsize=6)

    # Set the aspect ratio of the axes to be equal
    plt.axis('equal')

    # # Set the axes limits
    # plt.xlim([-4, 4])
    # plt.ylim([-2, 4])

    # Show grid lines
    plt.grid(True)

    if save:
        # Save the figure
        plt.savefig("meshpy_mesh.png", dpi=300)

    plt.show()

def export_mesh_as_obj(mesh, filename):
    with open(filename, 'w') as f:
        for p in mesh.points:
            f.write("v {} {} {}\n".format(0, p[0], p[1], 0)) # x-coordinate is 0 as it is a 2D mesh in YZ plane
        for element in mesh.elements:
            f.write("f")
            for vertex in element:
                f.write(" {}".format(vertex + 1)) # .obj format uses 1-indexing
            f.write("\n")

# Function to load and translate polygon from CSV file
def load_polygon(file_path, translate_vector):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)            
        polygon_pts = np.array(list(reader), dtype=np.float32)

    # Apply translation to the polygon points
    polygon_pts += translate_vector

    # Construct the polygon using only YZ plane coordinates
    return shapely.geometry.Polygon(polygon_pts[:, [1, 2]])

# Define file path for polygon points

# Define file path for polygon points
polygon_pts_csv_file_directory = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_centralized_rope/csv/"
polygon_pts_csv_file_name = "surface_floor_u_gate.csv"
polygon_pts_obj_file_directory = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_centralized_rope/models/"

# Use os.path.join for creating file paths
polygon_pts_csv_file_path = os.path.join(polygon_pts_csv_file_directory, polygon_pts_csv_file_name)

# Initialize translation vector for the polygon
polygon_translate = np.array([0.0, 0.0, 0.0])

# Load and prepare the polygon
polygon = load_polygon(polygon_pts_csv_file_path, polygon_translate)
print("Polygon number of vertices: " + str(len(polygon.exterior.coords) - 1))

# Triangulate the polygon using MeshPy
polygon_mesh_meshpy = create_mesh_with_meshpy(polygon, max_area=None)

# Visualize the MeshPy triangulation
plot_mesh_meshpy(polygon_mesh_meshpy, save=False)

# Prepare file name by removing .csv in the end and adding .obj as new extension
base_name, _ = os.path.splitext(polygon_pts_csv_file_name)
polygon_pts_obj_file_name = base_name + ".obj"

# Use os.path.join to create the full path for the OBJ file
polygon_pts_obj_file_path = os.path.join(polygon_pts_obj_file_directory, polygon_pts_obj_file_name)

export_mesh_as_obj(polygon_mesh_meshpy, polygon_pts_obj_file_path)
