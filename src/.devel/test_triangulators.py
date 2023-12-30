#!/usr/bin/env python3

import shapely.geometry # Polygon, LineString
import shapely.ops # nearest_points, triangulate

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
TODO.py

Author: Burak Aksoy

This script is designed to compare the triangulation methods of polygons using two different libraries: MeshPy and Shapely. The script consists of functions to create and visualize meshes from a given polygon using both libraries.

The MeshPy library is used to generate a triangulated mesh from a polygon. The function 'create_mesh_with_meshpy' takes a Shapely polygon as input and creates a mesh using MeshPy's triangulation capabilities. The resulting mesh is visualized using Matplotlib in the function 'plot_mesh_meshpy'.

For Shapely, the functions 'create_mesh_with_shapely' and 'plot_mesh_shapely' are implemented. Shapely's 'triangulate' function is used for triangulating the polygon. However, it has been observed that Shapely's triangulation method tends to produce a convex hull of the input geometry. This means that the triangulated mesh might not respect the concave aspects of the original polygon and might extend beyond its boundaries. This behavior is a limitation of the Shapely library's current triangulation approach, which is based on Delaunay triangulation and does not support concave polygons effectively.

Note that: The key difference lies in the level of control and the complexity of the triangulation process. While both use Delaunay triangulation as a base, MeshPy's integration with the Triangle library allows for more nuanced and controlled triangulation, which can lead to different results compared to Shapely's more straightforward approach.

The script also includes functionality to load polygon data from a CSV file, translate the polygon if needed, and set up Matplotlib for visualization. The comparisons made between MeshPy and Shapely triangulations in this script are crucial for understanding the limitations and capabilities of these libraries in handling polygon triangulations.

Note: For more complex or concave polygons, alternative libraries or methods may be required to achieve accurate and boundary-respecting triangulation.
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

def create_mesh_with_shapely(polygon):
    # Triangulate the polygon using Shapely
    triangles = shapely.ops.triangulate(polygon)

    # Extract the points and triangle indices
    points = []
    point_indices = {}
    tris = []

    for tri in triangles:
        idx = []
        for point in tri.exterior.coords[:-1]:  # Exclude the closing point of the triangle
            if point not in point_indices:
                point_indices[point] = len(points)
                points.append(point)
            idx.append(point_indices[point])
        tris.append(idx)

    # Convert to NumPy arrays for consistency with MeshPy format
    mesh_points = np.array(points)
    mesh_tris = np.array(tris)

    # Create a simple structure to hold the mesh information
    mesh = {'points': mesh_points, 'elements': mesh_tris}
    
    print("Mesh number of triangles (Shapely): " + str(mesh_tris.shape[0]))
    return mesh

def plot_mesh_shapely(mesh, save=False):
    mesh_points = mesh['points']
    mesh_tris = mesh['elements']

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

    # Show grid lines
    plt.grid(True)

    if save:
        # Save the figure
        plt.savefig("shapely_mesh.png", dpi=300)

    plt.show()

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
# polygon_pts_csv_file_path = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_centralized_rope/csv/surface_floor_u_gate.csv"
polygon_pts_csv_file_path = "/home/burak/catkin_ws_deformable/src/deformable_manipulations_centralized_rope/csv/surface_floor_box_narrow_out.csv"

# Initialize translation vector for the polygon
polygon_translate = np.array([0.0, 0.0, 0.0])

# Load and prepare the polygon
polygon = load_polygon(polygon_pts_csv_file_path, polygon_translate)
print("Polygon number of vertices: " + str(len(polygon.exterior.coords) - 1))

# Triangulate the polygon using MeshPy
polygon_mesh_meshpy = create_mesh_with_meshpy(polygon, max_area=None)

# Visualize the MeshPy triangulation
plot_mesh_meshpy(polygon_mesh_meshpy, save=False)

# Triangulate the polygon using Shapely
polygon_mesh_shapely = create_mesh_with_shapely(polygon)

# Visualize the Shapely triangulation
plot_mesh_shapely(polygon_mesh_shapely, save=False)


    

    


