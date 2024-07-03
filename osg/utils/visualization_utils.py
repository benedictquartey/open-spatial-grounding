import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from osg.utils.general_utils import pixel_to_world_frame


COLORS = {
    "Deep_Red": [1.0, 0.0, 0.0],
    "Deep_Green": [0.0, 0.5, 0.0],
    "Deep_Blue": [0.0, 0.0, 1.0],
    "Deep_Purple": [0.5, 0.0, 0.5],
    "Deep_Cyan": [0.0, 0.5, 0.5],
    "Deep_Magenta": [0.5, 0.0, 0.5],
    "Deep_Brown": [0.35, 0.16, 0.14],
    "Deep_Gray": [0.2, 0.2, 0.2],
    "Deep_Black": [0.0, 0.0, 0.0],
    "Deep_Maroon": [0.5, 0.0, 0.25],
    "Orange": [1.0, 0.5, 0.0],
    "Yellow": [1.0, 1.0, 0.0],
    "Green": [0.0, 1.0, 0.0],
    "Red": [1.0, 0.0, 0.0],
    "Blue": [0.0, 0.0, 1.0],
    "Pink": [1.0, 0.0, 1.0],
    "Teal": [0.0, 1.0, 1.0],
    "Lime": [0.5, 1.0, 0.0],
    "Gold": [1.0, 0.84, 0.0],
    "Crimson": [0.86, 0.08, 0.24],
    "Navy": [0.0, 0.0, 0.5],
    "Indigo": [0.29, 0.0, 0.51],
    "Olive": [0.5, 0.5, 0.0],
    "Coral": [1.0, 0.5, 0.31],
    "Black": [0.0, 0.0, 0.0],
    "light_gray": [0.8, 0.8, 0.8],
    "Royal_Blue":(0.1, 0.1, 0.5)
}

# Create a reverse lookup for color names
COLOR_LOOKUP = {tuple(value): key for key, value in COLORS.items()}

def visualize(pcd,items,h_min_bottom, h_max_top,filter_top_bottom=False):
    if filter_top_bottom:
        points = np.asarray(pcd.points)
        # Filter out points that are outside the height limits
        height_mask = (points[:, 2] > h_min_bottom) & (points[:, 2] < h_max_top)
        points_filtered = points[height_mask]
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        #color
        colors_filtered = np.asarray(pcd.colors)[height_mask]
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

    o3d.visualization.draw_geometries([pcd]+items)
    
    return pcd, items

# Function to create a top-down 2D view of the point cloud
def plot_top_down_view(point_cloud, z_threshold):
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Apply the z-axis threshold
    below_threshold = points[:, 2] < z_threshold
    points = points[below_threshold]
    point_colors = colors[below_threshold]

    # Extract the x, y coordinates and colors
    x = points[:, 0]
    y = points[:, 1]

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=point_colors, s=1, edgecolor='none')  # s is the size of the point
    plt.axis('equal')  # Equal scaling of the axes.
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Top-Down 2D View of the Point Cloud (z < {z_threshold})')
    plt.grid(False)  # Turn off the grid as it is not usually useful for point clouds
    plt.show()

def color_coded_heatmap(point_cloud, mask_positions_of_interest, threshold_percentage=1,kdtree_search_radius=0.5,use_colors=False):
    """
    Visualize a point cloud with a heatmap color coding, considering a threshold for points of interest.

    :param point_cloud: open3d.geometry.PointCloud object
    :param mask_positions_of_interest: dict with object names and positions of interest that will have unique colors
    :param threshold_percentage: float, percentage (0-1) of points of interest to be considered for coloring
    :return: None, displays the visualization
    """
    global COLORS
    global COLOR_LOOKUP

    num_points = np.asarray(point_cloud.points).shape[0]
    if use_colors:
            colors = np.asarray(point_cloud.colors) #Keep original colors
    else:
            colors = np.full((num_points, 3), COLORS["light_gray"]) # Base color for most points

    # Build a KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    #Try and get matching colors for each object if object name has a color
    object_colors={}
    unique_color_values = list(COLORS.values())
    object_names = list(mask_positions_of_interest.keys())

    for name in object_names:
        name_parts = name.split("_")
        for part in name_parts:
            for color_name in COLORS:
                if part.lower() in color_name.lower():
                    color_value = COLORS[color_name]
                    print("Found color for",name," || ","assigning color: ",color_name)
                    object_colors[name] = color_value  
                    #remove color from list of available colors
                    unique_color_values.remove(color_value)
                    object_names.remove(name)
                    break
    # Assign a unique color to each remaining object
    for i, obj_name in enumerate(object_names):
        object_colors[obj_name] = unique_color_values[i % len(unique_color_values)]


    # Iterate through each object type and its positions
    for object_name, positions in mask_positions_of_interest.items():
        color_value = tuple(object_colors[object_name])
        color_name = COLOR_LOOKUP[color_value]
        
        if positions:
            # Convert list of positions to numpy array for efficient processing
            positions_array = np.vstack(positions)

            # Determine how many points to consider based on the threshold percentage
            num_points_of_interest = int(len(positions_array) * threshold_percentage)
            
            # Randomly select a subset of points based on the threshold
            if threshold_percentage < 1.0:
                indices = np.random.choice(len(positions_array), num_points_of_interest, replace=False)
                positions_array = positions_array[indices]
            
            print(f"Object:{object_name:10s}|| Color:{color_name:5s} || Considering {num_points_of_interest} out of {len(positions)} points.")

            # For efficiency, use batch query for KDTree search
            total_colored_points = 0
            for position in positions_array:
                # print(f"Position: {position}")
                [k, idx, _] = kdtree.search_radius_vector_3d(position, kdtree_search_radius) #find neighbors with distance less than kdtree_search_radius
                # [k, idx, _] = kdtree.search_knn_vector_3d(position, 1000) #find 1000 closest neighbors
                if k > 0:
                    colors[idx, :] = color_value
                    total_colored_points += k  # Sum the number of points colored
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud
    # o3d.visualization.draw_geometries([point_cloud])

def get_all_mask3dpositions(elements,diff_threshold = 0.5):
    mask_points_of_interest = {}
    for element in elements:
        mask3dpositions=[]
        all_mask_pixels = element['mask_all_pixels']
        all_mask_depths = element['mask_all_pixels_depth']
        avg_nonzero_depth = np.mean(all_mask_depths[all_mask_depths!=0])

        for mask_pixel,mask_depth in zip(all_mask_pixels,all_mask_depths):
            center_y, center_x = mask_pixel
            pixel_depth = mask_depth
            if pixel_depth == 0 or abs(pixel_depth-avg_nonzero_depth)>diff_threshold: #if depth is 0 or too different from mean of non zero depths in mask (by diff_threshold) then skip
                continue
            rotation_matrix = element['origin_nodepose']['rotation_matrix']
            position = element['origin_nodepose']['position']
            transformed_point,bad_point = pixel_to_world_frame(center_y,center_x,pixel_depth,rotation_matrix,position)
            mask3dpositions.append(transformed_point)
        try:
            mask_points_of_interest[element["mask_label"]].extend(mask3dpositions)
        except:
            mask_points_of_interest[element["mask_label"]]=mask3dpositions
    return mask_points_of_interest

def place_elements_in_scene(elements,truth_unique_classes=None,size=0.15, show_origin=False):
    '''
    A function that visualizes a list of elements in a scene

    pcd: pointcloud of scene
    elements: list of elements to visualize
    colors: list of colors for each element
    '''
    global COLORS
    global COLOR_LOOKUP
    total_axes = []
    NoneType = type(None)

    #dynamically assign unique colors to each element based on label
    object_names = [element['mask_label'] for element in elements]
    unique_color_values = list(COLORS.values())
    unique_classes = {}

    #Try and get matching colors for each object if object name has a color
    for element_name in object_names:
        if element_name not in unique_classes:
            name_parts = element_name.split("_")
            for part in name_parts:
                for color_name in COLORS:
                    if part.lower() in color_name.lower():
                        color_value = COLORS[color_name]
                        print("Found color for",element_name," || ","assigning color: ",color_name)
                        unique_classes[element_name] = color_value
                        #remove color from list of available colors
                        unique_color_values.remove(color_value)
                        object_names.remove(element_name)
                        break
    # Assign a unique color to each remaining object
    for i, obj_name in enumerate(object_names):
        if obj_name not in unique_classes:
            unique_classes[obj_name] = unique_color_values[i % len(unique_color_values)]

    # print("Grounded Referents Key:",unique_classes)
    #print unique classes color names not values
    for key in unique_classes:
        print("Element:",key,"|| Color:",COLOR_LOOKUP[tuple(unique_classes[key])])

    #visualize the elements
    for element in (elements):
        element_label = element['mask_label']
        element_id = element['mask_id']
        if element_label not in unique_classes:
            continue
        else:
            class_color_value=tuple(unique_classes[element_label])
            # print("Processing element:", element_label, "with color",color)
            center_y, center_x = element['mask_center_pixel']
            pixel_depth = element['mask_depth']
            rotation_matrix = element['origin_nodepose']['rotation_matrix']
            position = element['origin_nodepose']['position']
            transformed_point = element['worldframe_3d_position']
            # print("Element:",element_id,"|| Position:",transformed_point)
            if type(transformed_point) == NoneType:
                print(f"Bad point found for element: {element_id} skipping ...")
            else:
                sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
                sphere_mesh.paint_uniform_color(class_color_value)
                sphere_mesh.translate(transformed_point)
                total_axes.append(sphere_mesh)
    # Add origin coordinate frame if show_origin is True
    if show_origin:
        origin_coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
        total_axes.append(origin_coordframe)

    return total_axes

def load_viz_data(result_dir_path,elements_filename, use_augmented=False):
    pcd, elements, waypoints, saved_plan = None, None, None, None
    try :
        if use_augmented==True:
            pcd = o3d.io.read_point_cloud(f"{result_dir_path}/augmented_pointcloud.pcd")
        else:
            pcd = o3d.io.read_point_cloud(f"{result_dir_path}/pointcloud.pcd")
    except:
        print("No pointcloud.pcd file found in result directory")
    try: 
        elements = np.load(f"{result_dir_path}/{elements_filename}.npy",allow_pickle=True)
    except:
        print(f"No {elements_filename}.npy file found in result directory")
    try:
        waypoints = np.load(f"{result_dir_path}/waypoints.npy",allow_pickle=True).item()
    except:
        print("No waypoints.npy file found in result directory")
    try:
        saved_plan = np.load(f"{result_dir_path}/saved_plan.npy",allow_pickle=True).item()
    except:
        print("No saved_plan.npy file found in result directory")
    return pcd,elements,waypoints,saved_plan
