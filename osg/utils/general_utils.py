import numpy as np
import open3d as o3d
import pickle
from PIL import Image
from os import path, makedirs
import networkx as nx

def get_spatial_referents(encoding_map):
    referent_spatial_details = parse_spatial_relations(encoding_map)
    return referent_spatial_details


# Function to recursively get unique referents as keys and spatial details as values
def parse_spatial_relations(input_list):
    result = {}

    def recursive_parse(item, parent=None):
        if '::' in item:
            item_components = item.split('::')
            object_name = item_components[0]
            details = item_components[1:]
            detail_str = '::'.join(details)
            open_parentheses = 0
            current_detail = ''
            
            for char in detail_str:
                if char == '(':
                    open_parentheses += 1
                elif char == ')':
                    open_parentheses -= 1

                current_detail += char

                if open_parentheses == 0 and char == ')':
                    current_detail = current_detail.strip()
                    # Remove leading '::' from the detail
                    if current_detail.startswith('::'):
                        current_detail = current_detail[2:]

                    if object_name not in result:
                        result[object_name] = []
                    if current_detail not in result[object_name]:
                        result[object_name].append(current_detail)
                    inside_parentheses = current_detail[current_detail.find("(")+1:current_detail.rfind(")")]
                    for part in inside_parentheses.split(','):
                        recursive_parse(part.strip(), object_name)
                    current_detail = ''

        else:
            if parent and item not in result:
                result[item] = []
            elif not parent and item not in result:
                result[item] = []

    for entry in input_list:
        recursive_parse(entry)
    
    return result


#Function to find depth associated with mask at center pixel
def get_center_pixel_depth(mask,depth_img):
    mask = np.asarray(mask)

    #Find coordinates of center pixel
    center_coords = np.argwhere(mask.astype(float)).mean(axis=0)
    center_x, center_y = center_coords.astype(int)
    
    #Retrieve depth value of center pixel
    depth_of_center_pixel = np.asarray(depth_img)[center_x, center_y]
    return (center_x, center_y),depth_of_center_pixel

#Function to find depth associated with all pixels in mask
def get_mask_pixels_depth(mask,depth_img):
    mask = np.asarray(mask)
    depth_img = np.asarray(depth_img)

    boolean_mask = (mask == True)

    #get coordinates of pixels where mask is true
    mask_pixel_coords = np.argwhere(boolean_mask)

    #id array of depth values at pixels where mask is true
    depths_associated_with_mask = depth_img[boolean_mask] 

    # get average of depth values that are not zero
    if isinstance(depths_associated_with_mask,np.ndarray):
        non_zero_depths_associated_with_mask = depths_associated_with_mask[depths_associated_with_mask != 0]
        if len(non_zero_depths_associated_with_mask) == 0:
            avg_non_zero_depths_associated_with_mask = 0.0
        else:
            avg_non_zero_depths_associated_with_mask = non_zero_depths_associated_with_mask.mean()

    return mask_pixel_coords,depths_associated_with_mask,avg_non_zero_depths_associated_with_mask

def get_bounding_box_center_depth(bounding_box, depth_img):
    y1,x1,y2,x2 = bounding_box
    center_x,center_y = int((x1+x2)/2), int((y1+y2)/2)

    #Retrieve depth value of center pixel
    depth_of_center_pixel = np.asarray(depth_img)[center_x, center_y]
    return (center_x, center_y),depth_of_center_pixel

def get_bounding_box_pixels_depth(bounding_box,depth_img):
    x1, y1, x2, y2 = bounding_box
    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
    # Get all coordinates of pixels within the bounding box
    bounding_box_pixel_coords = []
    # for y in range(y1, y2+1):
    #     for x in range(x1, x2+1):
    #         bounding_box_pixel_coords.append((y, x))

    for y in range(y1+1, y2-1):
        for x in range(x1+1, x2-1):
            bounding_box_pixel_coords.append((y, x))

      # Retrieve depth values for each pixel
    depths = []
    for pixel_coord in bounding_box_pixel_coords:
        depth = depth_img[pixel_coord[0], pixel_coord[1]]
        if depth != 0:
            depths.append(depth)
    #compute average depth
    depths = np.asarray(depths)
    average_depth = np.mean(depths)

    return bounding_box_pixel_coords,depths,average_depth


def pixel_to_world_frame(i,j,pixel_depth,rotation_matrix,position):
    '''
    Converts a pixel (i,j) in HxW image to 3d position in world frame (spot 'vision' frame)
    i,j: pixel location in image
    depth_img: HxW depth image
    rotaton_matrix: 3x3 rotation matrix in world frame
    position: 3x1 position vector in world frame
   
    Note: “vision” frame: An inertial frame that estimates the fixed location in the world (relative to where the robot is booted up),
    and is calculated using visual analysis of the world and the robot’s odometry.
    ''' 
    #hand_tform_camera comes from line below, just a hardcoded version of it
    #rot2 = mesh_frame.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
    
    hand_tform_camera = np.array([[ 3.74939946e-33,6.12323400e-17,1.00000000e+00],
    [-1.00000000e+00,6.12323400e-17,0.00000000e+00],
    [-6.12323400e-17,-1.00000000e+00,6.12323400e-17]])  

    #Intrinsics for RGB hand camera on spot
    CX = 320
    CY = 240
    FX= 552.0291012161067
    FY = 552.0291012161067

    #Compute 3d position of pixel(i,j) in camera frame/cordinate system. Optical center is origin
    z_RGB = pixel_depth
    x_RGB = (j - CX) * z_RGB / FX
    y_RGB = (i - CY) * z_RGB / FY   

    bad_z = z_RGB == 0 #if z_RGB is 0, the depth was 0
    
    #Move from camera frame to robot hand frame
    camera2hand = np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))

    #World (vision) frame is the hand frame rotated by the robot rotation matrix in world frame and translated by the robot position in world frame
    transformed_xyz = np.matmul(rotation_matrix,camera2hand) + position  
    return(transformed_xyz,bad_z)

def load_data(data_path,tmp_fldr):
    """
    This function loads the data from the data_path and returns the observation data and edge connectivity data.

    Parameters:
        data_path (str): The path to the data folder.
    
    Returns:
        observation_data (dict): The observation data dictionary.
        edge_connectivity (dict): The edge connectivity dictionary.
    """
    observation_data = {'images':{},
        'poses':{},
        'depth_data':{},
        'rep_pose':{}}

    #load pointcloud
    env_pointcloud = o3d.io.read_point_cloud(f"{data_path}/pointcloud.pcd")

    #load single pose data
    with open(f'{data_path}/pose_data.pkl', 'rb') as f:
        poses = pickle.load(f)

    #load cardinal pose data
    with open(f'{data_path}/pose_all_data.pkl', 'rb') as f:
        all_poses = pickle.load(f)

    #get waypoint edge connectivity
    with open(f'{data_path}/connectivty_cost_dict.pkl', 'rb') as f:
        edge_connectivity = pickle.load(f)

    #get corresponding images for each pose
    for id, waypoint_name in enumerate(poses.keys()):
        print(f"{id} out of {len(poses.keys())} || Getting cardinal images for waypoint:{waypoint_name}")
        image_collection={}
        depth_collection={}
        pose_collection={}

        #loading cardinal data
        for i in range(4):
            image = Image.open(f'{data_path}/color_{waypoint_name}-{i}.jpg').convert("RGB")
            depth =  np.load(f'{data_path}/depth_{waypoint_name}-{i}', allow_pickle=True)
            pose = all_poses[waypoint_name+f"-{i}"]
            image_collection[i]=image
            depth_collection[i]=depth
            pose_collection[i]=pose

        observation_data['rep_pose'][waypoint_name]=poses[waypoint_name]    
        observation_data['images'][waypoint_name]=image_collection
        observation_data['depth_data'][waypoint_name]=depth_collection
        observation_data['poses'][waypoint_name]=pose_collection  

    #save pointcloud to disk
    if not path.exists(tmp_fldr):
        makedirs(tmp_fldr)
    o3d.io.write_point_cloud(f"{tmp_fldr}/pointcloud.pcd", env_pointcloud)

    return observation_data, edge_connectivity, env_pointcloud

def create_observation_graph(observation_data,edge_connectivity,tmp_fldr=None):
    """
    This function creates the observation graph from the observation data and edge connectivity data.

    Parameters:
        observation_data (dict): The observation data dictionary.
        edge_connectivity (dict): The edge connectivity dictionary.

    Returns:
        observations_graph (nx.Graph): The observation graph.
        node_id2key (dict): The node id to node key dictionary.
        node_key2id (dict): The node key to node id dictionary.
        node_coords (dict): The node id to node coordinate dictionary.
    """
    observations_graph = nx.Graph()
    node_id2key = {}
    node_key2id = {}
    node_coords = {}

    for i,node_key in enumerate(observation_data['images'].keys()):
        node_id2key[i] = node_key
        node_key2id[node_key] = i

        # print(f"Adding node {i} with key {node_key} to graph")

        node_image=observation_data['images'][node_key]
        node_pose=observation_data['poses'][node_key]
        #poses for cardinal images
        for key in node_pose.keys():
            node_pose[key]['rotation_matrix'] = rotation_matrix_from_quaternion(node_pose[key]['quaternion(wxyz)']) #computing rotation matrix from quaternion
        #actual pose for waypoint
        rep_pose = observation_data['rep_pose'][node_key]
        rep_pose['rotation_matrix'] = rotation_matrix_from_quaternion(rep_pose['quaternion(wxyz)']) #computing rotation matrix from quaternion
        node_depth=observation_data['depth_data'][node_key]
        coord = tuple(rep_pose['position'][0:2]) #x,y axis from position

        observations_graph.add_node(node_for_adding=i, rgb=node_image, pose=node_pose, xy_coordinate=coord, depth_data=node_depth,waypoint_key=node_key,rep_pose=rep_pose)
        node_coords[i]=coord

    # print("\nAdding edges to graph...\n")
    # Add edge connectivity data
    for waypoint_name in edge_connectivity:
        for connected_waypoint in edge_connectivity[waypoint_name]:            
            origin_node_id, connected_node_id = node_key2id[waypoint_name], node_key2id[connected_waypoint[0]]
            origin_node_xy, connected_node_xy = node_coords[origin_node_id], node_coords[connected_node_id]

            ecludiean_distance = np.linalg.norm(np.array(origin_node_xy) - np.array(connected_node_xy))
            rounded = round(ecludiean_distance, 2)
            # print(f"Node: {str(origin_node_id):2s} is connected to Node: {str(connected_node_id):2s} || edge cost: {ecludiean_distance}")
            # print(f"Waypoint id: {str(origin_node_id):2s} || name: {waypoint_name:40s} connected to: Waypoint id: {str(connected_node_id):2s} || name: {connected_waypoint[0]:40s} with cost: {connected_waypoint[1]}")
            observations_graph.add_edge(origin_node_id, connected_node_id, distance=rounded)

    
    #save waypoint info to disk
    #create tmp_fldr folder if it doesn't exist
    if tmp_fldr != None:
        if not path.exists(tmp_fldr):
            makedirs(tmp_fldr)

        np.save(tmp_fldr+f'waypoints.npy', observation_data['rep_pose'])
    else:
        np.save(f'waypoints.npy', observation_data['rep_pose'])

    return observations_graph, node_id2key, node_key2id, node_coords

#function to compute rotation matrix from quaternion
def rotation_matrix_from_quaternion(quaternion):
   # Step 1: Normalize the quaternion
   quaternion = quaternion / np.linalg.norm(quaternion)

   # Step 2: Extract quaternion components
   w, x, y, z = quaternion

   # Step 3: Construct rotation matrix
   R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
               [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
               [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
   return R