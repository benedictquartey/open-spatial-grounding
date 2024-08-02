import numpy as np
import open3d as o3d
import pickle
from PIL import Image
from os import path, makedirs
import networkx as nx
import random
import string
from osg.utils.pointcloud_utils import get_posed_rgbd_dataset, get_pointcloud_from_graph_r3d, get_pointcloud_from_graph_robot
from osg.utils.map_compression_utils import compress_observation_graph
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch
import cv2
import tqdm
import torch
from torch import Tensor



#function to generate random numeric alphanumeric string
def random_string(string_length=20):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


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


def decompose_pose_batch(homogeneous_pose_matrix):
    """Decomposes a 4x4 homogeneous transformation pose matrix into position and rotation components.

    Args:
        pose: The pose array, with shape (B, 4, 4)

    Returns:
        position: The translation vector, with shape (B, 3)
        rotation: The rotation matrix, with shape (B, 3, 3)
    """
    position = homogeneous_pose_matrix[:, :3, 3]
    rotation = homogeneous_pose_matrix[:, :3, :3]
    return position, rotation

def decompose_pose_single(homogeneous_pose_matrix):
    """Decomposes a single 4x4 homogeneous transformation pose matrix into position and rotation components.

    Args:
        pose: The pose matrix, with shape (4, 4)

    Returns:
        position: The translation vector, with shape (3)
        rotation: The rotation matrix, with shape (3, 3)
    """
    position = homogeneous_pose_matrix[:3, 3]
    rotation = homogeneous_pose_matrix[:3, :3]
    return position, rotation

def load_data(data_path, data_src, tmp_fldr, pcd_downsample, compression_percentage, compression_technique=None):
    if data_src == "robot":
        return load_robot_data(data_path, tmp_fldr, pcd_downsample, compression_percentage, compression_technique)
    elif data_src == "r3d":
        return load_r3d_data(data_path, tmp_fldr, pcd_downsample, compression_percentage, compression_technique)

def load_r3d_data(data_file,tmp_fldr, pcd_downsample=False, compression_percentage=None, compression_technique=None):
    if not path.exists(tmp_fldr):
        makedirs(tmp_fldr)

    observations_graph = nx.Graph()
    node_id2key = {}
    node_key2id = {}
    node_coords = {}
    to_pil = ToPILImage()

    node_percentage_to_keep = 100 - compression_percentage

    posed_dataset = get_posed_rgbd_dataset(key="r3d", path=data_file)
    #posed_dataset consists of posedrgbditems, which each has image(tensor), depth(tensor), mask(tensor), pose(tensor), intrinsics(tensor)

    for idx, pose in enumerate(posed_dataset.poses):
        waypoint_name = random_string()
        node_id2key[idx] = waypoint_name
        node_key2id[waypoint_name] = idx
        # print(f"{idx+1} out of {len(posed_dataset.poses)} || Getting rgbd data for waypoint:{waypoint_name}")

        node_image = posed_dataset[idx].image 
        node_pil_image = to_pil(node_image)
        node_depth = posed_dataset[idx].depth
        depth_confidence_mask = posed_dataset[idx].mask # has low confidence depths as True and high confidence as False ## see
        
        #mask out depth values using depth_confidence_mask, set non high confidence to 0.0
        # print(f"Masking out low confidence depth values for waypoint:{waypoint_name}")
        confidence_masked_depth = node_depth.clone()
        confidence_masked_depth[depth_confidence_mask] = 0.0
        # print("\tNumber of low confidence depth values masked out:", torch.sum(depth_confidence_mask).item())
        # print("\tNumber of remaining high confidence depth values:", confidence_masked_depth[confidence_masked_depth!=0].shape[0])
        
        node_pose = {}
        position, rotation_matrix = decompose_pose_single(posed_dataset[idx].pose)
        node_pose['pose_matrix'] = posed_dataset[idx].pose
        node_pose['position'] = position
        node_pose['rotation_matrix'] = rotation_matrix
        coord = tuple(node_pose['position'][0:2]) #x,y axis from position
        intrinsics = posed_dataset[idx].intrinsics
        node_coords[idx]=coord

        observations_graph.add_node(node_for_adding=idx, 
                                    rep_pose=node_pose,
                                    rgb_tensor=node_image, 
                                    rgb_pil=node_pil_image,
                                    depth_data=node_depth, 
                                    pose_matrix =posed_dataset[idx].pose ,
                                    pose=node_pose, 
                                    waypoint_key=waypoint_name, 
                                    xy_coordinate=coord, 
                                    intrinsics=intrinsics, 
                                    depth_confidence_mask=depth_confidence_mask,
                                    confidence_masked_depth = confidence_masked_depth)
    
    ##compress map
    if compression_percentage!=None:
        count = len(observations_graph.nodes)
        if compression_technique in ["direction","position","position_direction"]:
            print(f"Map Compression ... Keeping {node_percentage_to_keep}% of posed rgbd")
            observations_graph = compress_observation_graph(node_percentage_to_keep, observations_graph, group_by=compression_technique, visualize=False, tmp_fldr=tmp_fldr)
        else:
            print("Invalid compression strategy. Please select from: ['direction','position','position_direction']")
            return
        print(f"\tCompression strategy: {compression_technique} || Before compression: {count} nodes || After compression: {len(observations_graph.nodes)} nodes")

    #load pointcloud
    if path.exists(f"{tmp_fldr}/pointcloud.pcd"):
        print("loading pointcloud already on disk...")
        env_pointcloud = o3d.io.read_point_cloud(f"{tmp_fldr}/pointcloud.pcd")
    else:
        print("generating scene pointcloud...")
        env_pointcloud = get_pointcloud_from_graph_r3d(observations_graph, downsample=pcd_downsample)
        # env_pointcloud = get_pointcloud_r3d_dataset(posed_dataset, downsample=pcd_downsample) #pointcloud from posedrgbd dataset

        #do not save pointcloud if already exists
        print("saving pointcloud to disk...")
        o3d.io.write_point_cloud(f"{tmp_fldr}/pointcloud.pcd", env_pointcloud)
        print("")

    #save waypoint info to disk
    np.save(tmp_fldr+f'waypoints.npy', dict(observations_graph.nodes("pose")))
        

    # return env_pointcloud, observations_graph, node_id2key, node_key2id, node_coords
    return env_pointcloud, observations_graph


def get_node_key_values(graph: nx.Graph, key: str) -> list:
    """Get a list of values for a specific key from all nodes in the graph.

    Args:
        graph: The NetworkX graph.
        key: The key to extract values for.

    Returns:
        A list of values associated with the specified key from all nodes.
    """
    values = []
    for node, data in graph.nodes(data=True):
        if key in data:
            values.append(data[key])
        else:
            values.append(None)  # Or handle missing keys as needed

    return values


def load_robot_data(data_path,tmp_fldr, pcd_downsample=False, compression_percentage=None, compression_technique=None):
    """
    This function loads the data from the data_path and returns the observation data and edge connectivity data.

    Parameters:
        data_path (str): The path to the data folder.
    
    Returns:
        observation_data (dict): The observation data dictionary.
        edge_connectivity (dict): The edge connectivity dictionary.
    """

    observations_graph = nx.Graph()
    node_id2key = {}
    node_key2id = {}
    node_coords = {}

    node_percentage_to_keep = 100 - compression_percentage

    #load single pose data
    with open(f'{data_path}/pose_data.pkl', 'rb') as f:
        poses = pickle.load(f)

    #load cardinal pose data
    with open(f'{data_path}/pose_all_data.pkl', 'rb') as f:
        all_poses = pickle.load(f)

    #get corresponding images for each pose
    for id, waypoint_name in enumerate(poses.keys()):
        print(f"{id} out of {len(poses.keys())} || Getting cardinal images for waypoint:{waypoint_name}")
        image_collection={}
        image_tensor_collection={}
        depth_collection={}
        pose_collection={}
        pose_matrix_collection={}

        #loading cardinal data
        for i in range(4):
            image = Image.open(f'{data_path}/color_{waypoint_name}-{i}.jpg').convert("RGB")
            #get image tensor
            image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float()/255
            depth =  np.load(f'{data_path}/depth_{waypoint_name}-{i}', allow_pickle=True)
            pose = all_poses[waypoint_name+f"-{i}"]

            #generate pose matrix
            pose_matrix = np.eye(4)
            pose_matrix[:3,:3] = rotation_matrix_from_quaternion(pose['quaternion(wxyz)'])
            pose_matrix[:3,3] = pose['position']
            pose_matrix_collection[i]=pose_matrix

            image_collection[i]=image
            image_tensor_collection[i]=image_tensor
            depth_collection[i]=depth
            pose_collection[i]=pose

        node_id2key[id] = waypoint_name
        node_key2id[waypoint_name] = id
        node_image=image_collection
        node_image_tensor=image_tensor_collection
        node_pose=pose_collection
        #poses for cardinal images
        for key in node_pose.keys():
            node_pose[key]['rotation_matrix'] = rotation_matrix_from_quaternion(node_pose[key]['quaternion(wxyz)']) #computing rotation matrix from quaternion

        #actual pose for waypoint
        rep_pose = poses[waypoint_name]
        rep_pose['rotation_matrix'] = rotation_matrix_from_quaternion(rep_pose['quaternion(wxyz)']) #computing rotation matrix from quaternion
        node_depth=depth_collection
        coord = tuple(rep_pose['position'][0:2]) #x,y axis from position
        observations_graph.add_node(node_for_adding=id, rgb=node_image, rgb_tensor=node_image_tensor, pose=node_pose, pose_matrix=pose_matrix_collection, xy_coordinate=coord, depth_data=node_depth,waypoint_key=waypoint_name,rep_pose=rep_pose)
        node_coords[i]=coord
    
    # ##compress map
    if compression_percentage!=None:
        count = len(observations_graph.nodes)
        if compression_technique in ["direction","position","position_direction"]:
            print(f"Map Compression ... Keeping {node_percentage_to_keep}% of posed rgbd")
            observations_graph = compress_observation_graph(node_percentage_to_keep, observations_graph, group_by=compression_technique, visualize=False, tmp_fldr=tmp_fldr)
        else:
            print("Invalid compression strategy. Please select from: ['direction','position','position_direction']")
            return
        print(f"\tCompression strategy: {compression_technique} || Before compression: {count} nodes || After compression: {len(observations_graph.nodes)} nodes")

    #load pointcloud
    if path.exists(f"{tmp_fldr}/pointcloud.pcd"):
        print("loading pointcloud already on disk...")
        env_pointcloud = o3d.io.read_point_cloud(f"{tmp_fldr}/pointcloud.pcd")
    else:
        print("generating scene pointcloud...")
        env_pointcloud = get_pointcloud_from_graph_robot(observations_graph, downsample=pcd_downsample)
        #do not save pointcloud if already exists
        print("saving pointcloud to disk...")
        o3d.io.write_point_cloud(f"{tmp_fldr}/pointcloud.pcd", env_pointcloud)
        print("")

    #save waypoint info to disk
    np.save(tmp_fldr+f'waypoints.npy', dict(observations_graph.nodes("pose")))

    #save pointcloud to disk
    if not path.exists(tmp_fldr):
        makedirs(tmp_fldr)
    o3d.io.write_point_cloud(f"{tmp_fldr}/pointcloud.pcd", env_pointcloud)

    # return env_pointcloud, observations_graph, node_id2key, node_key2id, node_coords
    return env_pointcloud, observations_graph


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


def draw_observations_graph(observations_graph, node_coords, plt_size=(20,20),axis=False):
    """
    This function draws the observation graph.

    Parameters:
        observations_graph (nx.Graph): The observation graph.
        node_coords (dict): The node id to node coordinate dictionary.  
        plt_size (tuple): The size of the plot.
        axis (bool): Whether to draw the graph with axis or not.

    Returns:
        None
    """
    options = {
        "font_size": 10,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "grey",
        "linewidths": 3,
        "width": 2,
        "edge_color": "black",
    }
    
    if axis == False:
        #Draw graph without axis
        plt.figure(figsize=plt_size)
        nx.draw_networkx(observations_graph,pos=node_coords, **options)

        # draw edge weights
        labels = nx.get_edge_attributes(observations_graph, 'distance')
        nx.draw_networkx_edge_labels(observations_graph, pos=node_coords, edge_labels=labels,font_size=options['font_size'])

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.1)
        plt.axis("off")
        plt.show()

    elif axis == True:
        #Draw graph with axis
        fig, ax = plt.subplots(figsize=plt_size)
        nx.draw(observations_graph, pos=node_coords, node_color='k', ax=ax)
        nx.draw(observations_graph, pos=node_coords, node_size=1500, ax=ax)  # draw nodes and edges
        nx.draw_networkx_labels(observations_graph, pos=node_coords)  # draw node labels/names

        # draw edge weights
        labels = nx.get_edge_attributes(observations_graph, 'distance')
        nx.draw_networkx_edge_labels(observations_graph, pos=node_coords, edge_labels=labels, ax=ax)

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.axis("on")
        plt.show()


def make_pointcloud(graph: nx.Graph, chunk_size: int = 16, threshold: float = 0.9, downsample=False) -> o3d.geometry.PointCloud:
	fill_in = False #if fill_in is True, then gaps in depth image are filled in with black (at maximal distance..?)
	save_pc = True #if true, save point cloud to same location as dir_path+dir_name

	pose_dir = pickle.load(open(f"{data_path}{pose_data_fname}","rb"))

	#######################################
	# Visualize point cloud

	print("Number of files: ", len(file_names), "number of ids", len(file_ids))
	total_pcds = []
	total_colors = []
	total_axes = []
	for idx,node in enumerate(graph.nodes):

		color_img = graph.nodes[node]['rgb']
		color_img = color_img[:,:,::-1]  # RGB-> BGR
		depth_img = pickle.load(open(f"{data_path}depth_{str(file_num)}","rb"))#cv2.imread(dir_path+dir_name+"depth_"+str(file_num)+".jpg")

		H,W = depth_img.shape
		for i in range(H):
			for j in range(W):
				#first apply rot2 to move camera into hand frame, then apply rotation + transform of hand frame in vision frame
				transformed_xyz,_ = spot_pixel_to_world_frame(i,j,depth_img,rotation_matrix,position)

				total_pcds.append(transformed_xyz)

				# Add the color of the pixel if it exists:
				if 0 <= j < W and 0 <= i < H:
					total_colors.append(color_img[i,j] / 255)
				elif fill_in:
					total_colors.append([0., 0., 0.])

		mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,origin=[0,0,0])
		mesh_frame = mesh_frame.rotate(rotation_matrix, center=(0, 0, 0)).translate(position)
		#mesh_frame.paint_uniform_color([float(file_num)/num_files, 0.1, 1-(float(file_num)/num_files)])

		total_axes.append(mesh_frame)
		
	pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
	pcd_o3d.points = o3d.utility.Vector3dVector(total_pcds)
	pcd_o3d.colors = o3d.utility.Vector3dVector(total_colors)

	#bb = o3d.geometry.OrientedBoundingBox(center=np.array([0,0,0]),R=rot2_mat,extent=np.array([1,1,1]))



	return(pcd_o3d)