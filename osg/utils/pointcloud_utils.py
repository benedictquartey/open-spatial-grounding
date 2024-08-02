import tqdm
import torch
import open3d as o3d
import more_itertools
from torch import Tensor
from typing import Iterator,NamedTuple
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from os import path, makedirs, listdir
from osg.utils.dataset_class import PosedRGBDItem,R3DDataset
import networkx as nx
from tqdm import tqdm
import numpy as np

def spot_pixel_to_world_frame_batched(depth: Tensor, pose: Tensor) -> Tensor:
    """
    Converts pixel coordinates in a depth image to 3D positions in the world frame for a batch of images.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """
    (bsz, height, width), device, dtype = depth.shape, depth.device, depth.dtype

    # Intrinsics for RGB hand camera on spot
    CX = 320
    CY = 240
    FX = 552.0291012161067
    FY = 552.0291012161067

    intrinsics = torch.tensor([
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype).unsqueeze(0).repeat(bsz, 1, 1)

    # Transformation from camera frame to robot hand frame
    hand_tform_camera = torch.tensor([
        [3.74939946e-33, 6.12323400e-17, 1.00000000e+00],
        [-1.00000000e+00, 6.12323400e-17, 0.00000000e+00],
        [-6.12323400e-17, -1.00000000e+00, 6.12323400e-17]
    ], device=device, dtype=dtype)

    # Generate pixel grid
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy"
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat(bsz, 1, 1)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Apply inverse intrinsics
    inv_intrinsics = torch.inverse(intrinsics).transpose(-1, -2)
    xyz = xyz @ inv_intrinsics
    xyz = xyz * depth.flatten(1).unsqueeze(-1)

    # Move from camera frame to robot hand frame
    xyz = xyz @ hand_tform_camera.T

    # Apply pose transformation
    rot_matrix = pose[:, :3, :3]
    trans_vector = pose[:, :3, 3]
    xyz = (xyz[..., None, :] * rot_matrix[..., None, :3, :3]).sum(dim=-1) + trans_vector[..., None, :3]

    # Reshape the output to (B, H, W, 3)
    xyz = xyz.unflatten(1, (height, width))

    return xyz

def spot_pixel_to_world_frame(i,j,pixel_depth,rotation_matrix,position):
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

    bad_z = False
    if z_RGB == 0:
        bad_z = True
    
    #Move from camera frame to robot hand frame
    camera2hand = np.matmul(hand_tform_camera,np.array([x_RGB,y_RGB,z_RGB]))

    #World (vision) frame is the hand frame rotated by the robot rotation matrix in world frame and translated by the robot position in world frame
    transformed_xyz = np.matmul(rotation_matrix,camera2hand) + position  
    return(transformed_xyz,bad_z)

def get_xyz_coordinate(x, y, depth_value, pose, intrinsics):
    """Returns the XYZ coordinates for a specific pixel coordinate.

    Args:
        depth: The depth array, with shape (1, H, W)
        pose: The pose array, with shape (4, 4)
        intrinsics: The intrinsics array, with shape (3, 3)
        x: The x coordinate of the pixel
        y: The y coordinate of the pixel

    Returns:
        The XYZ coordinates of the projected point, with shape (3,)
    """

    dtype = intrinsics.dtype

    # Extract the depth value for the specific pixel
    # depth_value = depth[y, x]

    # Create the homogeneous pixel coordinate
    pixel_coord = torch.tensor([x, y, 1], dtype=dtype)

    # Apply intrinsics
    xyz = pixel_coord @ get_inv_intrinsics(intrinsics).transpose(-1, -2)

    # Scale by the depth value
    xyz = xyz * depth_value

    # Apply pose transformation
    xyz = (xyz @ pose[:3, :3].transpose(0, 1)) + pose[:3, 3]

    # Mask out bad depth points.
    bad_z = False
    if depth_value == 0:
        print(f"Bad depth point at ({x}, {y})")
        bad_z = True

    return xyz,bad_z

def get_posed_rgbd_dataset(key: str,path: str) -> Dataset[PosedRGBDItem]:
        return R3DDataset(path)

def get_inv_intrinsics(intrinsics: Tensor) -> Tensor:
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics

def get_xyz(depth: Tensor, mask: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        mask: The mask array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """

    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    # Mask out bad depth points.
    xyz = xyz.unflatten(1, (height, width))
    xyz[mask.squeeze(1)] = 0.0

    return xyz

def get_pointcloud_r3d_dataset(ds: Dataset[PosedRGBDItem], chunk_size: int = 16, threshold:float = 0.9, downsample=False) -> Iterator[tuple[Tensor, Tensor]]:
    #threshold determines number of points to be included in the pointcloud
    """Iterates XYZ points from the dataset.

    Args:
        ds: The dataset to iterate points from
        desc: TQDM bar description
        chunk_size: Process this many frames from the dataset at a time

    Yields:
        The XYZ coordinates, with shape (B, H, W, 3), and a mask where a value
        of True means that the XYZ coordinates should be ignored at that
        point, with shape (B, H, W)
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ds_len = len(ds)  # type: ignore
    xyzs = []
    rgbs = []

    for inds in more_itertools.chunked(tqdm.trange(ds_len, desc='point cloud'), chunk_size):
        rgb, depth, mask, pose, intrinsics = (
            torch.stack(ts, dim=0)
            for ts in zip(
                #*((t.to(device) for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
                *((t for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
            )
        )
        rgb = rgb.permute(0, 2, 3, 1)
        xyz = get_xyz(depth, mask, pose, intrinsics)
        #The resulting tensor will be True only where both ~mask is True (i.e., the original mask was False, high confidence depth value) and the random value exceeds the threshold, a kind of sampling of points to keep
        mask = (~mask & (torch.rand(mask.shape, device=mask.device) > threshold)) #depth reading confidence masking
        # mask = (~mask & (torch.rand(mask.shape) > threshold)) 
        rgb, xyz = rgb[mask.squeeze(1)], xyz[mask.squeeze(1)]
        rgbs.append(rgb.detach().cpu())
        xyzs.append(xyz.detach().cpu())
    
    xyzs = torch.vstack(xyzs)
    rgbs = torch.vstack(rgbs)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs)
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs)
    if downsample:
        vl_size = 0.02
        print(f"Downsampling pointcloud || voxel_size: {vl_size}... ")
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=vl_size)
    else:
         print("Skipped downsampling ... ")
    return merged_pcd


def get_pointcloud_from_graph_r3d(graph: nx.Graph, chunk_size: int = 16, threshold: float = 0.9, downsample=False) -> o3d.geometry.PointCloud:
    #threshold determines number of points to be included in the pointcloud
    """Generates a 3D point cloud from a NetworkX observation graph.

    Args:
        graph: The observation graph with nodes containing 'rgb_tensor', 'depth_data', 'pose', 'mask', and 'intrinsics'.
        chunk_size: Process this many nodes from the graph at a time.
        threshold: Confidence threshold for depth values.
        downsample: Whether to downsample the point cloud.

    Returns:
        A 3D point cloud.
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nodes = list(graph.nodes(data=True))
    num_nodes = len(nodes)
    xyzs = []
    rgbs = []

    for chunk_start in range(0, num_nodes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_nodes)
        chunk_nodes = nodes[chunk_start:chunk_end]

        rgb, depth, mask, pose, intrinsics = (
            torch.stack([node_data[attr] for _, node_data in chunk_nodes], dim=0).to(device)
            for attr in ['rgb_tensor', 'depth_data', 'depth_confidence_mask', 'pose_matrix', 'intrinsics']
        )

        rgb = rgb.permute(0, 2, 3, 1)
        xyz = get_xyz(depth, mask, pose, intrinsics)
        #The resulting tensor will be True only where both ~mask is True (i.e., the original mask was False, high confidence depth value) and the random value exceeds the threshold, a kind of sampling of points to keep
        mask = (~mask & (torch.rand(mask.shape, device=mask.device) > threshold)) # depth reading confidence masking
        rgb, xyz = rgb[mask.squeeze(1)], xyz[mask.squeeze(1)]
        rgbs.append(rgb.detach().cpu())
        xyzs.append(xyz.detach().cpu())

    xyzs = torch.vstack(xyzs)
    rgbs = torch.vstack(rgbs)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs.numpy())
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs.numpy())

    if downsample:
        vl_size = 0.02
        print(f"Downsampling pointcloud || voxel_size: {vl_size}... ")
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=vl_size)
    else:
        print("Skipped downsampling ... ")

    return merged_pcd

# Define the function for creating the point cloud
def get_pointcloud_from_graph_robot(graph: nx.Graph, chunk_size: int = 16, threshold: float = 0.9, downsample=False) -> o3d.geometry.PointCloud:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    nodes = list(graph.nodes(data=True))
    num_nodes = len(nodes)
    total_pcds = []
    total_colors = []

    for chunk_start in tqdm(range(0, num_nodes, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, num_nodes)
        chunk_nodes = nodes[chunk_start:chunk_end]

        rgbs, depths, poses = [], [], []

        for _, node_data in chunk_nodes:
                for i in range(4):
                    rgb = node_data['rgb_tensor'][i]
                    depth = torch.from_numpy(node_data['depth_data'][i]).float()
                    pose = torch.from_numpy(node_data['pose_matrix'][i]).float()
                    rgbs.append(rgb)
                    depths.append(depth)
                    poses.append(pose)
        
        print("datapoint count in chunk: ", len(rgbs))
        rgbs = torch.stack(rgbs).permute(0, 2, 3, 1).to(device)  # (B, H, W, 3)
        depths = torch.stack(depths).to(device)  # (B, 1, H, W)
        poses = torch.stack(poses).to(device)  # (B, 4, 4)

        # Use the previously defined function to get 3D coordinates
        xyz = spot_pixel_to_world_frame_batched(depths, poses)
        
        # Mask out low-confidence depth points and apply random sampling
        mask = (depths > 0) & (torch.rand(depths.shape, device=device) > threshold)
        rgbs, xyz = rgbs[mask.squeeze(1)], xyz[mask.squeeze(1)]
        
        total_colors.append(rgbs.cpu())
        total_pcds.append(xyz.cpu())

    total_pcds = torch.vstack(total_pcds)
    total_colors = torch.vstack(total_colors)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(total_pcds.numpy())
    pcd_o3d.colors = o3d.utility.Vector3dVector(total_colors.numpy())

    if downsample:
        vl_size = 0.02
        print(f"Downsampling pointcloud || voxel_size: {vl_size}... ")
        pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=vl_size)
    else:
        print("Skipped downsampling ... ")

    return pcd_o3d