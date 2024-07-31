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

def get_pointcloud(ds: Dataset[PosedRGBDItem], chunk_size: int = 16, threshold:float = 0.9, downsample=False) -> Iterator[tuple[Tensor, Tensor]]:
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


def get_pointcloud_from_graph(graph: nx.Graph, chunk_size: int = 16, threshold: float = 0.9, downsample=False) -> o3d.geometry.PointCloud:
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
