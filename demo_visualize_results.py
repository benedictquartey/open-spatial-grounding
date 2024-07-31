from osg.utils.visualization_utils import load_viz_data, place_elements_in_scene, get_all_mask3dpositions, color_coded_heatmap, plot_top_down_view, visualize
import argparse
import open3d as o3d

# Eg running code
# python demo_visualize_results.py --result_dir results --rsm --show_elements --remove_roof --top_down --r3ddata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualize_results", description='Visualize semantic map, referents and plan')
    parser.add_argument('--result_dir', default='output_lab', type=str, help='This parameter indicates which folder contains files.')
    parser.add_argument('--elements_filename', default='filtered_relevant_elements_mindetails' , type=str, help='This parameter indicates which elements file to use')
    parser.add_argument('--show_elements', action='store_true', default = False, help='This parameter indicates whether to show backprojected center 3d positions for elements.')
    parser.add_argument('--rsm', action='store_true', default = False, help='This parameter indicates whether to show backprojected mask 3d positions [Segmentation view].')
    parser.add_argument('--remove_roof', action='store_true', default = False, help='This parameter indicates whether to filter out roof points.')
    parser.add_argument('--top_down', action='store_true', default = False, help='This parameter indicates whether to show jsut top down view.')
    parser.add_argument('--save_pointcloud', action='store_true', default = False, help='This parameter indicates whether to save the pointcloud to disk.')
    parser.add_argument('--show_origin', action='store_true', default = False, help='This parameter indicates whether to show the origin coordinate used for spatial reasoning.')
    parser.add_argument('--r3ddata',  action='store_true', default = False, help='This parameter indicates that r3d data is being used.')
    parser.add_argument('--robotdata',  action='store_true', default = False, help='This parameter indicates that robot data is being used.')
    args = parser.parse_args()

    ##load data
    pcd,elements,_,_ = load_viz_data(args.result_dir,args.elements_filename,use_augmented=False)
    items_to_visualize = []

    ## Downsample Pointcloud
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    
    if elements is not None:
        if args.show_elements:
            elements_centerpoints3d = place_elements_in_scene(elements,show_origin=args.show_origin)
            items_to_visualize += elements_centerpoints3d

        if args.rsm:
            if args.robotdata:
                mask_points_of_interest = get_all_mask3dpositions(elements,data_src="robot") 
            elif args.r3ddata:
                mask_points_of_interest = get_all_mask3dpositions(elements,data_src="r3d") 
            pcd = color_coded_heatmap(pcd, mask_points_of_interest,threshold_percentage=1,kdtree_search_radius=0.1)
        
    if args.top_down:                               ## Visualize Top-down View of the Point Cloud
        plot_top_down_view(pcd, z_threshold=0.5)
    else:                                           ## Visualize Point Cloud in 3D
        h_min_bottom= -3
        h_max_top= 1 #1.5 keep more of height default || see doors: 0.7 || Old value:-0.15
        if args.remove_roof:
            vis_pcd, extra_items = visualize(pcd, items_to_visualize, h_min_bottom, h_max_top, filter_top_bottom=True)
        else:
            vis_pcd, extra_items = visualize(pcd, items_to_visualize, h_min_bottom, h_max_top, filter_top_bottom=False)

    # Write pointcloud to disk
    if args.save_pointcloud:
        o3d.io.write_point_cloud("saved_pointcloud.pcd", vis_pcd)
        print("Pointcloud written to disk")




