# Experiment Setup
**Environment Setup:** Use Record3D to scan the environments. Recording should include: 
* All Obstacles in environments
* Complete floor where the robots can navigate
* All testing objects.

**Scan the Environment:** After positioning the objects and tapes, proceed to scan the environment and save the Record3D r3d file in the `ok-robot-navigation/r3d/` folder. If you just want to try out a sample r3d file, you can use `ok-robot-navigation/r3d/sample.r3d`.

**Extract PointCloud:** Then, use the `ok-robot-navigation/get_point_cloud.py` script to extract the pointcloud of the scene, which will be stored as `ok-robot-navigation/pointcloud.ply`. 
```
python get_point_cloud.py --input_file=[your r3d file]
```
