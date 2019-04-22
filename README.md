[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

[//]: # (Image References)

[original raw image]: ./image/original_point_cloud.png
[RANSAC objects]: ./image/RANSAC_objects.png
[cluster cloud]: ./image/cluster_cloud.png
[object recognition world1]: ./image/object_recognition_world1.png
[object recognition world2]: ./image/object_recognition_world2.png
[object recognition world3]: ./image/object_recognition_world3.png
[original unfiltered image]: ./image/original_unfiltered.png
[outlier filtering]: ./image/outlier_filtering.png
[passthrough x]: ./image/passthrough_x.png
[passthrough y]: ./image/passthrough_y.png
[passthrough z]: ./image/passthrough_z.png
[test1 training result]: ./image/test1_training_result1.png
[test2 training result]: ./image/test2_training_result.png
[test3 training result]: ./image/test3_training_result.png
[test1 training normalized]: ./image/test1_training_result_normalized.png
[test2 training normalized]: ./image/test2_training_result_normalized.png
[test3 training normalized]: ./image/test3_training_result_normalized.png
[voxel filtering]: ./image/voxel_downsample_filtering.png
[gazebo 1]: ./image/gazebo_01.jpg
[gazebo 2]: ./image/gazebo_02.jpg




# 3D Perception Project with PR2 Robot
In this project, PR2 robot identifies objects on a table, picks and places into the desginated bins. 
The robot is equipped with a RGB-D camera that reads in the noisy point cloud data. It is processed using various filtering techniques and clustering technique to have each object separated and clustered. 
To identify individual object, machine learning is used to train the object recognition model and to predict the given clustered object data. 
Lastely, pick and place task is completed to sort the objects into the corresponding dropboxes. 


# 1. 3D Perception Pipeline
The raw point cloud data from the RGB-D camera includes all the objects in the space including the table top objects, the table, and floor. To exclude only the tabletop objects, various filtering techniques are used. Then the objects are separated into the individual clusters using the clustering technique. 
![original unfiltered image][original unfiltered image] 
## Statistical Outlier Filtering
Often time, the data we get from an image source contains noises. One of the methods of eliminating these outliers is to use a statistical analysis on the neighboring points and disregard the points that do not meet certain criteria. In this proejct, PCL's Statistical Outlier Removal filter is used.
```python
    # Create a statistical outlier filter object for input point cloud
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point    
    outlier_filter.set_mean_k(5)
    # Set threshold scale factor
    x=0.00001
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Call the filter function 
    cloud_filtered = outlier_filter.filter()
    cloud_outlier = cloud_filtered
```
From trial and error, the number of neighboring points to analyze is set to 5 and the threshold value is set to 0.00001. 
The below image shows the result of the outlier filtering.
![outlier filtering][outlier filtering]

## VoxelGrid Downsampling
Running compution on a full resoluution cloud data is demanding and could slow the process. As a remedy, sparsely sampled data is used instead. In particular, VoxelGrid Downsampling filter is used to downsample the point cloud without a loss of important features. 

```python
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.005   
    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
```
![voxel downsampling][voxel filtering]

## Pass Through Filter
It is possible to downsample further if the information about the location of the target in the scene is known ahead by 'cropping' out the unnecessary point cloud data using a Pass Through Filter. 
The filter is applied a total of 3 times, each along Z, Y, and X axis. 
Along Z-axis
```python
    # Create a PassThrough filter object. 
    # Apply Z axis first
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object. 
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.605
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
```
![z axis][passthrough z]
Along Y-axis
```python
    # Apply Y axis
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object. 
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.45
    axis_max = 0.45
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
```
![y axis][passthrough y]
Along X-axis
```python
    # Create a PassThrough filter object 
    # Apply X axis
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object. 
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.35
    axis_max = 0.8
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
```
![x axis][passthrough x]
## RANSAC Plane Segmentation 
With the prior knowledge on the scene, the tabletop objects can be isoloated from the table using a segmentation technique. RANSAC(Random Sample Consensus) is a popular algorithm to identify points in the dataset that belongs to a particualr model. The plane model is chosen to identify the table from the scene, which is then extracted from the point cloud to leave only the tabletop objects.

```python
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.006
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    # Extract inliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    # Extract outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
```
![ransac objects][RANSAC objects]
## Euclidean Clustering
Now that we have separated the tabletop objects from the scene, we need to cluster the point cloud into each object. 
```python
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold as well as min/max cluster size
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(5000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
```
To visualize the result of above clustering, each cluster is masked with different color
```python
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene   
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
```
![clustering][cluster cloud]
# 2. Object Recognition from Machine Learning Model
As a part of object recognition, one of the supervised machine learning algorithms called Support Vecotr Machine (SVM) is used. Training of SVM model is done using the color features such as HSV values and the geometric features such as surface normals are extracted from the generated good data. 
## Color Feature Extraction
Histograms of HSV values are extracted using the following function.
```python
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # Compute histograms
    nbins = 64
    bins_range = (0,256)
    ch1_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    ch2_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    ch3_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0])).astype(np.float64)
    normed_features = hist_features/np.sum(hist_features)

    return normed_features
```
## Geometric Feature Extraction
Histograms of surface normal vectors are extracted using the following function.
```python
def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values (just like with color)
    nbins = 64
    bins_range = (0,256)
    x_hist = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    y_hist = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
    z_hist = np.histogram(norm_z_vals, bins=nbins, range=bins_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)
    normed_features = hist_features/np.sum(hist_features)

    return normed_features
```
## SVM 
Different parameters for the SVM are tested, and the final training is done using the following parameters

Parameters | Values
--- | ---: 
Kernel | rbf
C | 1.0
gamma | scale

The results of the training for 3 world settings are shown below
### World 1 Training Results
Test 1 | Values
--- | ---: 
Features in Training Set | 900
Invalid Features in Training Set | 0
Scores | [0.983  0.966  0.977  0.961  0.961]
Accuracy | 0.97 (+/- 0.02)
Accuracy Score | 0.97

![world1 training][test1 training result]
![world1 training normalized][test1 training normalized]

### World 2 Training Results
Test 2 | Values
--- | ---: 
Features in Training Set | 2500
Invalid Features in Training Set | 4
Scores | [0.934  0.921  0.925  0.917  0.937]
Accuracy | 0.93 (+/- 0.01)
Accuracy Score | 0.93

![world2 training][test2 training result]
![world2 training normalized][test2 training normalized]

### World 3 Training Results
Test 3 | Values
--- | ---: 
Features in Training Set | 4000
Invalid Features in Training Set | 10
Scores | [0.926  0.917  0.904  0.909  0.893]
Accuracy | 0.91 (+/- 0.02)
Accuracy Score | 0.91

![world3 training][test3 training result]
![world3 training normalized][test3 training normalized]

## Object Recognition
Now we have our SVM models trained, we can implement these into our 3D perception pipeline to identify each object from the detected clusters. The color and geometric features are extracted from the detected cluster, and the SVM model predicts which object it is. Below shows the object recognition results for different testing worlds.

### World 1 Object Recognition Results
![world1 object recognition][object recognition world1]
All 3 objects are successfully identified!
### World 2 Object Recognition Results
![world2 object recognition][object recognition world2]
All 5 objects are successfully identified!
### World 3 Object Recognition Results
![world3 object recognition][object recognition world3]
All 8 objects are successfully identified!

# 3. Pick and Place
As an additional challenge to the object recognition, the final task is to relocate the identified objects into the designated dropboxes. To pick up the object, its centroid from the cluster cloud is calculated and the corresponding path is planned via inverse kinematics. 
![gazebo01][gazebo 1]
![gazebo02][gazebo 2]



