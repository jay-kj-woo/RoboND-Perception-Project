#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    pcl.save(cloud,'original_cloud.pcd')
    ###########################################################
    # Statistical Outlier Filtering
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
    pcl.save(cloud_filtered,'stat_outlier_filtered.pcd')
    pcl_testing.publish(pcl_to_ros(cloud_outlier))
    #########################################################
    # Voxel Grid Downsampling
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.005   
    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    pcl.save(cloud_filtered,'voxel_downsampled.pcd')
    #################################################################
    # PassThrough Filter
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
    # Create a PassThrough filter object 
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
    pcl.save(cloud_filtered,'passthrough.pcd')
    #####################################################################
    # RANSAC Plane Segmentation
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
    #############################################
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
    #####################################################################
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
    ######################################################################
    # Convert PCL data to ROS messages
    ros_cloud = pcl_to_ros(cloud_filtered)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    
    # Publish ROS messages
    pcl_pub.publish(pcl_msg)
    pcl_cloud_pub.publish(ros_cloud)
    pcl_cloud_table_pub.publish(ros_cloud_table)
    pcl_cloud_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_cloud_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects = []
    detected_objects_labels = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals) 
        feature = np.concatenate((chists,nhists))
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    if len(detected_objects) > 0:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else:
        rospy.logwarn("No object is detected") 
    return

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    labels = []
    centroids = []      # to be list of tuples 
    dict_list=[]

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    # Parse parameters into individual variables
    test_scene_num.data = 3
    
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3]) 

#    # Rotate PR2 in place to capture side tables for the collision map
#    pr2_base_rotation_pub.publish(1.57)
#    rospy.sleep(20)
#    pr2_base_rotation_pub.publish(-1.57)
#    rospy.sleep(40)
#    pr2_base_rotation_pub.publish(0)
#    rospy.sleep(20)

    # Loop through the pick list
    for idx in range(0,len(object_list_param)):
        object_name.data = object_list_param[idx]['name']
        object_group = object_list_param[idx]['group']
   
        # Get the PointCloud for a given object and obtain it's centroid
        try:
            index = labels.index(object_name.data)
        except ValueError:
            print ("The pick list object, ",object_name.data," is not detected. Moving on to the next pick list item")
            continue
        
        pick_pose.position.x = np.asscalar(centroids[index][0])
        pick_pose.position.y = np.asscalar(centroids[index][1])
        pick_pose.position.z = np.asscalar(centroids[index][2])
    
        # Seacrh the corresponding dictionary from the list of dictionaris and extract the drop box position and the side of arm
        matching_dict = [dictionary for dictionary in dropbox_param if dictionary['group']==object_group][0]
        # Create 'place_pose' for the object
        place_pose.position.x = matching_dict['position'][0] #+ idx*0.01
        place_pose.position.y = matching_dict['position'][1] #+ idx*0.01
        place_pose.position.z = matching_dict['position'][2] 
        # Assign the arm to be used for pick_place
        arm_name.data = matching_dict['name']

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    yaml_filename = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception_pipeline', anonymous=True)	
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # Create Publishers
    pcl_pub = rospy.Publisher("/pcl_unfiltered", PointCloud2, queue_size=1)
    pcl_cloud_pub = rospy.Publisher("/pcl_cloud_filtered", PointCloud2, queue_size=1)
    pcl_cloud_table_pub = rospy.Publisher("/pcl_cloud_table", PointCloud2, queue_size=1)
    pcl_cloud_objects_pub = rospy.Publisher("/pcl_cloud_objects", PointCloud2, queue_size=1)
    pcl_cluster_cloud_pub = rospy.Publisher("/pcl_cluster_cloud", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pr2_base_rotation_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size =10)
    pcl_testing = rospy.Publisher("/pcl_filter_testing", PointCloud2, queue_size=1)
    # Load Model From disk
    model = pickle.load(open('model_world3.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
