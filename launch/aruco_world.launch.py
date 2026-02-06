import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro

def generate_launch_description():

    robotName = 'marker_follower_robot'
    packageName = 'aruco_marker_robot'
    modelFilePath = 'urdf/marker_follower_robot.urdf'

    # Get paths
    pathModelFile = os.path.join(get_package_share_directory(packageName), modelFilePath)
    # This world file should now contain the linear marker arrangement
    worldFilePath = os.path.join(get_package_share_directory(packageName), 'worlds', 'marker_world.sdf')
    
    # Process URDF
    robotDescription = xacro.process_file(pathModelFile).toxml()

    # Launch Gazebo with your linear marker world
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    )

    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch, 
        launch_arguments={
            'gz_args': f'-r -v4 {worldFilePath}',  
            'on_exit_shutdown': 'true'
        }.items()  
    )

    # Spawn robot at the starting position (x=0, y=0) facing the line of markers
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robotName,
            '-topic', 'robot_description',
            "-x", "0.0", "-y", "0.0", "-z", "0.1" # Starting at origin
        ],
        output='screen',
    )

    # Robot State Publisher
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robotDescription,
            'use_sim_time': True
        }]
    )

    ekf_params = os.path.join(
        get_package_share_directory(packageName),
        'config',
        'ekf.yaml'
    )

    # EKF Node is crucial here for tracking translational movement along the line
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_params,
            {'use_sim_time': True}
        ]
    )

    bridge_params = os.path.join(
        get_package_share_directory(packageName),
        'config',
        'bridge_parameters.yaml'
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',  
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}'
        ],
        output='screen', 
    )

    # Image Bridge for visual servoing processing
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image"],
        output="screen",
        parameters=[{
            'use_sim_time': True,
            'camera.image.compressed.jpeg_quality': 75  
        }],
    )

    # Relay node for camera_info to support distance estimation logic if needed
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['/camera/camera_info', '/camera/image/camera_info'],
        parameters=[{'use_sim_time': True}]
    )

    rviz_config_path = os.path.join(
        get_package_share_directory(packageName),
        'rviz',
        'visualize.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}]
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(gazeboLaunch)
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)
    launchDescriptionObject.add_action(ekf_node)
    launchDescriptionObject.add_action(gz_image_bridge_node)  
    launchDescriptionObject.add_action(relay_camera_info_node) 
    launchDescriptionObject.add_action(rviz_node)

    return launchDescriptionObject
