from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_info_topic', default_value='/drone/zed/zed_node/rgb/camera_info'),
        DeclareLaunchArgument('image_topic', default_value='/drone/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/drone/zed/zed_node/depth/depth_registered'),
        DeclareLaunchArgument('pose_topic', default_value='/drone/zed/zed_node/odom'),
        
        Node(
            package='llm',
            executable='ros2client-leakage.py',
            name='leakage_detection_client',
            output='screen',
            parameters=[
                {'camera_info_topic': '/drone/zed/zed_node/rgb/camera_info'},
                {'image_topic': '/drone/image_raw'},
                {'depth_topic': '/drone/zed/zed_node/depth/depth_registered'},
                {'pose_topic': '/drone/zed/zed_node/odom'},
            ]
        ),
    ])
