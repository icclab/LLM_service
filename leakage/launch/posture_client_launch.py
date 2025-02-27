from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_info_topic', default_value='/summit/oak/rgb/camera_info'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/summit/oak/stereo/image_raw'),
        DeclareLaunchArgument('pose_topic', default_value='/summit/odom'),
        
        Node(
            package='llm',
            executable='ros2client-posture-summit.py',
            name='posture_detection_client',
            output='screen',
            parameters=[
                {'camera_info_topic': '/summit/oak/rgb/camera_info'},
                {'image_topic': '/image_raw'},
                {'depth_topic': '/summit/oak/stereo/image_raw'},
                {'pose_topic': '/summit/odom'},
            ]
        ),
    ])
