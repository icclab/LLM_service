from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='posture_detection',  
            executable='posture',  
            name='posture_detection_client',
            output='screen',
            parameters=[
                {'image_topic': '/summit/oak/rgb/image_rect'},
                {'marker_topic': '/visualization_marker'},
                {'local_topic': '/summit/base_pose'},]
        ),
    ])
