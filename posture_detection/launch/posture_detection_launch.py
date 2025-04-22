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
                {'image_topic': '/summit/summit/color/image'},
                {'marker_topic': '/summit/summit/person_marker'},
                ]
        ),
    ])
