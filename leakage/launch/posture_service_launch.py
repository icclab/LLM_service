from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([     
        Node(
            package='llm',
            executable='ros2service-posture.py',
            name='posture_detection_service',
            output='screen',
        ),
    ])
