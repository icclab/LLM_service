from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([     
        Node(
            package='llm',
            executable='ros2service.py',
            name='ros2_service',
            output='screen',
        ),
    ])
