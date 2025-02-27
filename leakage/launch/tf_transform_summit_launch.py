from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([     
        Node(
            package='llm',
            executable='tf_transform_marker.py',
            name='camera_to_map',
            output='screen',
        ),
    ])
