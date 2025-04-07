from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='posture_detection',  
            executable='server',  
            name='posture_detection_server',
            output='screen',

        ),
    ])
