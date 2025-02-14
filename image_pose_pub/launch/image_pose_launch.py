from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_pose_pub',
            executable='pose_image_pub',
            name='image_pose',
            output='screen',
        )
    ])
