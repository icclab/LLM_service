from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_pose_pub',  
            executable='image_decompressor',  
            name='image_decompressor',
            output='screen',
            parameters=[
                {'compressed_image_topic': '/summit/oak/rgb/image_rect/compressed'},
                {'raw_image_topic': '/image_raw'} ]
        ),
        
        Node(
            package='image_pose_pub',
            executable='pose_image_pub',
            name='image_pose',
            output='screen',
        )
    ])
