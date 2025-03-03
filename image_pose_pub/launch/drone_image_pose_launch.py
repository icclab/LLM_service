from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_pose_pub',  
            executable='image_decompressor',  
            name='image_decompressor_summit',
            output='screen',
            parameters=[
                {'compressed_image_topic': '/summit/summit/oak/rgb/image_rect/compressed'},
                {'raw_image_topic': '/summit/image_raw'} ]
        ),
        
        Node(
            package='image_pose_pub',  
            executable='image_decompressor',  
            name='image_decompressor_drone',
            output='screen',
            parameters=[
                {'compressed_image_topic': '/drone/zed/zed_node/rgb/image_rect_color/compressed'},
                {'raw_image_topic': '/drone/image_raw'} ]
        ),
        
        # Node(
        #     package='image_pose_pub',
        #     executable='pose_image_pub',
        #     name='image_pose',
        #     output='screen',
        # )
    ])
