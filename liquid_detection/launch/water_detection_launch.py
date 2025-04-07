from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='liquid_detection',  
            executable='roboflow',  
            name='roboflow_water',
            output='screen',
            parameters=[
                {'image_topic': '/summit/summit/oak/rgb/image_rect/compressed'},
                {'gps_topic': '/summit/gps'},
                {'local_topic': '/summit/summit/mavros/local_position/pose'},]
        ),
    ])
