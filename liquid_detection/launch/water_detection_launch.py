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
                {'image_topic': '/summit/oak/rgb/image_rect'},
                {'gps_topic': '/summit/gps/fix'},
                {'local_topic': '/summit/base_pose'},]
        ),
    ])
