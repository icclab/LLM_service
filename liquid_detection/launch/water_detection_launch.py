from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='liquid_detection',  
            executable='roboflow',  
            name='roboflow_water',
            output='screen',
            remappings=[
                ('/tf', '/summit/tf'),
                ('/tf_static', '/summit/tf_static'),
                ('/leakage_marker', '/summit/leakage_marker'),
                ('/leakage_annotated_image', '/summit/leakage_annotated_image'),
            ],
            parameters=[
                {'image_topic': '/summit/color/image'},
                {'depth_topic': '/summit/stereo/depth'},
                {'camera_info_topic': '/summit/color/camera_info'},]
        ),
    ])
