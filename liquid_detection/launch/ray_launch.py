from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='liquid_detection',  
            executable='water',  
            name='ray_water',
            output='screen',
            remappings=[
                ('/tf', '/summit/tf'),
                ('/tf_static', '/summit/tf_static'),
                ('/leakage_marker', '/summit/leakage_marker'),
            ],
            parameters=[
                {'world_frame': 'map'},
                {'use_sim_time': True},
                {'pixel_topic': '/leakage_pixel_coords'},
                {'camera_info_topic': '/summit/color/camera_info'},
                {'output_topic': '/leakage_ground_point'},
                {'tf_lookup_timeout_sec': 1.0},]
        ),

        Node(
            package='liquid_detection',  
            executable='pixel_water',  
            name='pixel_water',
            output='screen',
            remappings=[
                ('/leakage_annotated_image', '/summit/leakage_annotated_image'),
            ],
            parameters=[
                {'image_topic': '/summit/color/image'},
                {'camera_info_topic': '/summit/color/camera_info'},]
        ),
    ])
