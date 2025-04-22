#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from llm.srv import CheckPosture
import time
import cv2
import numpy as np
from visualization_msgs.msg import Marker
import tf2_ros
from tf2_ros import Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer

class PostureDetectionClient(Node):
    def __init__(self):
        super().__init__('posture_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/summit/summit/color/image')
        self.declare_parameter('marker_topic', '/summit/summit/person_marker')

        # Get parameters from the launch file
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value

        self.client = self.create_client(CheckPosture, 'check_posture')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for posture service...')

        self.bridge = CvBridge()
        self.person_detected = False  
        self.request_in_progress = False  
        self.pending_requests = {}  

        # TF2 setup for coordinate transformation
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.image_subscription = Subscriber(self, ROSImage, image_topic)
        self.marker_subscription = Subscriber(self, Marker, marker_topic)

        # Synchronizer to match messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.marker_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)    

        # Publishers
        self.posture_publisher = self.create_publisher(String, '/posture', 10)
        self.marker_pub = self.create_publisher(Marker, '/posture_marker', 10)

        # Latest data storage
        self.latest_image = None
        self.latest_pose = None

    def synchronized_callback(self, ros_image: ROSImage, marker_msg: Marker):
        if self.request_in_progress:
            #self.get_logger().warn("Previous request still in progress, skipping new request.")
            return

        self.latest_image = ros_image
        self.latest_pose = marker_msg.pose

        self.get_logger().info(f"Synchronized image and marker received.")

        self.send_posture_request(self.latest_pose)
        
    def send_posture_request(self, person_position):
        if self.latest_image is None or self.request_in_progress:
            return

        request = CheckPosture.Request()
        request.image = self.bridge.cv2_to_imgmsg(self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8'), 'bgr8')

        self.request_in_progress = True
        future = self.client.call_async(request)
        future.add_done_callback(lambda f: self.handle_response(f, person_position))

    def handle_response(self, future, person_position):
        """ Handles the response from posture service. """
        try:
            response = future.result()
            posture_code = response.posture_detected
            posture_string = {0: "nothing found", 1: "standing", 2: "sitting", 3: "lying"}.get(response.posture_detected, "unknown posture")

            posture_msg = String()
            posture_msg.data = posture_string
            
            self.posture_publisher.publish(posture_msg)
            self.get_logger().info(f"Posture Detected: {posture_string} at {person_position}")

            # Publish corresponding marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = person_position
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            marker.color.a = 1.0

            if posture_code == 1:  # standing - green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif posture_code == 2:  # sitting - yellow
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif posture_code == 3:  # lying - pink
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0

            self.marker_pub.publish(marker)

        except Exception as e:
            self.get_logger().error(f"Posture Service Error: {str(e)}")

        self.request_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    client = PostureDetectionClient()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
