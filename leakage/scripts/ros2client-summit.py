#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import String
from cv_bridge import CvBridge
from llm.srv import CheckPosture
from llm.msg import PoseWithString
import time
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from message_filters import Subscriber, ApproximateTimeSynchronizer

class PostureDetectionClient(Node):
    def __init__(self):
        super().__init__('posture_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('camera_info_topic', '/summit/oak/rgb/camera_info')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('depth_topic', '/summit/oak/stereo/image_raw')
        self.declare_parameter('pose_topic', '/summit/base_pose')

        # Get parameters from the launch file
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value

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

        # Camera model
        self.camera_model = PinholeCameraModel()

        # Subscriptions
        self.yolo_subscription = self.create_subscription(YoloResult, '/yolo_result', self.yolo_cb, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        self.image_subscription = Subscriber(self, ROSImage, image_topic)
        self.depth_subscription = Subscriber(self, ROSImage, depth_topic)
        self.pose_subscription = Subscriber(self, PoseStamped, pose_topic)

        # Synchronizer to match messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.depth_subscription, self.pose_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)    

        # Publishers
        self.posture_pose_publisher = self.create_publisher(Pose, '/posture_pose_summit', 10)
        # self.posture_publisher = self.create_publisher(String, '/posture', 10)
        # self.person_pose_pub = self.create_publisher(PoseStamped, '/person_pose_camera', 10)
        self.person_pose_pub = self.create_publisher(PoseWithString, '/person_pose_with_posture_camera', 10)

        # Latest data storage
        self.latest_image = None
        self.latest_depth = None
        self.latest_pose = None
        self.request_in_progress = False  

    def camera_info_callback(self, camera_info_msg):
        """ Initialize camera model from CameraInfo. """
        self.camera_model.fromCameraInfo(camera_info_msg)

    def synchronized_callback(self, ros_image: ROSImage, depth_image: ROSImage, pose_msg: PoseStamped):
        if self.request_in_progress:
            #self.get_logger().warn("Previous request still in progress, skipping new request.")
            return  # Skip sending a new request

        self.latest_image = ros_image
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_image)
        self.latest_pose = pose_msg.pose
        
        self.get_logger().info(f"Synchronized data: Image/Depth/Pose")

    def yolo_cb(self, msg):
        """ Handles YOLO detections, extracts person position, and transforms it to the map frame. """
        if not msg.detections.detections:
            return

        for detection in msg.detections.detections:
            if not detection.results:
                continue  

            if detection.results[0].hypothesis.class_id == "person":
                #self.get_logger().info("Person detected!")

                bbox = detection.bbox
                center_x, center_y = int(bbox.center.position.x), int(bbox.center.position.y)
                size_x, size_y = int(bbox.size_x), int(bbox.size_y)

                if self.latest_depth is None:
                    # self.get_logger().warn("Depth image not available yet!")
                    continue

                # Extract depth using bounding box
                avg_depth = self.estimate_depth(center_x, center_y, size_x, size_y)

                if avg_depth is not None:
                    person_position = self.pixel_to_3d(center_x, center_y, avg_depth)
                    if person_position is not None:
                        self.send_posture_request(person_position)
                #         personpose = PoseStamped()
                #         personpose.header.stamp = self.get_clock().now().to_msg()
                #         personpose.header.frame_id = 'oak_rgb_camera_optical_frame'
                #         personpose.pose.position.x = person_position[0]
                #         personpose.pose.position.y = person_position[1]
                #         personpose.pose.position.z = person_position[2]
                #         self.get_logger().info(f"Person Position (Camera Frame): {personpose}")

                #         self.person_pose_pub.publish(personpose)

                # self.person_detected = True
                # self.process_posture()
                # return  

        # self.person_detected = False
    def send_posture_request(self, person_position):
        if self.latest_image is None or self.request_in_progress:
            return

        request = CheckPosture.Request()
        request.image = self.bridge.cv2_to_imgmsg(self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8'), 'bgr8')

        self.request_in_progress = True
        future = self.client.call_async(request)
        future.add_done_callback(lambda f: self.handle_response(f, person_position))


    def estimate_depth(self, center_x, center_y, size_x, size_y):
        """ Estimates average depth of person using bounding box pixels. """
        if self.latest_depth is None:
            return None

        x_min, y_min = max(0, center_x - size_x // 2), max(0, center_y - size_y // 2)
        x_max, y_max = min(self.latest_depth.shape[1], center_x + size_x // 2), min(self.latest_depth.shape[0], center_y + size_y // 2)

        sample_pixels = []
        for i in range(3):
            for j in range(3):
                x = x_min + i * (size_x // 3)
                y = y_min + j * (size_y // 3)
                depth_value = self.latest_depth[y, x] if 0 <= x < self.latest_depth.shape[1] and 0 <= y < self.latest_depth.shape[0] else 0
                if depth_value > 0:
                    sample_pixels.append(depth_value)

        return np.median(sample_pixels) if sample_pixels else None

    def pixel_to_3d(self, x, y, depth):
        """ Converts pixel coordinates to 3D position in the camera frame. """
        if depth <= 0 or not self.camera_model.fx():
            return None

        ray = np.array(self.camera_model.projectPixelTo3dRay((x, y)))
        return ray * (depth / ray[2])

    # def process_posture(self):
    #     """ Sends image for posture analysis. """
    #     if self.latest_image is None or self.latest_pose is None or self.request_in_progress:
    #         return

    #     self.get_logger().info('Checking posture...')

    #     request = CheckPosture.Request()
    #     request.image = self.bridge.cv2_to_imgmsg(self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8'), 'bgr8')

    #     self.request_in_progress = True
    #     future = self.client.call_async(request)
    #     self.pending_requests[future] = self.latest_pose

    #     future.add_done_callback(self.handle_response)

    def handle_response(self, future, person_position):
        """ Handles the response from posture service. """
        try:
            # pose = self.pending_requests.pop(future, None)
            # if pose is None:
            #     return

            response = future.result()
            posture_string = {0: "nothing found", 1: "standing", 2: "sitting", 3: "lying"}.get(response.posture_detected, "unknown posture")

            pose_with_string = PoseWithString()
            pose_with_string.posture = posture_string
            pose_with_string.pose.header.stamp = self.get_clock().now().to_msg()
            pose_with_string.pose.header.frame_id = 'oak_rgb_camera_optical_frame'
            pose_with_string.pose.position.x = person_position[0]
            pose_with_string.pose.position.y = person_position[1]
            pose_with_string.pose.position.z = person_position[2]

            self.person_pose_pub.publish(pose_with_string)
            self.get_logger().info(f"Posture Detected: {posture_string} at {person_position}")


            # self.posture_publisher.publish(String(data=posture_string))
            # if response.posture_detected:
            #     self.posture_pose_publisher.publish(pose)
            #     #self.get_logger().info(f"Posture: {posture_string} at Position: {pose.position}")

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
