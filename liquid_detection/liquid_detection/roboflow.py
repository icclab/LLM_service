import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import NavSatFix, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import cv2
import supervision as sv
from inference import get_model
from inference_sdk import InferenceHTTPClient
import os
import numpy as np
from tf2_ros import Buffer, TransformListener
from image_geometry import PinholeCameraModel
from tf2_geometry_msgs import do_transform_point

class LeakageDetection(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/summit/color/image')
        self.declare_parameter('camera_info_topic', '/summit/color/camera_info')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10)

        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.image_subscription = self.create_subscription(ROSImage, image_topic, self.image_callback, 10)
        
        self.point_pub = self.create_publisher(PointStamped, '/leakage_pixel_coords', 10)
        self.annotated_image_pub = self.create_publisher(ROSImage, '/leakage_annotated_image', 10)
        
        self.model_client = InferenceHTTPClient(
            api_url="https://roboflow.nephele-project.ch",
            api_key="wrJRdyQTYXns1D8MwoxW",
        )

        self.models = [
            ("water-leakage/2", sv.Color(255, 0, 0)),  # Red
            # ("water-ba8zz/1", sv.Color(0, 255, 0)),  # Green
            # ("spills-ax5xv/2", sv.Color(0, 0, 255))  # Blue
        ]

        self.original_width = 0
        self.original_height = 0

    def camera_info_callback(self, msg):
        """ Initialize camera model from CameraInfo. """
        self.original_width = msg.width
        self.original_height = msg.height

    def image_callback(self, ros_image):

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        # Dimensions of the image used for inference
        inference_width = cv_image.shape[1]  # 416
        inference_height = cv_image.shape[0]  # 416

        # Compute scale factors
        scale_x = self.original_width / inference_width
        scale_y = self.original_height / inference_height

        detections = []

        for model_id, color in self.models:
            results = self.model_client.infer(cv_image, model_id=model_id)
            detections.append(sv.Detections.from_inference(results))

        actual_detections = [d for d in detections if len(d) > 0]
        if not actual_detections:
            self.get_logger().warning("No detections found! Skipping annotation.")
            return

        for detection in actual_detections:
            for bbx in detection[0][0].xyxy:
                x1, y1, x2, y2 = bbx
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # Scale the coordinates
                scaled_x = center_x * scale_x
                scaled_y = center_y * scale_y
                
                pixel_coords = PointStamped()
                pixel_coords.header = ros_image.header
                pixel_coords.point.x = float(scaled_x)
                pixel_coords.point.y = float(scaled_y)
                pixel_coords.point.z = 0.0
                self.point_pub.publish(pixel_coords)

        # Annotate image with bounding boxes and labels
        annotated_image = self.annotate_image(cv_image, actual_detections)
        self.publish_annotated_image(annotated_image, ros_image.header)

    def annotate_image(self, image, detections):
        """ Annotate image with bounding boxes and class labels """
        annotated_image = image.copy()
        for i, detection in enumerate(detections):
            color = self.models[i][1]
            box_annotator = sv.BoxAnnotator(color=color)
            label_annotator = sv.LabelAnnotator(color=color)

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detection['class_name'], detection.confidence)]

            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detection)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detection, labels=labels)

        sv.plot_image(annotated_image)
        return annotated_image

    def publish_annotated_image(self, annotated_image, header):
        """ Publish annotated image as a ROS message """
        annotated_ros_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        annotated_ros_image.header = header
        self.annotated_image_pub.publish(annotated_ros_image)
        self.get_logger().info("Published annotated image.")


def main(args=None):
    rclpy.init(args=args)
    client = LeakageDetection()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
