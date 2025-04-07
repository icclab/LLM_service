import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, PoseStamped
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
import time


class LeakageDetection(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/drone/image_raw')
        self.declare_parameter('gps_topic', '/drone/mavros/global_position/global')
        self.declare_parameter('local_topic', '/drone/mavros/local_position/pose')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value
        local_topic = self.get_parameter('local_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)

        self.image_subscription = self.create_subscription(ROSImage, '/color/video/image', self.image_cb, 10)
        
        self.leakage_gps_pub = self.create_publisher(Point, '/leakage_gps', 10)
        self.leakage_local_pub = self.create_publisher(Point, '/leakage_local', 10)
        self.marker_pub = self.create_publisher(Marker, '/leakage_marker', 10)
        self.annotated_image_pub = self.create_publisher(ROSImage, '/leakage_annotated_image', 10)
        
        self.latest_gps = None
        self.latest_local = None 
        self.model_client = InferenceHTTPClient(
            api_url="http://160.85.253.140:30334",
            api_key=os.environ["ROBOFLOW_API_KEY"],
        )

        self.models = [
            #("water-leakage/2", sv.Color(255, 0, 0)),  # Red
            ("water-ba8zz/1", sv.Color(0, 255, 0)),  # Green
            ("spills-ax5xv/2", sv.Color(0, 0, 255))  # Blue
        ]

    def image_cb(self, ros_image):
        self.get_logger().info('Image received, running inference...')

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        detections = []
        total_inference_time = 0.0

        # Run inference for each model & measure time
        for model_id, color in self.models:
            start_time = time.time()
            results = self.model_client.infer(cv_image, model_id=model_id)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            self.get_logger().info(f"Inference time for {model_id}: {inference_time:.2f} seconds")
            detections.append(sv.Detections.from_inference(results))

        if len(detections) == 0:
            self.get_logger().warning("No detections found! Skipping annotation.")
            return

        self.get_logger().info("Leakage detected! Publishing last known position...")
        annotated_image = self.annotate_image(cv_image, detections)
        self.publish_annotated_image(annotated_image, ros_image.header)

        self.get_logger().info(f"Total inference time: {total_inference_time:.3f} seconds")

    def annotate_image(self, image, detections):
        """Annotates the image with bounding boxes and labels."""
        annotated_image = image.copy()
        for i, detection in enumerate(detections):
            color = self.models[i][1]
            box_annotator = sv.BoxAnnotator(color=color)
            label_annotator = sv.LabelAnnotator(color=color)

            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detection)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detection)

        sv.plot_image(annotated_image)
        return annotated_image

    def publish_annotated_image(self, annotated_image, header):
        """ Publish annotated image as a ROS message """
        annotated_ros_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        annotated_ros_image.header = header
        self.annotated_image_pub.publish(annotated_ros_image)
        self.get_logger().info("Published annotated image.")

    def publish_leakage_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "leakage"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z

        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)
        self.get_logger().info("Published leakage marker.")


def main(args=None):
    rclpy.init(args=args)
    client = LeakageDetection()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
