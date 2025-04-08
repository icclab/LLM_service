import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import NavSatFix, CameraInfo
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
import numpy as np
from tf2_ros import Buffer, TransformListener
from image_geometry import PinholeCameraModel
from tf2_geometry_msgs import do_transform_point

class LeakageDetection(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/summit/front_rgbd_camera/color/image_raw')
        self.declare_parameter('gps_topic', '/summit/gps/fix')
        self.declare_parameter('local_topic', '/summit/base_pose')
        self.declare_parameter('depth_topic', '/summit/stereo/depth')
        self.declare_parameter('camera_info_topic', '/summit/color/camera_info')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value
        local_topic = self.get_parameter('local_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_model = PinholeCameraModel()

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10)

        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        self.image_subscription = Subscriber(self, ROSImage, image_topic, qos_profile=qos_profile)
        # self.gps_subscription = Subscriber(self, NavSatFix, gps_topic, qos_profile=qos_profile)
        self.local_subscription = Subscriber(self, PoseStamped, local_topic, qos_profile=qos_profile)
        self.depth_subscription = Subscriber(self, ROSImage, depth_topic, qos_profile=qos_profile)

        # self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.gps_subscription, self.local_subscription], queue_size=10, slop=0.1)
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.depth_subscription, self.local_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        self.leakage_gps_pub = self.create_publisher(Point, '/leakage_gps', 10)
        self.leakage_local_pub = self.create_publisher(Point, '/leakage_local', 10)
        self.marker_pub = self.create_publisher(Marker, '/leakage_marker', 10)
        self.annotated_image_pub = self.create_publisher(ROSImage, '/leakage_annotated_image', 10)
        
        # self.latest_gps = None
        self.latest_local = None
        self.depth_img = None
        self.model_client = InferenceHTTPClient(
            api_url="https://roboflow.nephele-project.ch",
            api_key="wrJRdyQTYXns1D8MwoxW",
        )

        self.models = [
            #("water-leakage/2", sv.Color(255, 0, 0)),  # Red
            ("water-ba8zz/1", sv.Color(0, 255, 0)),  # Green
            ("spills-ax5xv/2", sv.Color(0, 0, 255))  # Blue
        ]

    def camera_info_callback(self, camera_info_msg):
        """ Initialize camera model from CameraInfo. """
        self.camera_model.fromCameraInfo(camera_info_msg)

    # def synchronized_callback(self, ros_image: ROSImage, gps_msg: NavSatFix, local_pos_msg: PoseStamped):
    def synchronized_callback(self, ros_image: ROSImage, depth_image: ROSImage, local_pos_msg: PoseStamped):
        self.get_logger().info('Image and Pose received, running inference...')

        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        self.latest_local = local_pos_msg
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_image)
        # self.latest_gps = gps_msg

        detections = []

        for model_id, color in self.models:
            results = self.model_client.infer(cv_image, model_id=model_id)
            detections.append(sv.Detections.from_inference(results))

        actual_detections = [d for d in detections if len(d) > 0]
        if not actual_detections:
            self.get_logger().warning("No detections found! Skipping annotation.")
            return

        self.get_logger().info('Leakage detected! Publishing last known position...')

        for det in actual_detections[0]:
            center_x = int(det.x)
            center_y = int(det.y)

            if self.depth_img is not None:
                try:
                    depth_value = self.depth_img[center_y, center_x]
                    if depth_value == 0:
                        self.get_logger().warn(f"Invalid depth at pixel ({center_x}, {center_y})")
                        continue

                    ray = self.camera_model.projectPixelTo3dRay((center_x, center_y))
                    ray = np.array(ray)
                    point_3d = ray * (depth_value / ray[2])

                    self.get_logger().info(f"Detection center at pixel ({center_x}, {center_y}) => 3D point: {point_3d}")
                    
                    leakage_msg = Point(x=point_3d[0], y=point_3d[1], z=point_3d[2])
                    target_frame = "map"
                    source_frame = "oak_rgb_camera_optical_frame"
                    transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time().to_msg(), timeout=rclpy.duration.Duration(seconds=5.0))
                    transformed_point = do_transform_point(leakage_msg, transform)

                    self.publish_leakage_marker(transformed_point.point)

                except Exception as e:
                    self.get_logger().error(f"Error processing 3D point: {e}")
                    continue
            else:
                self.get_logger().warn("No depth image received yet.")
           
        # if self.latest_local:
        #     leakage_local_msg = Point(x=self.latest_local.pose.position.x, y=self.latest_local.pose.position.y, z=self.latest_local.pose.position.z)
        #     self.leakage_local_pub.publish(leakage_local_msg)

        # if self.latest_gps:
        #     leakage_gps_msg = Point(x=self.latest_gps.latitude, y=self.latest_gps.longitude, z=self.latest_gps.altitude)
        #     self.leakage_gps_pub.publish(leakage_gps_msg)

        # self.publish_leakage_marker(self.latest_local.pose.position)

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

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

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
