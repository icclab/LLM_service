import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, PoseStamped
# from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from llm.srv import CheckImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class LeakageDetectionClient(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('image_topic', '/drone/image_raw')
        # self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('pose_topic', '/zed/zed_node/odom')
        # self.declare_parameter('gps_topic', '/drone/fmu/out/vehicle_gps_position')
        self.declare_parameter('gps_topic', '/mavros/global_position/global')
        # self.declare_parameter('local_topic', '/drone/fmu/out/vehicle_local_position')
        self.declare_parameter('local_topic', '/drone/mavros/local_position/pose')

        # Get parameters from the launch file
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        # depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value
        local_topic = self.get_parameter('local_topic').get_parameter_value().string_value

        self.client = self.create_client(CheckImage, 'check_leakage')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)

        # self.image_subscription = self.create_subscription(ROSImage, image_topic, self.image_cb, 10)
        # self.pose_subscription = self.create_subscription(Odometry, pose_topic, self.pose_cb, 10)
        # self.gps_subscription = self.create_subscription(SensorGps, gps_topic, self.gps_cb, 10)
        # self.local_subscription = self.create_subscription(VehicleLocalPosition, local_topic, self.local_cb, 10)
        self.image_subscription = Subscriber(self, ROSImage, image_topic, qos_profile=qos_profile)
        self.gps_subscription = Subscriber(self, NavSatFix, gps_topic, qos_profile=qos_profile)
        self.local_subscription = Subscriber(self, PoseStamped, local_topic, qos_profile=qos_profile)

        # Synchronizer to match messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.gps_subscription, self.local_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        self.leakage_gps_pub = self.create_publisher(Point, '/leakage_gps', 10)
        self.leakage_local_pub = self.create_publisher(Point, '/leakage_local', 10)
        self.marker_pub = self.create_publisher(Marker, '/leakage_marker', 10)

        self.latest_gps = None
        self.latest_local = None 
        self.request_in_progress = False  

    # def synchronized_callback(self, ros_image: ROSImage, local_pos_msg: VehicleLocalPosition, gps_msg: SensorGps):
    def synchronized_callback(self, ros_image: ROSImage, local_pos_msg: PoseStamped, gps_msg: NavSatFix):
        if self.request_in_progress:
            #self.get_logger().warn("Previous request still in progress, skipping new request.")
            return  # Skip sending a new request

        self.get_logger().info('Image and Pose received, sending service request...')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        
        self.latest_local = local_pos_msg.pose.position
        self.latest_gps = gps_msg

        # self.get_logger().info(f"Synchronized data: Local Pos({local_pos_msg.x}, {local_pos_msg.y}, {local_pos_msg.z}), "
        self.get_logger().info(f"Synchronized data: Local Pos({local_pos_msg.x}, {local_pos_msg.y}, {local_pos_msg.z}), "
                               f"GPS({gps_msg.latitude}, {gps_msg.longitude}, {gps_msg.altitude})")
               
        # Create a request object
        request = CheckImage.Request()
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        request.image.header = ros_image.header
        
        self.request_in_progress = True
        
        # Send the service request
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if response.leakage_detected:
                self.get_logger().info('Leakage detected! Publishing last known position...')

                if self.latest_local:
                    leakage_local_msg = Point()
                    leakage_local_msg.x = self.latest_local.x  # North position
                    leakage_local_msg.y = self.latest_local.y  # East position
                    leakage_local_msg.z = self.latest_local.z  # Down position
                    self.leakage_local_pub.publish(leakage_local_msg)

                if self.latest_gps:
                    leakage_gps_msg = Point()
                    leakage_gps_msg.x = self.latest_gps.latitude  # Latitude as X
                    leakage_gps_msg.y = self.latest_gps.longitude  # Longitude as Y
                    leakage_gps_msg.z = self.latest_gps.altitude  # Altitude as Z
                    self.leakage_gps_pub.publish(leakage_gps_msg)

                self.publish_leakage_marker(self.latest_local)
            else:
                self.get_logger().info('No leakage detected.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

        self.request_in_progress = False

    def publish_leakage_marker(self, position):
        """ Publishes a red marker at the leakage location """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "leakage"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z

        # Set scale (size of the marker)
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Publish marker
        self.marker_pub.publish(marker)
        self.get_logger().info("Published leakage marker.")

def main(args=None):
    rclpy.init(args=args)
    client = LeakageDetectionClient()

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

