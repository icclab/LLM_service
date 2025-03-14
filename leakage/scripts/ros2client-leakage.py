import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
# from sensor_msgs.msg import NavSatFix, NavSatStatus
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from px4_msgs.msg import SensorGps, VehicleLocalPosition
from cv_bridge import CvBridge
from llm.srv import CheckImage
from message_filters import Subscriber, ApproximateTimeSynchronizer


class LeakageDetectionClient(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('image_topic', '/drone/image_raw')
        # self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('pose_topic', '/zed/zed_node/odom')
        self.declare_parameter('gps_topic', '/drone/fmu/out/vehicle_gps_position')
        self.declare_parameter('local_topic', '/drone/fmu/out/vehicle_local_position')

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
        # self.image_subscription = self.create_subscription(ROSImage, image_topic, self.image_cb, 10)
        self.pose_subscription = self.create_subscription(Odometry, pose_topic, self.pose_cb, 10)
        # self.gps_subscription = self.create_subscription(SensorGps, gps_topic, self.gps_cb, 10)
        # self.local_subscription = self.create_subscription(VehicleLocalPosition, local_topic, self.local_cb, 10)
        self.image_subscription = Subscriber(self, ROSImage, image_topic)
        self.gps_subscription = Subscriber(self, SensorGps, gps_topic)
        self.local_subscription = Subscriber(self, VehicleLocalPosition, local_topic)

        # Synchronizer to match messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.gps_subscription, self.local_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        self.leakage_gps_pub = self.create_publisher(Point, '/leakage_gps', 10)
        self.leakage_local_pub = self.create_publisher(Point, '/leakage_local', 10)

        self.latest_gps = None
        self.latest_local = None 
        self.request_in_progress = False  

    def synchronized_callback(self, ros_image: ROSImage, local_pos_msg: VehicleLocalPosition, gps_msg: SensorGps):
        if self.request_in_progress:
            #self.get_logger().warn("Previous request still in progress, skipping new request.")
            return  # Skip sending a new request

        self.get_logger().info('Image and Pose received, sending service request...')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        
        self.latest_local = local_pos_msg
        self.latest_gps = gps_msg

        self.get_logger().info(f"Synchronized data: Local Pos({local_pos_msg.x}, {local_pos_msg.y}, {local_pos_msg.z}), "
                               f"GPS({gps_msg.latitude_deg}, {gps_msg.longitude_deg}, {gps_msg.altitude_msl_m})")
               
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
                    leakage_gps_msg.x = self.latest_gps.latitude_deg  # Latitude as X
                    leakage_gps_msg.y = self.latest_gps.longitude_deg  # Longitude as Y
                    leakage_gps_msg.z = self.latest_gps.altitude_msl_m  # Altitude as Z
                    self.leakage_gps_pub.publish(leakage_gps_msg)
            else:
                self.get_logger().info('No leakage detected.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

        self.request_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    client = LeakageDetectionClient()

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

