import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from px4_msgs.msg import SensorGps
from cv_bridge import CvBridge
from llm.srv import CheckImage

class LeakageDetectionClient(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('image_topic', '/drone/image_raw')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('pose_topic', '/zed/zed_node/odom')
        self.declare_parameter('gps_topic', '/drone/fmu/out/vehicle_gps_position')

        # Get parameters from the launch file
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        gps_topic = self.get_parameter('gps_topic').get_parameter_value().string_value

        self.client = self.create_client(CheckImage, 'check_leakage')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(ROSImage, image_topic, self.image_cb, 10)
        self.pose_subscription = self.create_subscription(Odometry, pose_topic, self.pose_cb, 10)
        self.gps_subscription = self.create_subscription(SensorGps, gps_topic, self.gps_cb, 10)

        self.gps_publisher = self.create_publisher(NavSatFix, '/leakage_gps', 10)

        self.latest_gps = None

    def pose_cb(self, msg):
        """ Updates latest camera pose. """
        self.latest_pose = msg.pose.pose    
        
    def gps_cb(self, msg):
        """ Updates latest drone gps. """
        self.latest_gps = msg

    def image_cb(self, ros_image):
        self.get_logger().info('Image received, sending service request...')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

        # Create a request object
        request = CheckImage.Request()
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

        # Send the service request
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            response = future.result()
            if response.leakage_detected:
                navsat_msg = NavSatFix()
                navsat_msg.header.stamp = self.get_clock().now().to_msg()
                navsat_msg.header.frame_id = 'gps'

                # Set status based on fix_type
                if self.latest_gps.fix_type >= SensorGps.FIX_TYPE_2D:
                    navsat_msg.status.status = NavSatStatus.STATUS_FIX
                else:
                    navsat_msg.status.status = NavSatStatus.STATUS_NO_FIX

                navsat_msg.status.service = NavSatStatus.SERVICE_GPS

                navsat_msg.latitude = self.latest_gps.latitude_deg
                navsat_msg.longitude = self.latest_gps.longitude_deg
                navsat_msg.altitude = self.latest_gps.altitude_msl_m

                # Position covariance (use eph/epv if available, else unknown)
                if self.latest_gps.eph > 0 and self.latest_gps.epv > 0:
                    navsat_msg.position_covariance = [
                        self.latest_gps.eph**2, 0.0, 0.0,
                        0.0, self.latest_gps.eph**2, 0.0,
                        0.0, 0.0, self.latest_gps.epv**2
                    ]
                    navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                else:
                    navsat_msg.position_covariance = [0.0] * 9
                    navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

                self.gps_publisher.publish(navsat_msg)             
                
                self.get_logger().info(f'Leakage detected! Drone Last gps published!')
            else:
                self.get_logger().info('No leakage detected.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    client = LeakageDetectionClient()

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

