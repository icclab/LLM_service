import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from llm.srv import CheckImage
from message_filters import Subscriber, ApproximateTimeSynchronizer


class LeakageDetectionClient(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('pose_topic', '/zed/zed_node/odom')

        # Get parameters from the launch file
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value

        self.client = self.create_client(CheckImage, 'check_leakage')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()
        self.pose_subscription = Subscriber(self, Odometry, pose_topic)
        self.image_subscription = Subscriber(self, ROSImage, image_topic)

        # Synchronizer to match messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_subscription, self.pose_subscription], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synchronized_callback)
        
        # self.leakage_gps_pub = self.create_publisher(Point, '/leakage_gps', 10)
        self.leakage_local_pub = self.create_publisher(Point, '/leakage_local', 10)

        self.latest_local = None 
        self.request_in_progress = False  

    def synchronized_callback(self, ros_image: ROSImage, local_pos_msg: Odometry):
        self.get_logger().info('Image and Pose received, sending service request...')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        
        self.latest_local = local_pos_msg.pose.pose.position

        self.get_logger().info(f"Synchronized data: Local Pos({self.latest_local.x}, {self.latest_local.y}, {self.latest_local.z})")
               
        # Create a request object
        request = CheckImage.Request()
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

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

