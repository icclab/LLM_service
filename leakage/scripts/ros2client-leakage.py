import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from leakage.srv import CheckImage

class LeakageDetectionClient(Node):
    def __init__(self):
        super().__init__('leakage_detection_client')
        self.client = self.create_client(CheckImage, 'check_leakage')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            ROSImage,
            '/intel_realsense_r200_depth/image_raw',  # Topic publishing the image
            self.image_cb,
            10
        )
        self.image_subscription  # Prevent unused variable warning

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
                self.get_logger().info('Leakage detected!')
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

