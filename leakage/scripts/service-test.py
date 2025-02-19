import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import random

from leakage.srv import CheckPosture  # Custom ROS 2 service

class ImageSub(Node):
    def __init__(self):
        super().__init__('image_sub')
        self.bridge = CvBridge()  # Initialize CvBridge to convert ROS Image messages to OpenCV images

        # Create the ROS 2 service
        self.service = self.create_service(CheckPosture, 'check_posture', self.handle_check_posture)

    def handle_check_posture(self, request, response):
        """
        Handles the ROS 2 service request for checking posture.
        """
        self.get_logger().info('Received request to check posture.')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')

        # Convert OpenCV image to PIL for potential future processing
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # For testing: Simply return a default response (1)
        response.posture_detected = self.detect_posture(pil_image)

        return response

    def detect_posture(self, pil_image):
        """
        Dummy posture detection: random responses (1, 2, or 3),.
        """
        posture_value = random.choice([1, 2, 3])
        self.get_logger().info(f'Detected posture: {posture_value}')
        return posture_value

def main(args=None):
    rclpy.init(args=args)
    image_sub = ImageSub()

    try:
        rclpy.spin(image_sub)
    except KeyboardInterrupt:
        image_sub.get_logger().info('Node stopped by user')
    finally:
        image_sub.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
