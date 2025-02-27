import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageDecompressor(Node):
    def __init__(self):
        super().__init__('image_decompressor')

        # Declare parameters with default values
        self.declare_parameter('compressed_image_topic', '/summit/oak/rgb/image_rect/compressed')
        self.declare_parameter('raw_image_topic', '/image_raw')

        # Get parameters from the launch file
        compressed_image_topic = self.get_parameter('compressed_image_topic').get_parameter_value().string_value
        raw_image_topic = self.get_parameter('raw_image_topic').get_parameter_value().string_value

        self.bridge = CvBridge()

        # Subscribe to the compressed image topic
        self.subscription = self.create_subscription(CompressedImage, compressed_image_topic, self.image_callback, 10)

        # Publisher for the decompressed raw image
        self.publisher = self.create_publisher(Image, raw_image_topic, 10)

        self.get_logger().info('Image decompressor node started.')

    def image_callback(self, msg):
        """ Callback function to decompress and publish raw image """
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert OpenCV image to ROS Image message
            raw_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            # Publish the decompressed raw image
            self.publisher.publish(raw_image_msg)

            self.get_logger().info('Published decompressed image.')

        except Exception as e:
            self.get_logger().error(f'Failed to decompress image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageDecompressor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
