import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

from llm.msg import PoseWithCompressedImage

class ImagePosePublisher(Node):
    def __init__(self):
        super().__init__('image_pose_publisher')

        # Store the latest messages
        self.latest_pose = None
        self.latest_image = None

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(PoseWithCompressedImage, 'robot_image_pose', 10)

        # Create a timer to periodically publish data (e.g., 10 Hz)
        self.timer = self.create_timer(0.1, self.publish_data)

    def pose_callback(self, msg):
        self.latest_pose = msg

    def image_callback(self, msg):
        self.latest_image = msg

    def publish_data(self):
        # Only publish if we have both pose and image
        if self.latest_pose is not None and self.latest_image is not None:
            custom_msg = PoseWithCompressedImage()
            custom_msg.pose = self.latest_pose
            custom_msg.image = self.latest_image

            self.publisher_.publish(custom_msg)
            self.get_logger().info('Publishing PoseWithImage message')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
