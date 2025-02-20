import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped

class PoseTransformer(Node):
    def __init__(self):
        super().__init__('pose_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_sub = self.create_subscription(PoseStamped, '/person_pose_camera', self.pose_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/person_pose_map', 10)

    def pose_callback(self, msg):
        """
        Callback function to transform the PoseStamped from camera frame to map frame.
        """
        try:
            target_frame = "map"
            source_frame = msg.header.frame_id

            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time().to_msg())

            transformed_pose = do_transform_pose_stamped(msg, transform)

            self.pose_pub.publish(transformed_pose)

            position = transformed_pose.pose.position
            self.get_logger().info(
                f'Transformed person pose to map frame: '
                f'x: {position.x}, y: {position.y}, z: {position.z}'
            )

        except Exception as e:
            self.get_logger().warn(f'TF2 Transform Error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

