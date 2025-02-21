import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped
from visualization_msgs.msg import Marker, MarkerArray
import math
class PoseTransformer(Node):
    def __init__(self):
        super().__init__('pose_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_sub = self.create_subscription(PoseStamped, '/person_pose_camera', self.pose_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/person_pose_map', 10)
        
        self.marker_pub = self.create_publisher(MarkerArray, '/person_markers', 10)

        # Store past markers
        self.marker_list = []
        self.marker_id = 0

        self.distance_threshold = 1  # 1m Threshold for merging markers

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
                f'Transformed person pose to map frame: x: {position.x}, y: {position.y}, z: {position.z}'
            )

            # Store and publish the marker list
            self.update_or_add_marker(position)

        except Exception as e:
            self.get_logger().warn(f'TF2 Transform Error: {str(e)}')

    def update_or_add_marker(self, position):
        """
        Checks if a new marker is close to an existing one (within 20cm).
        If so, updates the existing marker instead of creating a new one.
        """
        for marker in self.marker_list:
            dist = self.calculate_distance(position, marker.pose.position)
            if dist < self.distance_threshold:
                # If within threshold, update existing marker position
                marker.pose.position.x = position.x
                marker.pose.position.y = position.y
                marker.pose.position.z = position.z
                self.publish_markers()
                return

        self.add_new_marker(position)

    def add_new_marker(self, position):
        """
        Adds a new marker to the list.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "person_markers"
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z
        marker.pose.orientation.w = 1.0

        # Set marker scale (size of the sphere)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Set marker color (RGBA)
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set marker lifetime (0 means forever)
        marker.lifetime.sec = 0

        self.marker_list.append(marker)
        self.marker_id += 1

        self.publish_markers()

    def calculate_distance(self, pos1, pos2):
        """
        Calculates Euclidean distance between two points.
        """
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2 +
            (pos1.z - pos2.z) ** 2
        )

    def publish_markers(self):
        """
        Publishes all markers in the list.
        """
        marker_array = MarkerArray()
        marker_array.markers = self.marker_list
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PoseTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
