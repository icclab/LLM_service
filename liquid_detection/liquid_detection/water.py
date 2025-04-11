#!/usr/bin/env python3

"""
ROS 2 Node to project pixel coordinates onto the ground plane (Z=0).

Subscribes to camera info, pixel coordinates (PointStamped), and TF transforms.
Calculates the 3D intersection point of the ray cast from the camera through
the pixel with the Z=0 plane in a specified world frame. Publishes the
result as a PointStamped.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped, Point, Vector3, Vector3Stamped
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs # Import needed for do_transform_vector3
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import image_geometry
import numpy as np
# Optional, but useful for robust vector rotation if not using tf2_geometry_msgs helper
# from scipy.spatial.transform import Rotation as R

class PixelToGroundNode(Node):
    """
    Node that projects pixel coordinates onto the Z=0 ground plane.
    """
    def __init__(self):
        super().__init__('pixel_to_ground_node')

        # --- Parameters ---
        self.declare_parameter('world_frame', 'map') # Frame where Z=0 is ground
        self.declare_parameter('pixel_topic', '/leakage_pixel_coords')
        self.declare_parameter('camera_info_topic', '/summit/color/camera_info') # Adjust as needed
        self.declare_parameter('output_topic', '/leakage_ground_point')
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        pixel_topic = self.get_parameter('pixel_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.tf_lookup_timeout = self.get_parameter('tf_lookup_timeout_sec').get_parameter_value().double_value

        # --- State Variables ---
        self.camera_model = None
        self.camera_optical_frame_id = None # Will be read from CameraInfo

        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Image Geometry ---
        # Subscribe to CameraInfo
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            1) # QoS profile 1 for latching-like behavior

        # --- Publisher ---
        self.ground_point_pub = self.create_publisher(
            PoseStamped,
            output_topic,
            10)
        
        self.marker_pub = self.create_publisher(Marker, '/leakage_marker', 10)



        # --- Subscriber (must be created AFTER publisher and TF listener) ---
        self.pixel_sub = self.create_subscription(
            PointStamped,
            pixel_topic,
            self.pixel_callback,
            10)

        self.get_logger().info(f"Node '{self.get_name()}' started.")
        self.get_logger().info(f"World frame (for Z=0 plane): '{self.world_frame}'")
        self.get_logger().info(f"Waiting for CameraInfo on: '{camera_info_topic}'...")


    def camera_info_callback(self, msg: CameraInfo):
        """Callback for receiving camera calibration data."""
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)
            self.camera_optical_frame_id = msg.header.frame_id
            self.get_logger().info(f"Camera model initialized for frame '{self.camera_optical_frame_id}'.")
            # Once camera info is received, we don't need it constantly
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None # Avoid trying to destroy it again


    def pixel_callback(self, msg: PointStamped):
        """Callback for receiving pixel coordinates."""
        if self.camera_model is None:
            self.get_logger().warn("Camera model not initialized yet. Skipping pixel message.", throttle_duration_sec=5)
            return

        # Validate input frame ID - should match camera optical frame
        if msg.header.frame_id != self.camera_optical_frame_id:
            self.get_logger().error(
                f"Pixel coordinate frame '{msg.header.frame_id}' does not match "
                f"camera optical frame '{self.camera_optical_frame_id}'. Cannot process.")
            return

        # Assume pixel coordinates are in point.x and point.y
        pixel_u = msg.point.x
        pixel_v = msg.point.y
        timestamp = msg.header.stamp

        self.get_logger().debug(f"Received pixel coordinates: u={pixel_u}, v={pixel_v} in frame {msg.header.frame_id}")

        try:
            # --- 1. Get Ray in Camera Frame ---
            # This gives a unit vector in the camera's optical frame
            ray_cam_tuple = self.camera_model.projectPixelTo3dRay((pixel_u, pixel_v))
            ray_cam = Vector3(x=ray_cam_tuple[0], y=ray_cam_tuple[1], z=ray_cam_tuple[2]) # Make it a Vector3

            # --- 2. Get Camera Pose in World Frame ---
            # Get the transform from world_frame to camera_optical_frame
            # This transform represents the pose of the camera *in* the world frame
            try:
                 # Use timestamp=None for latest available, or msg.header.stamp for specific time
                 # Use rclpy.time.Time() for latest if msg timestamp is zero.
                lookup_time = rclpy.time.Time.from_msg(timestamp) if (timestamp.sec > 0 or timestamp.nanosec > 0) else None # Use None for latest if timestamp is zero
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.world_frame,                     # Target frame
                    self.camera_optical_frame_id,         # Source frame
                    lookup_time if lookup_time else rclpy.time.Time(), # Time (use now() if msg time is 0)
                    Duration(seconds=self.tf_lookup_timeout) # Timeout
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().error(f"TF lookup failed from {self.world_frame} to {self.camera_optical_frame_id}: {e}")
                return

            # Extract camera position (C_world) and orientation (R_world_cam) from the transform
            print(transform_stamped)
            cam_pos_world = transform_stamped.transform.translation # Vector3
            cam_quat_world = transform_stamped.transform.rotation   # Quaternion

            # --- 3. Transform Ray to World Frame ---
            # Method 1: Using tf2_geometry_msgs helper (requires importing it)
            # We need a transform to rotate vectors, which is essentially just the rotation part
            # Construct a minimal transform with only rotation to use do_transform_vector3
            rotation_transform = tf2_ros.TransformStamped()
            rotation_transform.transform.rotation = cam_quat_world
            ray_cam_stamped = Vector3Stamped()
            ray_cam_stamped.header.frame_id = self.camera_optical_frame_id
            ray_cam_stamped.header.stamp = timestamp
            ray_cam_stamped.vector = ray_cam
            # Transform the ray from camera frame to world frame
            ray_world_geo = tf2_geometry_msgs.do_transform_vector3(ray_cam_stamped, rotation_transform)
            ray_world = np.array([ray_world_geo.vector.x, ray_world_geo.vector.y, ray_world_geo.vector.z])

            # # Method 2: Using scipy (if you prefer numpy/scipy)
            # # Make sure cam_quat_world is in [x, y, z, w] format for scipy
            # rot_matrix = R.from_quat([cam_quat_world.x, cam_quat_world.y, cam_quat_world.z, cam_quat_world.w]).as_matrix()
            # ray_cam_np = np.array([ray_cam.x, ray_cam.y, ray_cam.z])
            # ray_world = rot_matrix @ ray_cam_np # Matrix multiplication

            # --- 4. Calculate Intersection with Z=0 Plane ---
            # Ray: P(t) = C_world + t * ray_world
            # Plane: Z = 0 (normal n = [0, 0, 1])
            # Intersection: n · P = 0 => n · (C_world + t * ray_world) = 0
            # C_world.z + t * (n · ray_world) = 0
            # C_world.z + t * ray_world.z = 0

            C_world = np.array([cam_pos_world.x, cam_pos_world.y, cam_pos_world.z])
            ray_world_z = ray_world[2] # The z-component of the direction vector

            # Check if ray is parallel to the ground plane (or pointing upwards from below ground)
            if abs(ray_world_z) < 1e-6:
                # Check if camera is already on the ground? Very unlikely needed.
                # if abs(C_world[2]) < 1e-6:
                #     self.get_logger().warn("Camera is on the ground plane and ray is parallel. Cannot determine unique point.")
                #     return
                # else:
                    self.get_logger().warn(f"Ray is parallel to the ground plane (Z component {ray_world_z:.4f}). Cannot intersect.")
                    return

            # Calculate intersection parameter t
            t = -C_world[2] / ray_world_z

            # Check if intersection is in front of the camera
            if t <= 0:
                self.get_logger().warn(f"Intersection parameter t = {t:.3f} is not positive. Intersection is behind the camera.")
                return

            # Calculate intersection point
            P_intersect = C_world + t * ray_world

            # --- 5. Publish Result ---
            output_msg = PoseStamped()
            output_msg.header.stamp = timestamp
            output_msg.header.frame_id = self.world_frame
            output_msg.pose.position.x = P_intersect[0]
            output_msg.pose.position.y = P_intersect[1]
            # Z should be very close to 0, setting it explicitly might be cleaner
            output_msg.pose.position.z = 0.0 # P_intersect[2]
            # output_msg.pose.position.z = P_intersect[2]

            self.ground_point_pub.publish(output_msg)
            self.publish_leakage_marker(output_msg.pose.position)

        except Exception as e:
            self.get_logger().error(f"Failed to process pixel callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def publish_leakage_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "leakage"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = position.x
        marker.pose.position.y = position.y
        marker.pose.position.z = position.z

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)
        self.get_logger().info("Published leakage marker.")


def main(args=None):
    rclpy.init(args=args)
    node = PixelToGroundNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.try_shutdown() # Use try_shutdown() in MultiThreadedExecutor context if applicable

if __name__ == '__main__':
    main()