import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import String
from cv_bridge import CvBridge
from leakage.srv import CheckPosture
import time
import cv2
import numpy as np
from image_geometry import PinholeCameraModel

class PostureDetectionClient(Node):
    def __init__(self):
        super().__init__('posture_detection_client')
        self.client = self.create_client(CheckPosture, 'check_posture')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()
        self.person_detected = False  # Flag to avoid duplicate requests

        self.previous_image = None  # Store the previous image
        self.request_in_progress = False  # Track the state of the service request
        self.pending_requests = {}  # Track pending requests: {future: (pose, timestamp)}

        self.yolo_subscription = self.create_subscription(YoloResult, '/yolo_result', self.yolo_cb, 10)

        self.posture_pose_publisher = self.create_publisher(Pose, '/posture_pose', 10)
        self.posture_publisher = self.create_publisher(String, '/posture', 10)
        
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed/zed_node/rgb/camera_info', self.camera_info_callback, 10)
        self.image_subscription = self.create_subscription(ROSImage, '/image_raw', self.image_cb, 10)
        self.depth_subscription = self.create_subscription(ROSImage, '/zed/zed_node/depth/depth_registered', self.depth_cb, 10)
        self.pose_subscription = self.create_subscription(PoseStamped, '/zed/zed_node/pose', self.pose_cb, 10)

        self.camera_model = PinholeCameraModel()
        # Store latest synchronized data
        self.latest_image = None
        self.latest_depth = None
        self.latest_pose = None


    def camera_info_callback(self, camera_info_msg):
        self.camera_model.fromCameraInfo(camera_info_msg)
        # self.get_logger().info("Camera model initialized from CameraInfo")

    def image_cb(self, msg):
        """
        Callback for image topic. Updates the latest image when received.
        """
        self.latest_image = msg

    def depth_cb(self, msg):
        """
        Callback for depth image topic. Updates the latest depth image when received.
        """
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(str(e))
            return

    def pose_cb(self, msg):
        """
        Callback for pose topic. Updates the latest pose when received.
        """
        self.latest_pose = msg.pose    
    
    def yolo_cb(self, msg):
        """
        Callback for YOLO detections. Checks if a person is detected.
        """
        if not msg.detections.detections:  # Ensure there are detections
            # self.get_logger().info("No detections found")
            return

        for detection in msg.detections.detections:
            if not detection.results:
                continue  # Skip if no results in detection

            object_id = detection.results[0].hypothesis.class_id

            # Check if the detected object is a person
            if object_id == "person":
                self.get_logger().info(f"Person detected!")
                
                # Extract bounding box coordinates
                bbox = detection.bbox
                center_x, center_y = int(bbox.center.position.x), int(bbox.center.position.y)
                size_x, size_y = int(bbox.size_x), int(bbox.size_y)
                
                if self.latest_depth is None:
                    self.get_logger().warn("Depth image not available yet!")
                    continue
                
                # Compute top-left and bottom-right coordinates
                x_min, y_min = max(0, center_x - size_x // 2), max(0, center_y - size_y // 2)
                x_max, y_max = min(self.latest_depth.shape[1], center_x + size_x // 2), min(self.latest_depth.shape[0], center_y + size_y // 2)

                # Estimate depth using bounding box
                avg_depth = self.estimate_depth(x_min, y_min, x_max, y_max)

                # Convert to 3D point in camera frame
                if avg_depth is not None and self.latest_pose is not None:
                    person_position = self.pixel_to_3d(center_x, center_y, avg_depth)
                    self.get_logger().info(f"Person 3D Position in Camera Frame: {person_position}")

                self.person_detected = True
                self.process_posture()
                return  # Exit after detecting the first person

        # If no person detected, reset flag
        self.person_detected = False
    
    def estimate_depth(self, x_min, y_min, x_max, y_max):
        """
        Estimate the depth of a person using multiple sample pixels inside the bounding box.
        """
        if self.latest_depth is None:
            self.get_logger().warn("Depth image not received yet.")
            return None

        sample_pixels = []
        step_x = max(1, (x_max - x_min) // 3)  # Avoid zero division
        step_y = max(1, (y_max - y_min) // 3)

        for i in range(3):
            for j in range(3):
                x = x_min + i * step_x
                y = y_min + j * step_y

                if 0 <= x < self.latest_depth.shape[1] and 0 <= y < self.latest_depth.shape[0]:
                    depth_value = self.latest_depth[y, x]
                    if depth_value > 0:
                        sample_pixels.append(depth_value)

        if not sample_pixels:
            self.get_logger().warn("No valid depth values found in bounding box.")
            return None

        avg_depth = np.median(sample_pixels)  # Use median for robustness
        return avg_depth

    def pixel_to_3d(self, x, y, depth):
        """
        Convert pixel coordinates (x, y) and depth to a 3D point in the camera frame.
        """
        if depth <= 0:
            return None

        if not self.camera_model.fx():  # Check if camera model is initialized
            self.get_logger().warn("Camera model not initialized yet!")
            return None

        # Get the unit direction ray from the pixel
        ray = self.camera_model.projectPixelTo3dRay((x, y))
        ray = np.array(ray)  # Convert to NumPy array

        # Scale the ray by depth to get the actual 3D position
        point_3d = ray * (depth / ray[2])  # Normalize using the Z component

        return point_3d

    def process_posture(self):
        """
        Check posture when a person is detected.
        """
        if self.latest_image is None or self.latest_pose is None:
            return

        # Only proceed if no request is in progress
        if self.request_in_progress:
            #self.get_logger().info('Request in progress, skipping new image.')
            return

        self.get_logger().info('Sending image for posture check...')

        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')

        # Record the time before sending the request
        request_time = time.time()

        # Create a request object
        request = CheckPosture.Request()
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

        # Mark that a request is in progress
        self.request_in_progress = True

        # Send the service request
        future = self.client.call_async(request)

        # Store the pose and timestamp associated with this request
        self.pending_requests[future] = (self.latest_pose, request_time)

        # Add a callback to handle the response
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        try:
            # Retrieve the pose and timestamp associated with this response
            pose, request_time = self.pending_requests.pop(future, (None, None))
            if pose is None:
                self.get_logger().error('Pose not found for this response.')
                return

            # Calculate the response time
            response_time = time.time() - request_time
            #response_time = time.time() - self.last_request_time
            self.get_logger().info(f'Time taken for service response: {response_time:.3f} seconds')

            response = future.result()
            if response is None:
                self.get_logger().error("No response received.")
                return
            
            # Check the posture_detected value and map it to a string
            if response.posture_detected == 0:
                posture_string = "nothing found"
            elif response.posture_detected == 1:
                posture_string = "standing"
            elif response.posture_detected == 2:
                posture_string = "sitting"
            elif response.posture_detected == 3:
                posture_string = "lying"
            else:
                posture_string = "unknown posture"  # Handle unexpected values
            
            posture_msg = String()
            posture_msg.data = posture_string
            self.posture_publisher.publish(posture_msg)

            if response.posture_detected !=0:
                self.get_logger().info('Person posture detected!')
                # Publish the robot pose to /posture_pose
                self.posture_pose_publisher.publish(pose)
                self.get_logger().info(f'Person posture detected: {posture_string}, Published posture pose: Position=({pose.position.x}, {pose.position.y}, {pose.position.z})'
                )
            else:
                self.get_logger().info('No posture detected.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

        self.request_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    client = PostureDetectionClient()

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

