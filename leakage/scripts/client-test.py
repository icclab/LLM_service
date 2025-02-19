import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
from leakage.srv import CheckPosture
import time
import cv2
import numpy as np

class PostureDetectionClient(Node):
    def __init__(self):
        super().__init__('posture_detection_client')
        self.client = self.create_client(CheckPosture, 'check_posture')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.bridge = CvBridge()
        self.person_detected = False  # Flag to avoid duplicate requests
        self.request_in_progress = False  # Track the state of the service request
        self.pending_requests = {}  # Track pending requests: {future: timestamp}

        # Subscribe to YOLO detection topic
        self.yolo_subscription = self.create_subscription(
            DetectionArray, '/yolo/detections', self.yolo_cb, 10
        )

        # Subscribe to RGB image topic (RealSense or any camera)
        self.image_subscription = self.create_subscription(
            ROSImage, '/camera/color/image_raw', self.image_cb, 10
        )

        # Publishers for detected posture
        self.posture_publisher = self.create_publisher(String, '/posture', 10)

        # Store the latest image for posture detection
        self.latest_image = None
        
    def yolo_cb(self, msg):
        """
        Callback for YOLO detections. Checks if a person is detected.
        """
        person_detected = any(det.class_name == "person" for det in msg.detections)

        if person_detected and not self.person_detected:
            self.person_detected = True
            self.get_logger().info('Person detected! Waiting for next image...')
        elif not person_detected:
            self.person_detected = False  # Reset flag when no person is detected

    def image_cb(self, ros_image):
        """
        Callback for image topic. Sends an image to check posture when a person is detected.
        """
        if not self.person_detected:
            return  # Skip processing if no person detected
        
        if self.request_in_progress:
            return  # Skip if a request is already in progress

        self.get_logger().info('Sending image for posture check...')

        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

        # Record the time before sending the request
        request_time = time.time()

        # Create a request object
        request = CheckPosture.Request()
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

        # Mark that a request is in progress
        self.request_in_progress = True

        # Send the service request
        future = self.client.call_async(request)

        # Store the timestamp associated with this request
        self.pending_requests[future] = request_time

        # Add a callback to handle the response
        future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        """
        Handles the response from the posture detection service.
        """
        try:
            request_time = self.pending_requests.pop(future, None)
            if request_time is None:
                self.get_logger().error('Request timestamp not found.')
                return

            response_time = time.time() - request_time
            self.get_logger().info(f'Time taken for service response: {response_time:.3f} seconds')

            response = future.result()

            # Map posture_detected value to a string
            posture_map = {
                0: "nothing found",
                1: "standing",
                2: "sitting",
                3: "lying"
            }
            posture_string = posture_map.get(response.posture_detected, "unknown posture")

            self.get_logger().info(f'Person posture detected: {posture_string}')

            # Publish detected posture
            posture_msg = String()
            posture_msg.data = posture_string
            self.posture_publisher.publish(posture_msg)

        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

        # Reset request status
        self.request_in_progress = False

def main(args=None):
    rclpy.init(args=args)
    client = PostureDetectionClient()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Node stopped by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
