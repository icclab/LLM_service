import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose, PoseStamped
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import String
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

        self.previous_image = None  # Store the previous image
        self.request_in_progress = False  # Track the state of the service request
        self.pending_requests = {}  # Track pending requests: {future: (pose, timestamp)}

        self.yolo_subscription = self.create_subscription(YoloResult, '/yolo_result', self.yolo_cb, 10)

        self.posture_pose_publisher = self.create_publisher(Pose, '/posture_pose', 10)
        self.posture_publisher = self.create_publisher(String, '/posture', 10)

        self.image_subscription = self.create_subscription(ROSImage, '/image_raw', self.image_cb, 10)
        self.depth_subscription = self.create_subscription(ROSImage, '/zed/zed_node/depth/depth_registered', self.depth_cb, 10)
        self.pose_subscription = self.create_subscription(PoseStamped, '/zed/zed_node/pose', self.pose_cb, 10)

        
        # Store latest synchronized data
        self.latest_image = None
        self.latest_depth = None
        self.latest_pose = None

    def image_cb(self, msg):
        """
        Callback for image topic. Updates the latest image when received.
        """
        self.latest_image = msg    
        
    def depth_cb(self, msg):
        """
        Callback for depth image topic. Updates the latest depth image when received.
        """
        self.latest_depth = msg

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
            confidence = detection.results[0].hypothesis.score

            # Check if the detected object is a person
            if object_id == "person":
                # self.get_logger().info(f"Person detected with confidence: {confidence}")
                self.person_detected = True
                self.process_posture()
                return  # Exit after detecting the first person

        # If no person detected, reset flag
        self.person_detected = False
    
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

