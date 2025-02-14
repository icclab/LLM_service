import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose
from leakage.msg import PoseWithCompressedImage
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

        self.previous_image = None  # Store the previous image
        self.request_in_progress = False  # Track the state of the service request
        self.pending_requests = {}  # Track pending requests: {future: (pose, timestamp)}

        self.yolo_subscription = self.create_subscription(DetectionArray, '/yolo/detections', self.yolo_cb, 10)
        # self.image_subscription = self.create_subscription(ROSImage, '/intel_realsense_r200_depth/image_raw', self.image_cb, 10)

        self.posture_pose_publisher = self.create_publisher(Pose, '/posture_pose', 10)
        self.posture_publisher = self.create_publisher(String, '/posture', 10)

        self.image_subscription = self.create_subscription(
            PoseWithCompressedImage,
            'robot_image_pose',  # Topic publishing the image
            self.pose_image_cb,
            10
        )
        
        # Store the latest image for posture detection
        self.latest_image = None
        
    def yolo_cb(self, msg):
        """
        Callback for YOLO detections. Checks if a person is detected.
        """
        if msg.detections and len(msg.detections) > 0 :
            obj_class_name = msg.detections[0].class_name
        
            if obj_class_name == "person":  # Check if detection is a person
                if not self.person_detected:
                    self.person_detected = True
                    self.get_logger().info('Person detected! Capturing image...')
                return  # Exit after detecting at least one person

        # If no person detected, reset flag
        self.person_detected = False
    
    def pose_image_cb(self, msg):
        """
        Callback for image topic. Sends an image to check posture when a person is detected.
        """
        if not self.person_detected:
            return  # Skip processing if no person detected
        
        # Only proceed if no request is in progress
        if self.request_in_progress:
            #self.get_logger().info('Request in progress, skipping new image.')
            return

        self.get_logger().info('Sending image for posture check...')

        pose = msg.pose.pose
        compressed_image = msg.image

        self.get_logger().info(
            f'Received Pose: Position=({pose.position.x}, '
            f'{pose.position.y}, {pose.position.z})'
        )

        # Decompress the image using OpenCV
        np_arr = np.frombuffer(compressed_image.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
        self.pending_requests[future] = (pose, request_time)

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

            self.get_logger().info(f'Person posture detected: {posture_string}')

            posture_msg = String()
            posture_msg.data = posture_string
            self.posture_publisher.publish(posture_msg)

            if response.posture_detected:
                self.get_logger().info('Person posture detected!')
                # Publish the robot pose to /posture_pose
                self.posture_pose_publisher.publish(pose)
                self.get_logger().info(
                    f'Published posture pose: Position=({pose.position.x}, {pose.position.y}, {pose.position.z})'
                )
            else:
                self.get_logger().info('No posture detected.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    client = PostureDetectionClient()

    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

