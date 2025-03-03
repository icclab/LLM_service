#!/usr/bin/env python3

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
#from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

from llm.srv import CheckPosture
from llm.srv import CheckImage


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB') #PILImage.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class ImageSub(Node):
    def __init__(self):
        super().__init__('image_sub')
        self.bridge = CvBridge()  # Initialize CvBridge to convert ROS Image messages to OpenCV images

        # Load the InternVL2 model
        path = 'OpenGVLab/InternVL2-8B'
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

        # Create the ROS 2 service
        self.service_posture = self.create_service(CheckPosture, 'check_posture', self.handle_check_posture)
        self.service_leakage = self.create_service(CheckImage, 'check_leakage', self.handle_check_leakage)

    def handle_check_posture(self, request, response):
        self.get_logger().info('Received request to check posture.')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Detect posture in the received image
        response.posture_detected = self.detect_posture(pil_image)

        return response

    def detect_posture(self, pil_image):

        # Load the image and preprocess
        pixel_values = load_image(pil_image, max_num=12).to(torch.bfloat16).cuda()

        # Ask the model about the posture
        # question = '<image>\nPlease check the image properly, if there is any posture, find it and tell me in the percentage of the pixels of the image, if no, just simply answer NO without anything extra..'
        question = '<image>\nPlease describe the image, if there is a person, alert me by answering YES, there is a person!! And tell me if he or she is standing or sitting or lying down'
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)

        self.get_logger().info(f'User: {question}\nAssistant: {response}')

        # Determine if there is a posture
        response_lower = response.lower()

        if "standing" in response_lower:
            return 1
        elif "sitting" in response_lower:
            return 2
        elif "lying" in response_lower:
            return 3
        else:
            return 0  # Default value if none of the words are found

    def handle_check_leakage(self, request, response):
        self.get_logger().info('Received request to check leakage.')

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Detect leakage in the received image
        response.leakage_detected = self.detect_leakage(pil_image)

        return response

    def detect_leakage(self, pil_image):

        # Load the image and preprocess
        pixel_values = load_image(pil_image, max_num=12).to(torch.bfloat16).cuda()

        # Ask the model about the leakage
        question = '<image>\nPlease check the image properly, if there is any leakage, find it and tell me the percentage of the pixels of the image, if no, just simply answer NO without anything extra..'
        response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)

        self.get_logger().info(f'User: {question}\nAssistant: {response}')

        # Determine if there is a leakage
        return "leakage" in response.lower()


def main(args=None):
    rclpy.init(args=args)
    image_sub = ImageSub()

    rclpy.spin(image_sub)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


