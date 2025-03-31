from inference import get_model
import supervision as sv
import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

filename = "/home/ros/rap/leakage.jpg"

# create a client object
client = InferenceHTTPClient(
    api_url="http://160.85.253.140:30334",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

image = cv2.imread(filename)

results = client.infer(image, model_id="puddle-detection/8")

print(results)

detections = sv.Detections.from_inference(results)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)