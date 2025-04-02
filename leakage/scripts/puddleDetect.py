from inference import get_model
import supervision as sv
import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

filenames = [
    "/home/ros/rap/spillmilk.png",
    "/home/ros/rap/Spilledmilk.png",
    "/home/ros/rap/coke.png",
    "/home/ros/rap/floor.jpeg",
    "/home/ros/rap/leakage.jpg",
    "/home/ros/rap/milk.jpeg",
    "/home/ros/rap/floor2.jpeg",
    "/home/ros/rap/water2.jpg",
    "/home/ros/rap/spill.jpg",
    "/home/ros/rap/water4.jpeg",
    "/home/ros/rap/oil-leak.jpg",
]


# create a client object
client = InferenceHTTPClient(
    api_url="http://160.85.253.140:30334",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

for filename in filenames:
    print(f"\nProcessing image: {filename}")
    
    try:
        image = cv2.imread(filename)
        
        if image is None:
            print(f"Error: Could not load image {filename}")
            continue
        if image is None:
            print(f"Error: Could not load image {filename}")
            continue

        results1 = client.infer(image, model_id="water-leakage/2")
        results2 = client.infer(image, model_id="water-ba8zz/1")

	# Combine detections from both models
        combined_results = results1["predictions"] + results2["predictions"]

        if combined_results:
            print("detected Liquid!!!!!!!")

	     # Log detected classes
            for prediction in combined_results:
                print(
                    f"Detection: Class={prediction['class']}, Confidence={prediction['confidence']:.2f}, "
                    f"Bounding Box=[{prediction['x']}, {prediction['y']}, {prediction['width']}, {prediction['height']}]"
                )
        else:
            print('No leakage detected.')

        print(f"\nResults for {filename} with water-leakage/2 model:")
        print(results1)

        classes1 = [prediction["class"] for prediction in results1["predictions"]]
        for class_name in classes1:
            print(f'Detected class with water-leakage/2: {class_name}')

        print(f"\nResults for {filename} with water-ba8zz/1 model:")
        print(results2)

        classes2 = [prediction["class"] for prediction in results2["predictions"]]
        for class_name in classes2:
            print(f'Detected class with water-ba8zz/1: {class_name}')

        # load the results into the supervision Detections api
        detections1 = sv.Detections.from_inference(results1)
        detections2 = sv.Detections.from_inference(results2)

        # create supervision annotators
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotate the image with our inference results
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections1)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections1)
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections2)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections2)

        # display the image
        sv.plot_image(annotated_image)

    except Exception as e:
        print(f"Error processing image {filename}: {str(e)}")