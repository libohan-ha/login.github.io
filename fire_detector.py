from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from typing import Union, List, Dict, Any # For type hinting

# Define the detection function
def detect_fire_in_image(image_input: Union[str, Image.Image], model_object: Any) -> List[Dict[str, Any]]:
    """
    Detects fire and smoke in an image using a pre-loaded YOLO model object.

    Args:
        image_input (Union[str, Image.Image]): Path to the input image or a PIL Image object.
        model_object (Any): Pre-loaded YOLO model object (e.g., from YOLO("model.pt")).

    Returns:
        list: A list of dictionaries, where each dictionary represents a detection
              and contains 'class_name', 'confidence', and 'bounding_box'.
              Returns an empty list if no objects are detected or if an error occurs.
    """
    detections = []
    try:
        # Use the pre-loaded model object
        # Perform inference on the image
        # model() or model.predict() can take a path or a PIL Image object
        results = model_object(image_input, verbose=False) # verbose=False for cleaner output

        # Process the results
        if results and len(results) > 0:
            result = results[0]  # Get the results for the first image/frame
            
            # Ensure names are available (class names)
            class_names = model_object.names if hasattr(model_object, 'names') and model_object.names else \
                          (result.names if hasattr(result, 'names') and result.names else {0: 'object', 1: 'fire', 2: 'smoke'}) # Fallback

            for box in result.boxes:
                bounding_box = box.xyxy[0].tolist()
                
                # Confidence score
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names.get(class_id, f"class_{class_id}")

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": bounding_box
                })
        else:
            # If image_input is a path, use it in the message
            image_name = image_input if isinstance(image_input, str) else "the provided image"
            print(f"No results from model for {image_name}")

    except FileNotFoundError: # This will only be caught if image_input was a path
        print(f"Error: Image file not found at {image_input}")
    except Exception as e:
        print(f"An error occurred during detection: {e}")
    
    return detections

def draw_detections_on_image(image_object: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """
    Draws bounding boxes and labels on a PIL Image object based on detections.

    Args:
        image_object (Image.Image): The PIL Image to draw on.
        detections (List[Dict[str, Any]]): A list of detection dictionaries from detect_fire_in_image.

    Returns:
        Image.Image: The PIL Image object with detections drawn.
    """
    draw = ImageDraw.Draw(image_object)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for obj in detections:
        box = obj['bounding_box']
        label = f"{obj['class_name']} ({obj['confidence']:.2f})"
        
        draw.rectangle(box, outline="red", width=3)
        
        text_position = (box[0], box[1] - 15 if box[1] - 15 > 0 else box[1] + 1)
        
        if hasattr(draw, 'textbbox'):
            text_bbox_dims = draw.textbbox((0,0), label, font=font)
            text_width = text_bbox_dims[2] - text_bbox_dims[0]
            text_height = text_bbox_dims[3] - text_bbox_dims[1]
        else: 
            text_width, text_height = draw.textsize(label, font=font)
        
        label_bg_coords = [text_position[0], text_position[1], text_position[0] + text_width + 2, text_position[1] + text_height + 2]
        draw.rectangle(label_bg_coords, fill="red")
        draw.text(text_position, label, fill="white", font=font)
    
    return image_object

# Main section for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fire and smoke in an image using a YOLO model.")
    parser.add_argument("-m", "--model_path", 
                        type=str, 
                        required=True, 
                        help="Path to the pre-trained YOLO model file (e.g., best.pt or a Hugging Face repo ID).")
    parser.add_argument("-i", "--image_path", 
                        type=str, 
                        required=True, 
                        help="Path to the input image.")
    parser.add_argument("-o", "--output_path", 
                        type=str, 
                        default="output.jpg", 
                        help="Path to save the output image with detections (default: output.jpg).")
    
    args = parser.parse_args()

    # Check if the image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'.")
        print("Please provide a valid path to an image file.")
        exit()
        
    try:
        # Load the YOLO model once
        print(f"Attempting to load model from: {args.model_path}")
        model = YOLO(args.model_path)
        print("Model loaded successfully.")

        # Load the image using PIL
        print(f"Attempting to load image from: {args.image_path}")
        input_image_pil = Image.open(args.image_path).convert("RGB")
        print("Image loaded successfully.")

        # Call the detection function
        # Pass the image path or PIL image to the detection function
        detected_objects = detect_fire_in_image(input_image_pil, model) # Or pass args.image_path

        if detected_objects:
            print(f"\nDetected objects in '{args.image_path}':")
            for obj in detected_objects:
                print(f"  Class: {obj['class_name']}, Confidence: {obj['confidence']:.2f}, BBox: {[round(c, 2) for c in obj['bounding_box']]}")
            
            # Draw detections on the image
            output_image_pil = draw_detections_on_image(input_image_pil, detected_objects)
            
            # Save the output image
            output_image_pil.save(args.output_path)
            print(f"\nOutput image with detections saved as '{args.output_path}'")
            # output_image_pil.show() # Optionally display the image
        else:
            print(f"No objects detected in '{args.image_path}'.")

    except FileNotFoundError: # Should primarily be for model if image is pre-checked
        print(f"Error: Model file not found at {args.model_path} (or image at {args.image_path} if not caught above).")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
            
    # elif os.path.exists(IMAGE_PATH): # Check if image existed but no objects found
                                     # Model path check is tricky if it's a HF ID
        # print(f"No objects detected in '{IMAGE_PATH}'.")
    # else:
        # Error messages for missing files were already printed or handled by YOLO
        # print("Detection process could not be completed (see previous errors).")

print("\nScript finished.")
