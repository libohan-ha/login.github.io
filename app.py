import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np # Added for PIL to OpenCV conversion
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, abort, render_template # Added render_template

from ultralytics import YOLO
from fire_detector import detect_fire_in_image, draw_detections_on_image

# --- Global Setup ---
UPLOADS_DIR = "uploads"
PROCESSED_VIDEOS_DIR = "processed_videos"

# Load the YOLO model
MODEL_PATH = "best.pt"  # User must provide this model file
MODEL = None
try:
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please download 'best.pt' from TommyNgx/YOLOv10-Fire-and-Smoke-Detection on Hugging Face")
        print("and place it in the root directory or provide the correct path.")
        # For a real application, you might exit or raise an error here if model is critical.
        # For now, we'll let it proceed, and endpoints will fail if MODEL is None.
    else:
        MODEL = YOLO(MODEL_PATH)
        print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model from '{MODEL_PATH}': {e}")
    MODEL = None # Ensure model is None if loading fails

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Max upload size: 32MB for video

# --- Directory Creation ---
try:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
    print(f"Directories '{UPLOADS_DIR}' and '{PROCESSED_VIDEOS_DIR}' ensured.")
except OSError as e:
    print(f"Error creating directories: {e}")
    # Depending on severity, you might want to exit or raise an error
    # For now, we print the error and continue; endpoints might fail if dirs are not writable.

# --- Helper Function for Error Response ---
def make_error_response(message, status_code):
    response = jsonify({"status": "error", "message": message})
    response.status_code = status_code
    return response

# --- Root Endpoint to Serve Frontend ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoint: /api/upload_video ---
@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if MODEL is None:
        return make_error_response("Model not loaded. Cannot process video.", 500)

    if 'video' not in request.files:
        return make_error_response("No video file part in the request.", 400)
    
    file = request.files['video']
    if file.filename == '':
        return make_error_response("No selected video file.", 400)

    if file:
        try:
            filename = secure_filename(file.filename)
            video_path = os.path.join(UPLOADS_DIR, filename)
            file.save(video_path)
            print(f"Video saved to {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return make_error_response("Error opening video file.", 500)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_filename = f"processed_{filename}"
            output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)
            
            # Using 'mp4v' for .mp4 files, or 'XVID' for .avi
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Processing video: {filename} (FPS: {fps}, Size: {width}x{height})")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # Convert BGR (OpenCV) frame to RGB, then to PIL.Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Perform detection
                detections = detect_fire_in_image(pil_image, MODEL) # Pass PIL image and loaded model
                
                # Draw detections
                processed_pil_image = draw_detections_on_image(pil_image, detections)
                
                # Convert processed PIL Image back to BGR OpenCV frame
                processed_bgr_frame = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)
                
                out.write(processed_bgr_frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows() # Should not be needed for server-side processing
            print(f"Video processing complete. Output: {output_video_path}, Frames processed: {frame_count}")

            return jsonify({"status": "success", "processed_video_filename": output_filename})

        except werkzeug.exceptions.RequestEntityTooLarge:
            return make_error_response(f"Video file too large. Max size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB.", 413)
        except Exception as e:
            print(f"Error during video processing: {e}")
            return make_error_response(f"An error occurred during video processing: {str(e)}", 500)
        finally:
            # Clean up uploaded file after processing if desired, or keep it.
            # For now, we keep it. If you want to delete:
            # if os.path.exists(video_path):
            #     os.remove(video_path)
            pass
            
    return make_error_response("Unknown error handling video upload.", 500)


# --- API Endpoint: /download_video/<filename> ---
@app.route('/download_video/<path:filename>', methods=['GET'])
def download_video(filename):
    try:
        return send_from_directory(PROCESSED_VIDEOS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        abort(404, description="File not found.")
    except Exception as e:
        print(f"Error downloading video {filename}: {e}")
        return make_error_response("Error occurred while trying to download the video.", 500)

# --- API Endpoint: /api/detect_camera_frame ---
@app.route('/api/detect_camera_frame', methods=['POST'])
def detect_camera_frame_api():
    if MODEL is None:
        return make_error_response("Model not loaded. Cannot process frame.", 500)

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return make_error_response("No image data provided in JSON payload.", 400)

        base64_image_data = data['image']
        
        # Clean base64 header (e.g., "data:image/jpeg;base64,")
        if ',' in base64_image_data:
            header, base64_string = base64_image_data.split(',', 1)
        else:
            base64_string = base64_image_data # Assume no header if no comma

        # Decode base64 string
        try:
            img_bytes = base64.b64decode(base64_string)
        except base64.binascii.Error as e: # Specific error for bad base64
            return make_error_response(f"Invalid base64 string: {str(e)}", 400)

        # Create PIL Image from bytes
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Perform detection
        detections = detect_fire_in_image(pil_image, MODEL) # Pass PIL image and loaded model
        
        # Draw detections
        processed_pil_image = draw_detections_on_image(pil_image, detections)
        
        # Convert processed PIL Image back to base64 string
        buffered = io.BytesIO()
        processed_pil_image.save(buffered, format="JPEG") # You can use PNG too
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Construct the full base64 data URI
        output_base64_image = f"data:image/jpeg;base64,{img_str}"

        return jsonify({"status": "success", "image": output_base64_image, "detections": detections})

    except Exception as e:
        print(f"Error in /api/detect_camera_frame: {e}")
        return make_error_response(f"An error occurred during frame detection: {str(e)}", 500)

# --- Main Execution Block ---
if __name__ == '__main__':
    # Ensure that the MODEL is loaded before starting the app if it's critical
    if MODEL is None:
        print("CRITICAL: YOLO Model could not be loaded. The application might not function correctly.")
        # Depending on policy, you might want to exit:
        # exit(1) 
    
    app.run(debug=True, host='0.0.0.0', port=5000)
