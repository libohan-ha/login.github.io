# YOLO-based Fire Detection System

This system utilizes a pre-trained YOLO (You Only Look Once) model to detect instances of fire and smoke within images. It is designed to be a command-line tool that processes an input image and outputs an image with detections highlighted, along with console information about the detected objects.

### 1. Overview
   - **Brief description:** This system uses a pre-trained YOLO model to detect fire and smoke in images.
   - **Model used:** "[TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection)" from Hugging Face. This model is based on a YOLOv8/v10 architecture and is fine-tuned for fire and smoke detection.

### 2. Setup and Installation

#### Prerequisites:
   - Python 3.x (developed with Python 3.8+)

#### Dependencies:
   - The system requires the following Python libraries:
     - `Flask`: For the web application framework.
     - `ultralytics`: For the YOLO model framework and inference.
     - `Pillow`: For image manipulation (drawing bounding boxes).
     - `opencv-python`: For video processing (reading and writing video frames).
     - `numpy`: For numerical operations, often a dependency for OpenCV and image processing.
   - Install the dependencies using pip:
     ```bash
     pip install Flask ultralytics Pillow opencv-python numpy
     ```

#### Download the Pre-trained Model:
   - Download the model file `best.pt` from the Hugging Face model repository:
     [TommyNgx/YOLOv10-Fire-and-Smoke-Detection/tree/main](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection/tree/main)
     (You may need to navigate to the "Files and versions" tab and agree to the terms to download the file).
   - **Crucially, place the downloaded `best.pt` file in the root directory of this project.** The `app.py` and `fire_detector.py` scripts expect to find it there by default.

#### Expected Directory Structure:
   For the application to run correctly, your project directory should look like this:
   ```
   /your_project_root_directory/
   |-- app.py                     # Flask application
   |-- fire_detector.py           # Detection logic script
   |-- best.pt                    # Downloaded YOLO model file
   |-- README.md                  # This file
   |-- /templates/
   |   |-- index.html             # Frontend HTML
   |-- /static/
   |   |-- style.css              # Frontend CSS
   |   |-- script.js              # Frontend JavaScript
   |-- /uploads/                  # Created automatically for uploaded videos
   |-- /processed_videos/         # Created automatically for processed videos
   ```

### 3. Usage

This project offers two main ways to perform fire and smoke detection: a command-line script for individual images and a web application for video upload and live camera processing.

#### Running the Web Application (`app.py`)

1.  **Start the Flask application:**
    Open your terminal, navigate to the project's root directory, and run:
    ```bash
    python app.py
    ```
2.  **Access the web interface:**
    Open your web browser and go to:
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

#### Web Application Features:

*   **Video Upload:**
    -   Upload a video file through the web interface.
    -   The backend processes the video frame by frame, performing fire and smoke detection.
    -   Once processing is complete, a download link for the annotated video (with detections drawn) is provided.
*   **Live Camera Detection:**
    -   Start your webcam directly from the browser.
    -   The application will capture frames, send them to the backend for processing, and display the live feed with fire and smoke detections overlaid in real-time.

---

#### Using the `fire_detector.py` Script (Command-Line Tool)

The `fire_detector.py` script is used to perform detection on a single image.

##### Command-Line Arguments:

*   `--model_path` / `-m` (Required):
    Path to the pre-trained YOLO model file (e.g., `best.pt`). You can also try using the Hugging Face model ID directly (e.g., `TommyNgx/YOLOv10-Fire-and-Smoke-Detection`) if `ultralytics` can resolve it automatically, though providing the local `.pt` file is recommended for reliability.
*   `--image_path` / `-i` (Required):
    Path to the input image file (e.g., `sample_fire_image.jpg`).
*   `--output_path` / `-o` (Optional):
    Path to save the output image with bounding boxes and labels drawn around detections. Defaults to `output.jpg` in the current directory if not specified.

#### Example Command:

```bash
python fire_detector.py -m best.pt -i path/to/your/fire_image.jpg -o results/detected_fire.jpg
```
Or, if `best.pt` is in the current directory:
```bash
python fire_detector.py --model_path best.pt --image_path my_image.png --output_path my_detected_image.png
```

#### Expected Output:

*   **Console Output:**
    The script will print information about the detected objects to the console, including:
    - Class name (e.g., "fire", "smoke")
    - Confidence score (how sure the model is about the detection)
    - Bounding box coordinates (`[x1, y1, x2, y2]`)
    ```
    Attempting to load model from: best.pt
    Attempting to load image from: path/to/your/fire_image.jpg

    Detected objects in 'path/to/your/fire_image.jpg':
      Class: fire, Confidence: 0.85, BBox: [150.0, 200.0, 250.0, 300.0]
      Class: smoke, Confidence: 0.72, BBox: [160.0, 150.0, 240.0, 220.0]

    Output image with detections saved as 'results/detected_fire.jpg'

    Script finished.
    ```

*   **Image Output:**
    An image file (e.g., `results/detected_fire.jpg`) will be saved at the specified output path. This image will be a copy of the input image with red bounding boxes drawn around the detected fire/smoke regions and labels indicating the class and confidence score.

### 4. Testing and Refinement

- **Testing Strategy:**
  To evaluate the system's effectiveness, test it with a diverse set of images:
    - Images with clear fire/smoke.
    - Images with ambiguous fire-like objects (e.g., sunsets, red lights) to check for false positives.
    - Images with small or obscured fire/smoke to check for false negatives.
    - Images from various environments and lighting conditions.
- **Refinement:**
  For optimal performance in specific environments or for detecting particular types of fire/smoke not well-represented in the original training data, fine-tuning the "TommyNgx/YOLOv10-Fire-and-Smoke-Detection" model (or another YOLO base model) on a custom dataset of relevant images is highly recommended. This involves collecting and annotating images specific to the target use case.

### 5. Future Improvements

This system can be extended and enhanced in several ways:

- **Real-time Video Detection:** Adapt the script to process video streams from a webcam or video files for real-time fire and smoke detection.
- **Integration with Alert Systems:** Connect the detection output to an alert mechanism (e.g., email notifications, SMS, alarms) for immediate response.
- **More Comprehensive Evaluation Metrics:** Implement detailed performance tracking using metrics like Precision, Recall, F1-score, and mAP on a dedicated test set.
- **User Interface:** Develop a graphical user interface (GUI) or a web interface for easier interaction, allowing users to upload images or configure video feeds without using the command line.
- **Configuration File:** Allow model parameters (confidence threshold, IoU threshold) to be set via a configuration file.

### 6. Notes

This system provides a starting point for fire detection using a pre-trained YOLO model. Its performance may vary based on image quality, the complexity of the scene, specific environmental conditions, and the characteristics of the pre-trained model. For critical applications, rigorous testing and potential fine-tuning are essential.

### 7. Troubleshooting / Common Issues

*   **Issue: Model file (`best.pt`) not found when running `app.py` or `fire_detector.py`.**
    *   **Solution:** Ensure `best.pt` is downloaded from the Hugging Face link provided in the "Download the Pre-trained Model" section and placed **directly in the project's root directory**. The application expects to find it there.

*   **Issue: Camera not working or permission denied in the web application.**
    *   **Solution:**
        1.  Check that your browser has permission to access the camera. You might need to grant permission when prompted by the browser, or adjust settings in your browser's privacy/security section for this website (127.0.0.1).
        2.  Ensure no other application (e.g., Zoom, Skype, another browser tab) is exclusively using the camera.
        3.  Try a different browser to see if the issue is browser-specific.

*   **Issue: Video upload or processing is slow.**
    *   **Note:** Fire detection on video is computationally intensive. Processing time will depend on the video's length, resolution, and the server's (your computer's) processing power. Live camera detection performance is also heavily dependent on your computer's CPU/GPU capabilities.

*   **Issue: Errors related to Python dependencies (e.g., "ModuleNotFoundError").**
    *   **Solution:** Ensure all dependencies listed in the "Setup and Installation > Dependencies" section are correctly installed in your Python environment. You can reinstall them using:
        ```bash
        pip install Flask ultralytics Pillow opencv-python numpy
        ```
        Consider using a virtual environment to manage project dependencies.

*   **Issue: "Address already in use" or "Port already in use" when starting `app.py`.**
    *   **Solution:** This means another application is already using port 5000.
        1.  Identify and stop the other application.
        2.  (Optional, advanced) If you know how, you can modify `app.py` to run on a different port, for example, by changing `app.run(debug=True, host='0.0.0.0', port=5000)` to `app.run(debug=True, host='0.0.0.0', port=5001)`. Remember to access the web app at the new port (e.g., `http://127.0.0.1:5001/`).
