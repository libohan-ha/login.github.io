document.addEventListener('DOMContentLoaded', () => {
    // Video Upload Elements
    const videoUploadInput = document.getElementById('videoUpload');
    const uploadButton = document.getElementById('uploadButton');
    const videoUploadResult = document.getElementById('videoUploadResult');

    // Camera Detection Elements
    const startCameraButton = document.getElementById('startCameraButton');
    const stopCameraButton = document.getElementById('stopCameraButton');
    const cameraFeed = document.getElementById('cameraFeed'); // Raw feed, hidden
    const cameraCanvas = document.getElementById('cameraCanvas'); // Display processed feed
    const cameraStatus = document.getElementById('cameraStatus');
    const ctx = cameraCanvas.getContext('2d');
    let localStream = null;
    let processingInterval = null;

    // Video Upload Logic
    uploadButton.addEventListener('click', async () => {
        if (videoUploadInput.files.length === 0) {
            videoUploadResult.textContent = 'Please select a video file first.';
            return;
        }
        const file = videoUploadInput.files[0];
        const formData = new FormData();
        formData.append('video', file);
        videoUploadResult.textContent = 'Uploading and processing... please wait.';

        try {
            const response = await fetch('/api/upload_video', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok && data.status === 'success') {
                videoUploadResult.innerHTML = `Processing complete! <a href="/download_video/${data.processed_video_filename}" download>${data.processed_video_filename}</a>`;
            } else {
                videoUploadResult.textContent = `Error: ${data.message || 'Upload failed'}`;
            }
        } catch (error) {
            videoUploadResult.textContent = `Error: ${error.message}`;
        }
    });

    // Camera Detection Logic
    startCameraButton.addEventListener('click', async () => {
        if (localStream) { // Stop existing stream before starting a new one
            localStream.getTracks().forEach(track => track.stop());
        }
        if (processingInterval) {
            clearInterval(processingInterval);
        }

        try {
            localStream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraFeed.srcObject = localStream; 
            cameraStatus.textContent = 'Camera started. Initializing...';
            
            cameraFeed.onloadedmetadata = () => {
                cameraCanvas.width = cameraFeed.videoWidth;
                cameraCanvas.height = cameraFeed.videoHeight;
                cameraStatus.textContent = 'Processing frames...';

                processingInterval = setInterval(async () => {
                    if (cameraFeed.readyState < cameraFeed.HAVE_ENOUGH_DATA || cameraFeed.paused || cameraFeed.ended) {
                        return; 
                    }

                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = cameraFeed.videoWidth;
                    tempCanvas.height = cameraFeed.videoHeight;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(cameraFeed, 0, 0, tempCanvas.width, tempCanvas.height);
                    const base64ImageData = tempCanvas.toDataURL('image/jpeg');

                    try {
                        const response = await fetch('/api/detect_camera_frame', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: base64ImageData })
                        });
                        const data = await response.json();
                        if (response.ok && data.status === 'success') {
                            const outputImage = new Image();
                            outputImage.onload = () => {
                                ctx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height); // Clear before drawing
                                ctx.drawImage(outputImage, 0, 0, cameraCanvas.width, cameraCanvas.height);
                            };
                            outputImage.src = data.image;
                        } else {
                            cameraStatus.textContent = `Error processing frame: ${data.message || 'Unknown error'}`;
                        }
                    } catch (error) {
                        cameraStatus.textContent = `Error sending frame: ${error.message}`;
                        if (processingInterval) clearInterval(processingInterval); 
                    }
                }, 200); // Adjusted interval for potentially better performance balance
            };

        } catch (error) {
            cameraStatus.textContent = `Error starting camera: ${error.message}`;
            console.error('Error starting camera:', error);
        }
    });

    stopCameraButton.addEventListener('click', () => {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
        ctx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height); 
        cameraStatus.textContent = 'Camera stopped.';
        cameraFeed.srcObject = null; // Release the video element's source
    });
});
