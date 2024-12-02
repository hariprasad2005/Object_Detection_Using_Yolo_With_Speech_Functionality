import cv2

def start_camera():
    # Start the video capture (use 0 for webcam)
    cap = cv2.VideoCapture(0)
    return cap

def capture_frame(cap):
    # Capture a single frame from the video feed
    ret, frame = cap.read()
    return frame
