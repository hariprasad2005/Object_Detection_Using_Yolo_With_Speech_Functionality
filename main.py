import cv2
import torch
import pyttsx3
import sys
sys.path.append("E:/DL Mini Project")
from my_utils.speech import init_speech_engine, announce_detection
from model.yolo_model import load_model, detect_objects  # Adjusted for model folder
from my_utils.camera import start_camera, capture_frame  # Adjusted for utils folder

def main():
    # Initialize the YOLO model, speech engine, and camera
    model = load_model()
    engine = init_speech_engine()
    cap = start_camera()
    
    detected_objects = set()  # To avoid announcing the same object repeatedly
    
    while True:
        # Capture a frame from the camera
        frame = capture_frame(cap)
        
        if frame is None:
            break
        
        # Detect objects in the frame using YOLO
        results = detect_objects(model, frame)
        
        # Loop through each detected object
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            label = results.names[int(class_id)]
            
            # Announce new objects (avoid repeating the same object)
            if label not in detected_objects:
                announce_detection(engine, label)
                detected_objects.add(label)
                
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show the frame with detection
        cv2.imshow("YOLO Object Detection with Speech", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
