# Object_Detection_Using_Yolo_With_Speech_Functionality
README.md

Object Detection with Speech Functionality

This project implements real-time object detection using the YOLOv5 model and integrates a speech functionality to announce detected objects. The system uses a webcam to capture video frames, detects objects in the frames using YOLOv5, and provides audio feedback for newly detected objects.


---

Features

Real-time Object Detection: Detects objects in video streams using YOLOv5.

Audio Feedback: Announces the names of detected objects using text-to-speech functionality.

80 Object Categories: Supports object detection from the COCO dataset with pre-trained weights.

Interactive Display: Shows bounding boxes and labels on the detected objects.



---

Tech Stack

Programming Language: Python

Libraries Used:

PyTorch: For loading and running the YOLOv5 model.

OpenCV: For video processing and displaying results.

pyttsx3: For converting text to speech.




---

Dataset

The YOLOv5 model is pre-trained on the COCO (Common Objects in Context) dataset, which includes 80 common object classes such as person, car, bicycle, dog, and more.


---

Installation

1. Clone the Repository

git clone https://github.com/<your-username>/Object-Detection-With-Speech.git  
cd Object-Detection-With-Speech


2. Install Dependencies
Ensure you have Python 3.8+ installed. Run the following command to install required libraries:

pip install -r requirements.txt

Contents of requirements.txt:

torch  
torchvision  
opencv-python  
pyttsx3


3. Download YOLOv5
The YOLOv5 model is loaded dynamically from the Ultralytics GitHub repository. Ensure an active internet connection during the first run.


4. Add Project Directory to Python Path
Modify the sys.path in the code if required to include your project's directory for importing custom modules.




---

Usage

1. Run the Script
Start the application by running the main script:

python main.py


2. Press q to Quit
Use the 'q' key to stop the application.




---

Code Overview

1. Loading YOLOv5 Model

Loads the YOLOv5 pre-trained model from the Ultralytics repository:

def load_model():  
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')  
    return model

2. Object Detection

Processes video frames to detect objects and returns detection results:

def detect_objects(model, frame):  
    results = model(frame)  
    return results

3. Speech Announcement

Converts detected object labels into speech:

def announce_detection(engine, label):  
    engine.say(f"{label} detected")  
    engine.runAndWait()

4. Main Logic

Combines the camera feed, object detection, and speech functionalities:

def main():  
    model = load_model()  
    engine = init_speech_engine()  
    cap = start_camera()  

    detected_objects = set()  

    while True:  
        frame = capture_frame(cap)  
        if frame is None:  
            break  

        results = detect_objects(model, frame)  
        for detection in results.xyxy[0]:  
            x1, y1, x2, y2, confidence, class_id = detection.tolist()  
            label = results.names[int(class_id)]  

            if label not in detected_objects:  
                announce_detection(engine, label)  
                detected_objects.add(label)  

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

        cv2.imshow("YOLO Object Detection with Speech", frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()


---

Customization

Object Detection Model: You can switch to other YOLOv5 variants (yolov5m, yolov5l, etc.) for improved accuracy or speed. Update the following line:

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', source='github')

Confidence Threshold: Adjust the detection confidence threshold by modifying the YOLO model parameters.



---

License

This project is licensed under the MIT License. See the LICENSE file for details.


---

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


---

Happy Coding!
