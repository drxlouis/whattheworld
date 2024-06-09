import cv2
import os
import subprocess
import threading
import time
from roboflow import Roboflow
from ultralytics import YOLO

# Function to use macOS 'say' command for TTS
def say(text):
    subprocess.run(['say', text])

# Initialize Roboflow and download the model
rf = Roboflow(api_key="1dhMdCURvopTGznlVenU")
project = rf.workspace("drxlouis").project("technoproject")
version = project.version(3)
dataset = version.download("yolov8")

# Correct the path for the model file
model_path = './last.pt'

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

# Load YOLOv8 model
model = YOLO(model_path)

# Specify the classes you want to detect
# allowed_classes = {"cell phone", "art", "human", "pointing", "pointing finger",
#                    "finger", "hand", "mobile", "artwork", "person"}
allowed_classes = {"art", "human", "phone", "pointing", "person", "cell phone"}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variable to store detected objects
detected_objects = set()

def detect_objects(frame):
    global detected_objects
    results = model(frame)
    current_objects = set()

    for result in results:
        for obj in result.boxes.data.tolist():
            class_id = int(obj[5])
            class_name = model.names[class_id]  # Accessing class names from the model
            if class_name in allowed_classes:
                current_objects.add(class_name)
    
    detected_objects = current_objects
    return detected_objects, results

def commentate(detected_objects):
    if detected_objects:
        commentary = f"Detected {', '.join(detected_objects)}"
        print(commentary)
        say(commentary)

def output_detected_objects():
    while True:
        if detected_objects:
            output_string = f"Objects detected in the last 30 seconds: {', '.join(detected_objects)}"
            print(output_string)
            # Write the detected objects to a file
            with open("detected_objects.txt", "w") as file:
                file.write(output_string)
        time.sleep(30)

# Start the thread to output detected objects every 30 seconds
threading.Thread(target=output_detected_objects, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_objects, results = detect_objects(frame)
    detection_string = ", ".join(detected_objects)
    print(f"Detected objects: {detection_string}")

    # Commentate the detected objects
    commentate(detected_objects)
    print(detected_objects)

    # Display the frame with detections
    for result in results:
        for obj in result.boxes.data.tolist():
            class_id = int(obj[5])
            class_name = model.names[class_id]
            if class_name in allowed_classes:
                x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                conf = obj[4]
                label = f"{class_name} {conf:.1f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
