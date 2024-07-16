from ultralytics import YOLO
import os
import pyttsx3
import cv2
import torch

# Load YOLO model
model = YOLO('yolov8m.pt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak detected objects
def speak1(text):
    engine.say(text)
    engine.runAndWait()

# Capture webcam video
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model.predict(source=frame, conf=0.4, verbose=False)

        # Get detected object labels
        detected_objects = []
        for result in results:
            class_indices = result.boxes.cls if isinstance(result.boxes.cls, torch.Tensor) else torch.tensor(result.boxes.cls)
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                label = result.names[int(class_indices[i])]  # Get the label for the detected object
                detected_objects.append(label)

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert detected object labels to speech
        if detected_objects:
            objects_text = ', '.join(detected_objects)
            speak1(f"Detected objects are: {objects_text}")

        # Display the frame with bounding boxes
        cv2.imshow('YOLO', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
