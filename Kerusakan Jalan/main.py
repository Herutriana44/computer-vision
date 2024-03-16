file_path = "047.jpg"  # Change to your file path
mode = "" # camera or file

from ultralytics import YOLO
import cv2
import math

# Load the YOLO model
model = YOLO("best_yolov8l_kerusakan_jalan.pt")

# Object classes
classNames = ['alligator cracking', 'corrugation', 'edge cracking', 'pothole', 'rutting']


if mode == "camera":
    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 800)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # Loop through the detected results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Extract confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Extract class ID
                cls = int(box.cls[0])

                # Check if class ID is within classNames list
                if cls < len(classNames):
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Add class name and confidence as text
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, f"{classNames[cls]} ({confidence})", org, font, fontScale, color, thickness)
                    
                 

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to process an image or video file
def process_file(file_path):
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(file_path)
        results = model(file_path)
        # results = model(img)
        # img = cv2.imread('bus.jpg')
        while(True):
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                    # Extract confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Extract class ID
                    cls = int(box.cls[0])

                    # Check if class ID is within classNames list
                    if cls < len(classNames):
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # Add class name and confidence as text
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, f"{classNames[cls]} ({confidence})", org, font, fontScale, color, thickness)
                        cv2.imwrite("res_cam.jpg", img)
            cv2.imshow('File Processing', img)
            if cv2.waitKey(1) == ord('q'):
                break

        # cv2.imshow('File Processing', crop)

    elif file_path.lower().endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        cap.set(3, 800)
        cap.set(4, 800)

        while True:
            success, img = cap.read()
            if not success:
                break
            results = model(img)
            
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                    # Extract confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Extract class ID
                    cls = int(box.cls[0])

                    # Check if class ID is within classNames list
                    if cls < len(classNames):
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # Add class name and confidence as text
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, f"{classNames[cls]} ({confidence})", org, font, fontScale, color, thickness)

            cv2.imshow('File Processing', img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
    else:
        print(f"Unsupported file format for {file_path}")

process_file(file_path)
cv2.destroyAllWindows()