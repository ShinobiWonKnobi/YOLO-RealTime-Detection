import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    
    # Run YOLO on the frame
    results = model(frame)

    # Display the frame
    cv2.imshow("YOLO Detection", results[0].plot())

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
