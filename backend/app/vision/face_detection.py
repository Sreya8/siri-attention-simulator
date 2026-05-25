import cv2
import mediapipe as mp

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create face detector
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to RGB frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Frame
    results = face_detection.process(rgb_frame)

    # Check if face was detected
    if results.detections:
        for detection in results.detections:
            # Draw bounding box
            mp_drawing.draw_detection(frame, detection)
    
    # Display Frame
    cv2.imshow("Face Detection", frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()