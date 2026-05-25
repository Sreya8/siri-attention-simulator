import cv2

# Open default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Display frame
    cv2.imshow("Camera Feed", frame)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()

# Close all windows
cv2.destroyAllWindows()