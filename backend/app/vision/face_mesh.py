import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Configure Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks=True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture Frame")
        break

    #flip image horizontally for natural selfie view
    frame = cv2.flip(frame, 1)

    # conbert bgr to rgb
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # improve performance
    rgb_image.flags.writeable = False

    # process the image
    results = face_mesh.process(rgb_image)

    # Reenable writing
    rgb_image.flags.writeable = True

    # Draw Landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
               image=frame,
               landmark_list=face_landmarks,
               connections=mp_face_mesh.FACEMESH_TESSELATION,
               landmark_drawing_spec=None,
               connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style() 
            )
            # mp_drawing.draw_landmarks(
            #    image=frame,
            #    landmark_list=face_landmarks,
            #    connections=mp_face_mesh.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style() 
            # )
            # mp_drawing.draw_landmarks(
            #    image=frame,
            #    landmark_list=face_landmarks,
            #    connections=mp_face_mesh.FACEMESH_IRISES,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style() 
            # )
    
    cv2.imshow("Face Mesh", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()