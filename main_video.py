import cv2
from simple_facerec import SimpleFacerec

# Initialize the SimpleFacerec object
sfr = SimpleFacerec()
# Load encoded images from the "images/" directory
sfr.load_encoding_images("images/")

# Open a connection to the webcam (index 2 might be wrong, commonly it is 0 or 1)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        top, right, bottom, left = face_loc
        print(f"Location: {face_loc}, Name: {name}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
