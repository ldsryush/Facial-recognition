import cv2
import face_recognition
import os
#code from face_recognition.py
script_dir = os.path.dirname(__file__)
known_faces_dir = os.path.join(script_dir, "../known_faces")

known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    filepath = os.path.join(known_faces_dir, filename)
    img = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splittext(filename)[0])
#video stream starts

video_capture = cv2.ViedoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

#Resizing the frame to 1/4 resolution

    small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1] #changing BGR to RGB

    #Find all the faces and encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        #Match faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        #close match if its reasonable distance

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
        #scale face back to original size
        top, right, bottom, left = [v * 4 for v in face_location]
        #draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)

    cv2.imshow("Live Face Recognition", frame)

    #quitting program by pressing q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

