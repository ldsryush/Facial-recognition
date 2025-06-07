import face_recognition
import os

known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{filename}")
    encoding = face_recognition.face_encodings(img)[0]

known_encodings.append(encoding)
known_names.append(filename.split(".")[0])
