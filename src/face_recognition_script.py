import face_recognition
import os

script_dir=os.path.dirname(__file__)
known_faces_dir = os.path.join(script_dir, "../known_faces")

known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    filepath = os.path.join(known_faces_dir, filename)
    if not os.path.isfile(filepath):
        continue

    img = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(img)
if encodings:
    encoding = encodings[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(filename)[0])
else:
    print(f"No face found")

print("loaded images: ")
for name, encoding in zip(known_names, known_encodings):
    print(f"{name} | Encoding length: {len(encoding)}")


