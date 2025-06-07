import cv2 # OpenCV that helps with loading, processing, and visualizing images
import dlib # dlib for face detection 

image_path = "data/face.jpg"
image = cv2.imread(image_path)  #this loads the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # this converts the image into gray for faster speeds

detector = dlib.get_frontal_face_detector()
faces = detector(gray)

for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box

cv2.imwrite("output.jpg", image)  # Saves the processed image
cv2.waitKey(0) #wait for the key to be pressed by user
cv2.destroyAllWindows()
