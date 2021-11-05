import cv2
# Grabs the haarcascade XML file so we can use it as a Model
face_cascade = cv2.CascadeClassifier("./Files/haarcascade_frontalface_default.xml")
# Load img 
img=cv2.imread('./Files/photo.jpg')
# Convert img to grayscale (You can accomplish this with 0 but we want to show the color img at the end, Its better to use a black and white img to search for faces)
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Method searches for cascade classifier in our img and return the coordinates of the face in the img
# Show gray img
cv2.imshow('Gray', gray_img)
# press any key it closes window
cv2.waitKey(0)
cv2.destroyAllWindows()