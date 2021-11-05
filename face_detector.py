import cv2
# Grabs the haarcascade XML file so we can use it as a Model
face_cascade = cv2.CascadeClassifier("./Files/haarcascade_frontalface_default.xml")
# Load img 
img=cv2.imread('./Files/news.jpg')
# Convert img to grayscale (You can accomplish this with 0 but we want to show the color img at the end, Its better to use a black and white img to search for faces)
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Method searches for cascade classifier in our img and return the number of the row and the column of the upper left point of the face 
# Also give you height and width of the face and then we can draw a rectangle over the face
# Store Width and Height (if you lower scale factor you can get rid of hand on news photo)
faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# loop through faces array to access coord
for x,y,width,height in faces:
    # draw rectangle, 1st arg starting point of rectangle (top-left), 2nd arg coord of other corner (bottom-right), 3rd Color, 4th Width of rectangle
    img=cv2.rectangle(img, (x,y),(x + width, y + height),(0,255,0),3)


# Resize img width is 0 and height is 1
resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
print(faces)
print(type(faces))
# Show img
cv2.imshow('Face Detected', resized)
# press any key it closes window
cv2.waitKey(0)
cv2.destroyAllWindows()