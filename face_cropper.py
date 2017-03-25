import PythonMagick
import cv2
import sys

print 'initialize clissifier'
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
print 'initialized'

n = 2

tooth_id = 0

for i in range(n):
    imagePath = "{}.jpg".format(i)
    print 'working for {}'.format(imagePath)
    
    print 'reading image'
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print 'detecting face'
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    print 'cropping faces'
    for (x, y, w, h) in faces:
        image = PythonMagick.Image(imagePath)

        image.crop(PythonMagick.Geometry(int(w), int(h/2), int(x), int(y+ h/2))) # w, h, x, y

        image.write('{}.tooth.jpg'.format(tooth_id))
        tooth_id += 1

