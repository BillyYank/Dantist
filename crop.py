
import PythonMagick
n = 1
for i in range(n):
    image = PythonMagick.Image("{}.jpg".format(i))
    image.crop(PythonMagick.Geometry(1500, 1000, 0, 00)) # w, h, x, y
    image.write('{}.tooth.jpg'.format(i))

