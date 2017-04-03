# -*- coding: utf-8 -*-
import PIL
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib import colors
import Image
import colorsys
import cognitive_face as CF # нужно подключить апи когнитив сервиса
import shapely.geometry   # для создания маски
import sys

img_path = sys.argv[1]
save_path = sys.argv[2]

## Detect mouth
KEY = 'b2c2405308414acea246dd51629aeef4 '  # MS ключ на мой акк
CF.Key.set(KEY) #установка ключа

print 'reading iamge'
FileData = open(img_path, "rb")               # не очень понимаю что это но без этого не работает.
print 'detecting face'
result = CF.face.detect(FileData, face_id = False, landmarks = True)              # запрос
print result

# создание словаря rot
rot = {'mouthRight':result[0]['faceLandmarks']['mouthRight'],       # правый край рта
'mouthLeft':result[0]['faceLandmarks']['mouthLeft'],           # левый край рта
'upperLipBottom':result[0]['faceLandmarks']['upperLipBottom'],         # верхний кончик верхней губы, можно нижний сделать
'underLipBottom':result[0]['faceLandmarks']['underLipBottom']}      # нижний кончик нижней губы

# Округление
rot['mouthRight']['x']=round(rot['mouthRight']['x'])
rot['mouthRight']['y']=round(rot['mouthRight']['y'])
rot['upperLipBottom']['x']=round(rot['upperLipBottom']['x'])
rot['upperLipBottom']['y']=round(rot['upperLipBottom']['y'])
rot['mouthLeft']['x'] = round(rot['mouthLeft']['x'])
rot['mouthLeft']['y'] = round(rot['mouthLeft']['y'])
rot['underLipBottom']['x'] = round(rot['underLipBottom']['x'])
rot['underLipBottom']['y'] = round(rot['underLipBottom']['y'])

print 'creating mask'
# создание полигона
right, top, left, bottom = map(np.array, [
    (rot['mouthRight']['x'],rot['mouthRight']['y']),
    (rot['upperLipBottom']['x'],rot['upperLipBottom']['y']),
    (rot['mouthLeft']['x'],rot['mouthLeft']['y']),
    (rot['underLipBottom']['x'],rot['underLipBottom']['y'])])

vertex = [left, top, right, 
    (right[0]*0.5 + bottom[0] * 0.5, right[1] * 0.3 + bottom[1] * 0.7),
    bottom,
    (left[0]*0.5 + bottom[0] * 0.5, left[1] * 0.3 + bottom[1] * 0.7),
    ]

poly = shapely.geometry.Polygon(vertex)

#Определение размеров изображения
image = PIL.Image.open(img_path) # it will be used later
print 'Image size:', image.size
width, height = image.size
size = (height, width)

min_x = int(min(zip(*vertex)[0]))
max_x = int(max(zip(*vertex)[0]))
min_y = int(min(zip(*vertex)[1]))
max_y = int(max(zip(*vertex)[1]))
print 'min_x', min_x
print 'max_x', max_x
print 'min_y', min_y
print 'max_y', max_y


# создание маски
mask_mouth = np.zeros(size).astype(bool)
for i in range(min_y, max_y + 1):
    for j in range(min_x, max_x + 1):
        point = shapely.geometry.Point(j, i)
        mask_mouth[i, j] = point.intersects(poly)

#plt.imshow(mask_mouth, interpolation='nearest', cmap = 'bone')
#plt.show()

print 'mask is done'
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

print 'to hsv'
img = image.convert('RGBA')
arr = np.array(np.asarray(img).astype('float'))
r, g, b, a = np.rollaxis(arr, axis=-1)
h, s, v = rgb_to_hsv(r, g, b)

print 'transform'
####
l1, r2 = 15., 100.  
f = (h * 360 > l1) & (h * 360 < r2) & mask_mouth

f_blured = (1 - gaussian_filter(f.astype(float), sigma=5))    
s *= f_blured
v_max = 250
v = v_max - (v_max - v) * (1 - gaussian_filter(f.astype(float), sigma=5) * 0.3)
###

#plt.imshow(f, interpolation='nearest', cmap = 'bone')
#plt.figure()
#plt.imshow(gaussian_filter(f.astype(float), sigma=10), interpolation='nearest', cmap = 'bone')
print 'to rgb'
r, g, b = hsv_to_rgb(h, s, v)
arr = np.dstack((r, g, b, a))
new_img = Image.fromarray(arr.astype('uint8'), 'RGBA')
new_img.save(save_path)
print 'done'
