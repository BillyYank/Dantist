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

# контрольный вывод
print(rot['mouthRight']['x'])
print(rot['mouthRight']['y'])
print(rot['upperLipBottom']['x'])
print(rot['upperLipBottom']['y'])
print(rot['mouthLeft']['x'])
print(rot['mouthLeft']['y'])
print(rot['underLipBottom']['x'])
print(rot['underLipBottom']['y'])

print 'creating mask'
# создание полигона
poly = shapely.geometry.Polygon([(rot['mouthRight']['x'],rot['mouthRight']['y']),
 (rot['upperLipBottom']['x'],rot['upperLipBottom']['y']),
 (rot['mouthLeft']['x'],rot['mouthLeft']['y']),
 (rot['underLipBottom']['x'],rot['underLipBottom']['y'])])

#Определение размеров изображения
image = PIL.Image.open(img_path) # it will be used later
print 'Image size:', image.size
width, height = image.size
size = (height, width)


# создание маски
mask_mouth = np.zeros(size).astype(bool)
for i in range(mask_mouth.shape[0]):
    for j in range(mask_mouth.shape[1]):
        point = shapely.geometry.Point(j, i)
        mask_mouth[i, j] = point.intersects(poly)


##
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
h1 = np.copy(h)
s1 = np.copy(s)
v1 = np.copy(v)
l1, l2, r1, r2 = 15., 20., 95., 100.  
f = (h * 360 > l1) & (h * 360 < r2) & mask_mouth

f_blured = (1 - gaussian_filter(f.astype(float), sigma=5))    
s1 *= f_blured
v_max = 250
v1 = v_max - (v_max - v1) * (1 - gaussian_filter(f.astype(float), sigma=5) * 0.3)
###

#plt.imshow(f, interpolation='nearest', cmap = 'bone')
#plt.figure()
#plt.imshow(gaussian_filter(f.astype(float), sigma=10), interpolation='nearest', cmap = 'bone')
print 'to rgb'
r, g, b = hsv_to_rgb(h1, s1, v1)
arr = np.dstack((r, g, b, a))
new_img = Image.fromarray(arr.astype('uint8'), 'RGBA')
new_img.save(save_path)
print 'done'
