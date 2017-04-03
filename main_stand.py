# -*- coding: utf-8 -*-
import PIL
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib import colors
from PIL import Image
import colorsys
import cognitive_face as CF # нужно подключить апи когнитив сервиса
import shapely.geometry   # для создания маски
import sys
import cv2
import pylab
import matplotlib.image as mpimg

cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("preview", cv2.WINDOW_AUTOSIZE)
#cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#cv2.namedWindow('loading_img')
vc = cv2.VideoCapture(0)

window_size = (2560 / 2, 1600 / 2)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", cv2.resize(frame, window_size))
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 13:
        retval, im = vc.read()
        im = cv2.resize(im, (window_size[0] / 2, window_size[1] / 2))
        cv2.imwrite("my_fancy_smile.jpg", im)
        
        img_path = "my_fancy_smile.jpg"
        save_path = "white_smile.jpg"
        
        
        #image = mpimg.imread("loading.jpg")
        #plt.imshow(image)
        #plt.show()
        cv2.imshow("preview", cv2.resize(cv2.imread('loading.jpg'), window_size))
        key = cv2.waitKey(20)
        
        
        KEY = 'b2c2405308414acea246dd51629aeef4 '  # MS ключ на мой акк
        CF.Key.set(KEY) #установка ключа

        print 'reading iamge'
        FileData = open(img_path, "rb")               # не очень понимаю что это но без этого не работает.
        print 'detecting face'
        result = CF.face.detect(FileData, face_id = False, landmarks = True)              # запрос
        print result

        image = PIL.Image.open(img_path) # it will be used later
        print 'Image size:', image.size
        width, height = image.size
        size = (height, width)

        mask_mouth = np.zeros(size).astype(bool)

        for index in range(len(result)):
            print 'face {} of {}'.format(index + 1, len(result)) 
        # создание словаря rot
            rot = {'mouthRight':result[index]['faceLandmarks']['mouthRight'],       # правый край рта
            'mouthLeft':result[index]['faceLandmarks']['mouthLeft'],           # левый край рта
            'upperLipBottom':result[index]['faceLandmarks']['upperLipBottom'],         # верхний кончик верхней губы, можно нижний сделать
            'underLipBottom':result[index]['faceLandmarks']['underLipBottom']}      # нижний кончик нижней губы

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


            min_x = int(min(zip(*vertex)[0]))
            max_x = int(max(zip(*vertex)[0]))
            min_y = int(min(zip(*vertex)[1]))
            max_y = int(max(zip(*vertex)[1]))
            print 'min_x', min_x
            print 'max_x', max_x
            print 'min_y', min_y
            print 'max_y', max_y

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
        
        res_img = np.concatenate((cv2.imread(img_path), cv2.imread(save_path)), axis=1)
        resized = cv2.resize(res_img, (window_size[0], window_size[1]/2))
        disp = cv2.resize(np.zeros_like(cv2.imread(img_path)), window_size)
        disp[window_size[1] / 4 : window_size[1] / 4 + resized.shape[0], :] = resized
        cv2.imshow("preview", disp)
        key = cv2.waitKey()
        

vc.release()
cv2.destroyWindow("preview")
