
# coding: utf-8

# In[1]:


# importing packages

import more_itertools as mit
import cv2
import math
from skimage.io import imread
import matplotlib.pyplot as plt
from pytesseract import image_to_string
from skimage.color import rgb2gray
from skimage.morphology import convex_hull_image
from PIL import Image
import numpy as np
from PIL import ImageEnhance

from skimage import img_as_float


# In[3]:


#For name and face extraction

# Resizing Image
def crop_image(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # plt.imshow(image)
    # plt.show()
    isw_size=40
    zeropix=[]
    for i,arr in enumerate(image):
        for j,l in enumerate(arr):
            if image[i][j] == 255:
                zeropix.append((i,j))
    groups = {}
    for x, y in zeropix:
        groups.setdefault(y, []).append(x)
    heights_extracted = [list(group) for group in mit.consecutive_groups(groups.keys())]
    height_ranges = [[min(i),max(i),isw_size-(max(i)-min(i))] for i in heights_extracted if max(i)-min(i) in range(2,10) ]
    hrs_modified = []
    for i in height_ranges:
        if i[2] % 2 == 0:
            half1,half2 = [int(i[2]/2)]*2
        else: half1, half2 = int(math.floor(i[2]/2)), int(math.ceil(i[2]/2))
        hrs_modified.append([i[0]-half1, i[1]+half2])
    min_p=1000
    max_p=0
    for i in hrs_modified:
        if(min_p>min(i)):
            min_p=min(i)
        if(max_p<max(i)):
            max_p=max(i)
    return(min_p,max_p)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)

        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


# In[4]:


#Edge detection
def edgeDetection(img):
    im1 = 1 - rgb2gray(img)
    im1 = im1+0.25
    threshold = 0.7
    im1[im1 <= threshold] = 0
    im1[im1 > threshold] = 1
    chull = convex_hull_image(im1)
    imageBox = Image.fromarray((chull*255).astype(np.uint8)).getbbox()
    # print(imageBox)
    if imageBox[0]>=5:
        imBox = (imageBox[0]-5,imageBox[1],imageBox[2],imageBox[3])
    else:
        imBox =imageBox
    # svimg = im.fromarray(data.astype('uint8'))
    cropped = Image.fromarray(img).crop(imBox)
    return cropped




def face_extraction(image):
    image=cv2.imread(image)
    # try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #Make sure above xml file is in your directory
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = img[y-20:y+h+20, x-5:x+w+5]
    return roi
    # except:
    #     return -1


# In[7]:

def get_data(image_name):
    # Using above loaded modules
    print(image_name)
    img_path = image_name
    image = cv2.imread(img_path)
    image = image_resize(image, width=800)
    # print(image.shape)
    cv2.imwrite('resized.jpg', image)
    image = edgeDetection(image)
    # print(type(image))
    image = np.array(image)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width, channels = image.shape
    y_start=int(height*0.20)
    y_end=int(height*0.05)
    x_end=int(width*0.50)

    image=image[y_start:-y_end,:x_end]
    #image = image[110:,5:270]
    cv2.imwrite("name_and_face.jpg", image)
    #face extraction
    img=face_extraction('name_and_face.jpg')
    cv2.imwrite("Driving/media/documents/face.jpg", img)

    ims=[]
    # In[8]:


    # Name Extraction

    image=imread('name_and_face.jpg')
    # plt.imshow(img)
    # plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    zeropix = []
    # print(image)
    image=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    for i, arr in enumerate(image):
        for j, l in enumerate(arr):
            # if l > 80:
            #     image[i][j] = 255
            # else:
            #     image[i][j] = 0
            if image[i][j] == 0:
                zeropix.append((i, j))
    cv2.imwrite('pixcelnae.jpg', image)
    groups = {}

    for x, y in zeropix:
        groups.setdefault(x, []).append(y)

    # In[11]:

    isw_size = 20
    heights_extracted = [list(group) for group in mit.consecutive_groups(groups.keys())]
    # print(heights_extracted)
    height_ranges = [[min(i), max(i), isw_size - (max(i) - min(i))] for i in heights_extracted if
                     max(i) - min(i) in range(20, 40)]
    print(height_ranges)
    hrs_modified = []
    for i in height_ranges:
        if i[2] % 2 == 0:
            half1, half2 = [int(i[2] / 2)] * 2
        else:
            half1, half2 = int(math.floor(i[2] / 2)), int(math.ceil(i[2] / 2))
        hrs_modified.append([i[0] - half1, i[1] + half2])


    for i in hrs_modified:
        img = cv2.imread('name_and_face.jpg')
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(i)

        if i[0]>5:
            img = img[i[0]-10:i[1]+10, :]
            img = Image.fromarray(img)
            #plt.imshow(img)
            #plt.show()
            var1 = image_to_string(img, config='--psm 7')
            # allowed = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            #            'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')']
            # for chra in var1:
            #     if (chra not in allowed):
            #         var1 = var1.replace(chra, "")
            ims.append(var1)


    # In[9]:


    image=imread('resized.jpg')
    y_start1=int(height*0.2)
    y_end1=int(height*0.07)
    x_start1=int(width*0.36)
    image_other=image[y_start1:-y_end1,x_start1:]
    cv2.imwrite("other_fields.jpg", image_other)
    image=image[y_start1:-y_end1,x_start1:x_start1+150]
    #plt.imshow(image)
    #plt.show()
    # identifying other fields using black pixcel concept
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    zeropix = []
    # print(image)
    image=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    for i, arr in enumerate(image):
        for j, l in enumerate(arr):
            # if l > 80:
            #     image[i][j] = 255
            # else:
            #     image[i][j] = 0
            if image[i][j] == 0:
                zeropix.append((i, j))
    cv2.imwrite('pixcel.jpg', image)
    groups = {}

    for x, y in zeropix:
        groups.setdefault(x, []).append(y)

    # In[11]:

    isw_size = 20
    heights_extracted = [list(group) for group in mit.consecutive_groups(groups.keys())]
    # print(heights_extracted)
    height_ranges = [[min(i), max(i), isw_size - (max(i) - min(i))] for i in heights_extracted if
                     max(i) - min(i) in range(19, 40)]
    #print(height_ranges)
    hrs_modified = []
    for i in height_ranges:
        if i[2] % 2 == 0:
            half1, half2 = [int(i[2] / 2)] * 2
        else:
            half1, half2 = int(math.floor(i[2] / 2)), int(math.ceil(i[2] / 2))
        hrs_modified.append([i[0] - half1, i[1] + half2])


    for i in hrs_modified:
        img = cv2.imread('other_fields.jpg')
        min_p,max_p=crop_image(img)
        img=img[:,:max_p]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print(i)
        if i[0]>5:

            img = img[i[0]-10:i[1]+10, :]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            img = Image.fromarray(img)

            #plt.imshow(img)
            #plt.show()
            enhancer = ImageEnhance.Contrast(img)
            img=enhancer.enhance(0.5)
            # enhancer = ImageEnhance.Brightness(img)
            # img=enhancer.enhance(4)
            var1 = image_to_string(img, config='--psm 7')
            # print(var1)
            allowed = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                       'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')']
            for chra in var1:
                if (chra not in allowed):
                    var1 = var1.replace(chra, "")
            ims.append(var1)
        else:
            pass
    if(ims[-1][9]=='2'):
        upda=list(ims[-1])
        upda[9] = 'Z'
        ims[-1]=''.join(upda)
    for i,ele in enumerate(ims):
        ims[i]=ims[i].rstrip('-')
        ims[i] = ims[i].rstrip(' ')
    return ims




