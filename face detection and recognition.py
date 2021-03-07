#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import numpy as np


# In[2]:


# Loading all the images
img_hrithik1 = face_recognition.load_image_file('imagedetec/Hrithik-Roshan-1.jpg') #train image #returns an array representation of images
img_hrithik1 = cv2.cvtColor(img_hrithik1, cv2.COLOR_BGR2RGB) # BGR TO RGB
img_hrithik2 = face_recognition.load_image_file('imagedetec/Hrithik-Roshan-2.jpg') # test image
img_hrithik2 = cv2.cvtColor(img_hrithik2, cv2.COLOR_BGR2RGB)
img_ratantata = face_recognition.load_image_file('imagedetec/Ratan_tata.jpg') # Other image
img_ratantata = cv2.cvtColor(img_ratantata, cv2.COLOR_BGR2RGB)


# In[3]:


img_hrithik1.shape, img_hrithik2.shape, img_ratantata.shape


# In[4]:


# locating face location in train image
face_loc = face_recognition.face_locations(img_hrithik1)[0]
hrithik_encod = face_recognition.face_encodings(img_hrithik1)[0]


# In[5]:


face_loc, hrithik_encod


# In[6]:


# Since we have now the face location coordinates of train image lets draw a rectangle around them using opencv
cv2.rectangle(img_hrithik1,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,255,0),3)


# In[7]:


# locating face location, encoding the test image then putting the rectangle arounf the face location in test image
test_loc = face_recognition.face_locations(img_hrithik2)[0]
test_encod = face_recognition.face_encodings(img_hrithik2)[0]
cv2.rectangle(img_hrithik2,(test_loc[3],test_loc[0]),(test_loc[1],test_loc[2]),(0,255,0),3)


# # Step 3 - Comparing the faces using their encodings

# In[8]:


results = face_recognition.compare_faces([hrithik_encod], test_encod)

# Checking the face distance to see the similarities
facedis = face_recognition.face_distance([hrithik_encod], test_encod)


# In[9]:


results, facedis


# In[10]:


# Putting the results on the required images using opencv
cv2.putText(img_hrithik2, str(results)+'  '+str(round(facedis[0],2)), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)


# In[11]:


# viewing the images using opencv
cv2.imshow('1',img_hrithik1)
cv2.imshow('2',img_hrithik2)
cv2.imshow('3',img_ratantata)

cv2.waitKey(0) #always remember to include it when using cv2.imshow to view the iamges


# In[ ]:




