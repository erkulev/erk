# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:42:11 2020

@author: admin
"""


import matplotlib.pyplot as plt
import cv2
jmg=cv2.imread('image1.jpg')

#for i in range(1000):
#    for j in range(1000):
 #       jmg [i][j]=[0,255,0]
x1,y1 =1550,1050
x2,y2 =1950,1370
color =[0,255,0]
width =10
jmg=cv2.rectangle(jmg,(x1,y1),(x2,y2),color,width)
x,y =1550,1030
font_size=4
font=cv2.FONT_HERSHEY_SIMPLEX
width =10
color =[0,255,0]
text='car'
jmg=cv2.putText(jmg,text,(x,y),font,font_size,color,width)
jmg=cv2.flip(jmg,0 )
cv2.imwrite('image1_1.jpg',jmg)
rgb_jmg = cv2.cvtColor(jmg, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_jmg)
plt.show()

size=(2560,1440)
codec=cv2.VideoWriter_fourcc(*'MJPG')
out=cv2.VideoWriter('output.avi',codec,25,size)
cap=cv2.VideoCapture('video.mp4')
while  cap.isOpened():
    print('.',end='')
    ret,frame =cap.read()
    if not ret:
        break
    frame= cv2.resize(frame,size)
    frame1=cv2.flip( frame, 1 )
    out.write(frame1)
cap.release()
out.release()
    
    

