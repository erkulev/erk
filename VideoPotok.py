# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:55:53 2020

@author: admin
"""



import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import Image 
from IPython import display
size=(2560,1440)
codec=cv2.VideoWriter_fourcc(*'MJPG')
out=cv2.VideoWriter('output.avi',codec,5,size)
cap=cv2.VideoCapture('http://109.236.111.203:90/mjpg/video.mjpg')

net=cv2.dnn.readNet('yolo/yolov3.weights','yolo/yolov3.cfg')
with open('yolo/coco.names.txt') as f:
    labels=f.read().strip().split('\n')
layers_names=net.getLayerNames()
out_layers_indexes_arr=net.getUnconnectedOutLayers()
out_layers_indexes=[index[0] -1 for index in out_layers_indexes_arr]
out_layers_names = [layers_names[index] for index in out_layers_indexes]
#count=0
count1=0
def appley_yolo( img):
    height, width,_= img.shape
    blob=cv2.dnn.blobFromImage(img,1/255,(608,608),(0,0,0),swapRB=True)
    net.setInput(blob)
    out_layers=net.forward(out_layers_names)
    object_boxes=[]
    object_probas=[]
    object_labels=[]

    for layer in out_layers:
       for result in layer:           
           x,y,w,h=result[:4]
           x=int(x * width)
           w=int(w * width)
           y=int(y*height)
           h=int(h*height)
           probas=result[5:]
           max_probas_index= np.argmax(probas)
           max_probas=probas[max_probas_index]
           if  max_probas >0:
               object_boxes.append([x,y,w,h] )
               object_probas.append( float(max_probas))
               object_labels.append(labels[max_probas_index])
           count=object_labels.count('car')
           
           filter_boxes_indexes=cv2.dnn.NMSBoxes(object_boxes,object_probas,0.0,0.4)
           for index_arr in filter_boxes_indexes:
                index=index_arr[0]
                box=object_boxes[index]
                x,y,w,h=box
               
                img= draw_object(img,x,y,w,h,object_labels[index],count) 
    return img
                
def draw_object(img, x,y,w,h,label,count):
   
    x1,y1 =x-w//2,y-h//2
    x2,y2 =x+w//2,y+h//2
    color =[0,255,0]
    width =5
    img=cv2.rectangle(img,(x1,y1),(x2,y2),color,width)   
    font_size=2
    font=cv2.FONT_HERSHEY_SIMPLEX
    text=label+str(count)
    img=cv2.putText(img,text,(x1,y1-10),font,font_size,color,width)
    #img=cv2.putText(img,str(count),(1000,1000),font,2,color,width)
   
    return img
                
               
for i in range(10):
#while  cap.isOpened():
 
    
    print('.',end='')
    ret,frame =cap.read()
    if not ret:
        break
    
    frame= cv2.resize(frame,size)
    frame=appley_yolo(frame)
    out.write(frame)
"""
    plt.figure(figsize=(20,15))
    plt.imshow( frame)
    display.clear_output(wait=True)
    display.display(plt.gcf())
"""
   
    
    
cap.release()
out.release()

    