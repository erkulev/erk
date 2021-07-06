# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:11:42 2020

@author: admin
"""
import time
import numpy as np
import cv2
net=cv2.dnn.readNet('yolo/yolov3.weights','yolo/yolov3.cfg')
with open('yolo/coco.names.txt') as f:
    labels=f.read().strip().split('\n')
img=cv2.imread('output 1.jpg')
height, width,_= img.shape
blob=cv2.dnn.blobFromImage(img,1/255,(640,480),(0,0,0),swapRB=True)
net.setInput(blob)
layers_names=net.getLayerNames()
out_layers_indexes_arr=net.getUnconnectedOutLayers()
out_layers_indexes=[index[0] -1 for index in out_layers_indexes_arr]
out_layers_names = [layers_names[index] for index in out_layers_indexes]
out_layers=net.forward(out_layers_names)
box1=[[30,730,1290,1440],
      [1290,730,2000,1440],
      [2000,820,2560,1440]
      ]
def draw_boxes(img,box1):
       color =[255,0,0]
       width =7
       for x3,y3 ,x4,y4 in box1:
           img=cv2.rectangle(img,(x3,y3),(x4,y4),color,width)
       return img    
def check_cords(x,y,label):
    if label not in ['car','trak','bus','motorbike'] :
        return False
    for x3,y3 ,x4,y4 in box1:
           if x3 <= x <= x4 and y3 <= y <= y4:
                return True
    return False
higtory=[]
def draw_car_count (img,success_count):
    higtory.append([time.time(),success_count])
    timestamp=time.time()-30
    filtered_higtory=[]
    for t,c in higtory:
        if t >= timestamp:
            filtered_higtory.append(c)
    mid_count=int(sum(filtered_higtory) / len(filtered_higtory)) 
    filtered_higtory_2=[]
    for c in filtered_higtory:
        if t >=mid_count:
            filtered_higtory_2.append(c)
    max_count= max(filtered_higtory)   
            
    color =[0,255,0]
    width =7
    font_size=2
    font=cv2.FONT_HERSHEY_SIMPLEX
    text=str(success_count)+ ' car'
    img=cv2.putText(img,text,(50,150),font,font_size,color,width)
    return img
def draw_object(img, x,y,w,h,label,success):
    x1,y1 =x-w//2,y-h//2
    x2,y2 =x+w//2,y+h//2
    color =[0,255,0] if success else [0,0,255]
    width =10
    img=cv2.rectangle(img,(x1,y1),(x2,y2),color,width)   
    font_size=4
    font=cv2.FONT_HERSHEY_SIMPLEX
    text=label
    img=cv2.putText(img,text,(x1,y1-10),font,font_size,color,width)
    return img

object_boxes=[]
object_probas=[]
object_labels=[]

for layer in out_layers:
    for result in layer: 
       # h,w,y,x=result[:4]
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
            
            #print(x,y,w,h,labels[ max_probas_index], probas)


img=draw_boxes(img,box1)
success_count=0

filter_boxes_indexes=cv2.dnn.NMSBoxes(object_boxes,object_probas,0.0,0.3)
for index_arr in filter_boxes_indexes:
     index=index_arr[0]
     box=object_boxes[index]
     x,y,w,h=box
     #print(object_labels[1])
     success=check_cords(x,y,object_labels[index])
     img= draw_object(img,x,y,w,h,object_labels[index],success)
     if success:
         success_count+=1
img=draw_car_count (img,success_count)         
cv2.imwrite('image1_1.jpg',img)     
    