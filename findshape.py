#!/usr/bin/env python
# -*- coding: utf-8 -*-


' image2data3'

__author__ = 'Jun Zhang'

import numpy as np
import math
import cv2
import queue
import os


def findshape(image,x,y,b_color,bound,b_threshold):
    y_range, x_range, deepth = image.shape
    x_min=max(x-bound,0)
    y_min=max(y-bound,0)
    x_max=min(x+bound,x_range)
    y_max=min(y+bound,y_range)

    shape_set = np.zeros((x_max-x_min+1, y_max-y_min+1), dtype="uint8")

    shape_min_x=x_max
    shape_min_y = y_max
    shape_max_x = x_min
    shape_max_y = y_min

    y_length = 2 * bound + 2

    for x_index in range(x_min,x_max):
      start_flag = 0
      for y_index in range(y_min,y_max):
          t_color = image[y_index, x_index]
          if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold):
              shape_set[x_index-x_min,y_index-y_min]=1
              if (start_flag==0):
                  start_flag=1
                  start_y=y_index
              if(x_index>shape_max_x):
                  shape_max_x=x_index
              if (x_index < shape_min_x):
                  shape_min_x = x_index
              if (y_index > shape_max_y):
                  shape_max_y = y_index
              if (y_index < shape_min_y):
                  shape_min_y = y_index
          else:
              if(start_flag==1):
                  end_y=y_index
                  if(end_y-start_y+1<y_length):
                    y_length=end_y-start_y+1

    if(y_length==2*bound+2):
        y_length=0
    #print("2 bound+2",2*bound+2)
    #print (y_length,shape_min_x,shape_min_y,shape_max_x,shape_max_y,shape_max_y-shape_min_y+1,shape_max_x-shape_min_x+1)
    return y_length*y_length/(shape_max_y-shape_min_y+1)/(shape_max_x-shape_min_x+1)



def findshape_v2(image,x,y,b_color,bound,b_threshold):

    y_range, x_range, deepth = image.shape
    x_min=max(x-bound,0)
    y_min=max(y-bound,0)
    x_max=min(x+bound,x_range)
    y_max=min(y+bound,y_range)

    shape_set = np.zeros((x_max-x_min+1, y_max-y_min+1), dtype="uint8")

    shape_min_x=x_max
    shape_min_y = y_max
    shape_max_x = x_min
    shape_max_y = y_min

    y_length_1 = 0
    x_length_2 = 0

    y_length_2 = 0
    x_length_1 = 0

    start_x=0
    end_x=0
    start_y=0
    end_y=0

    y_index=y
    start_flag = 0
    for x_index in range(x,x_min-1,-1):

      t_color = image[y_index, x_index]
      #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),x_index,y_index,x_min,x_max,y_min,y_max)
      if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold):
          #print(start_y, start_x, y_index, x_index, y, x, start_flag, "<")
          if (start_flag == 0):
              start_flag = 1
              start_x = x_index
      else:
          #print(start_y,start_x,y_index, x_index, y, x, start_flag, ">")
          if(start_flag==1):
              end_x=x_index
              x_length_1=abs(end_x-start_x+1)
              start_flag = 2
              #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))))
              break;

    if (start_flag == 1):
      x_length_1=abs(x-x_min+1)

    if (start_flag == 0):
        print(t_color)
        print(b_color)
        print("error x1")

    y_index = y
    start_flag = 0
    for x_index in range(x, x_max):

        t_color = image[y_index, x_index]
        if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold):
            #print(start_y, start_x, y_index, x_index, y, x, start_flag, "<")
            if (start_flag == 0):
                start_flag = 1
                start_x = x_index
        else:
            #print(start_y,start_x,y_index, x_index, y, x, start_flag, ">")
            if (start_flag == 1):
                end_x = x_index
                x_length_2 = end_x - start_x + 1
                start_flag =2
                #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))))
                break;

    if (start_flag == 1):
       x_length_2 = x_max - x+1

    if (start_flag == 0):
        print(t_color)
        print(b_color)
        print("error x2")

    x_index=x
    start_flag = 0
    for y_index in range(y,y_min-1,-1):

      t_color = image[y_index, x_index]
      if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold):
          #print(start_y, start_x, y_index, x_index, y, x, start_flag, "<")
          if (start_flag == 0):
              start_flag = 1
              start_y = y_index
      else:
          #print(start_y, start_x, y_index, x_index, y, x, start_flag, ">")
          if(start_flag==1):
              end_y=y_index
              y_length_1=abs(end_y-start_y+1)
              start_flag = 2
              #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))))
              #print("y")
              break;

    if(start_flag==1):
      y_length_1=abs(y-y_min+1)

    if (start_flag == 0):
      print(t_color)
      print(b_color)
      print("error y1")

    x_index = x
    start_flag = 0
    for y_index in range(y, y_max):

        t_color = image[y_index, x_index]
        if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold):
            #print(start_y, start_x, y_index, x_index, y, x, start_flag, "<")
            if (start_flag == 0):
                start_flag = 1
                start_y = y_index
        else:
            #print(start_y,start_x,y_index, x_index, y, x, start_flag, ">")
            if (start_flag == 1):
                end_y = y_index
                y_length_2 = end_y - start_y + 1
                start_flag = 2
                #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))))
                break;

    if (start_flag == 1):
      y_length_2 = y_max - y+1


    if (start_flag == 0):
        print(t_color)
        print(b_color)
        print("error y2")

    #print(x_length_1,x_length_2,y_length_1,y_length_2)

    #print(x_length_1, x_length_2, y_length_1, y_length_2)

    if(x_length_1>=1 and y_length_1>=1 and x_length_2>=1 and y_length_2>=1):
       ratio_1 = min(x_length_1 * x_length_2 / y_length_1 / y_length_2,y_length_1 * y_length_2 / x_length_1 / x_length_2)
       ratio_2 = x_length_1 * x_length_2 / ((x_length_1 + x_length_2) / 2) / ((x_length_1 + x_length_2) / 2)
       ratio_3 = y_length_1 * y_length_2 / ((y_length_1 + y_length_2) / 2) / ((y_length_1 + y_length_2) / 2)
       #print(ratio_1,ratio_2,ratio_3)
       return (ratio_1*ratio_2*ratio_3)
    else:
       #print("kk")
       #print(max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))))
       #print(x_length_1,x_length_2,y_length_1,y_length_2)
       #print(x,y)
       return 0


def findshape_v4(x_min,x_max,y_min,y_max):

    x_length=abs(x_max-x_min)
    y_length=abs(y_max-y_min)

    if(x_length==0 or y_length==0):
        return 0;
    else:
        ratio_1=x_length/y_length;
        ratio_2=y_length/x_length;
        return min(ratio_1,ratio_2);
