#!/usr/bin/env python
# -*- coding: utf-8 -*-


' detect_bk'

__author__ = 'Jun Zhang'

import numpy as np
import math
import cv2
import queue
import os



def detect_bk(image,step,x_block,y_block):

    bound_range=15;
    b_threshold=10;
    #b_threshold = 10;
    #nb_threshold = 5;
    nb_threshold=15;
    nb_conv=40;
    nb_ratio_threshold=0.6
    y_range, x_range, deepth = image.shape

    #disk_image = np.zeros((y_range, x_range, 3), dtype="uint8")  # back point image
    disk_mask = np.zeros((y_block, x_block), dtype="uint8")  # back point mask

    bound_mask = np.zeros((y_block, x_block), dtype="uint8")  # back point mask

    b_color=image[0,0]
    #b_color[0] = np.mean(image[0:step-1, 0:step-1,0])
    #b_color[1] = np.mean(image[0:step - 1, 0:step - 1, 1])
    #b_color[2] = np.mean(image[0:step - 1, 0:step - 1, 2])


    last_y1 = 1
    last_y2 = y_block - 1

    for x_count in range(0, x_block):
        x=x_count*step
        last_t_color = image[0, x]

        #last_t_color[0] = np.mean(image[y:y+step - 1, x:+step - 1, 0])
        #last_t_color[1] = np.mean(image[y:y + step - 1, x:+step - 1, 1])
        #last_t_color[2] = np.mean(image[y:y + step - 1, x:+step - 1, 2])

        disk_mask[0:last_y1 - 1, x_count] = 1

        for y_count in range(last_y1, y_block):
            y=y_count*step
            t_color = image[y, x]
            # this_r=this_color[0]
            # this_g=this_color[1]
            # this_b=this_color[2]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_y_min=y_count;
                bound_y_max=max(0,y_count+bound_range)
                bound_mask[bound_y_min:bound_y_max, x_count] = 1

                if (y_count > y_block / 2):
                    last_y1 = 1
                else:
                    last_y1 = max(1, y_count - 10)
                # print(last_y1)
                break

        last_t_color = image[y_range-1, x]
        disk_mask[last_y2:y_block - 1, x_count] = 1

        for y_count in range(last_y2, 0, -1):
            y=y_count*step
            t_color = image[y, x]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_y_max=y_count;
                bound_y_min=max(0,y_count-bound_range)
                bound_mask[bound_y_min:bound_y_max, x_count] = 1
                if (y_count < y_block / 2):
                    last_y2 = y_block - 2
                else:
                    last_y2 = min(y_block - 2, y_count + 10)
                break

    last_x1 = 1
    last_x2 = x_block - 1
    for y_count in range(0, y_block):
        y=y_count*step
        last_t_color = image[y, 0]
        disk_mask[y_count, 0:last_x1 - 1] = 1
        for x_count in range(last_x1, x_block):
            x=x_count*step
            # print(y)
            # this_color=list(image[y,x])
            t_color = image[y, x]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_x_min=x_count;
                bound_x_max=max(0,x_count+bound_range)
                bound_mask[y_count, bound_x_min:bound_x_max] = 1

                if (x_count > x_block / 2):
                    last_x1 = 1
                else:
                    last_x1 = max(1, x_count - 10)
                # print(last_y1)
                break

        last_t_color = image[y, x_range - 1]
        disk_mask[y_count, last_x2:x_block - 1] = 1

        for x_count in range(last_x2, 0, -1):
            x=x_count*step
            t_color = image[y, x]
            # print(y)
            # this_r=this_color[0]
            # this_g=this_color[1]
            # this_b=this_color[2]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_x_max=x_count;
                bound_x_min=max(0,x_count-bound_range)
                bound_mask[y_count, bound_x_min:bound_x_max] = 1
                if (x_count < x_block / 2):
                    last_x2 = x_block - 2
                else:
                    last_x2 = min(x_block - 2, x_count + 10)
                break

#fix background by average

    for x_count in range(0, x_block):
        for y_count in range(0, y_block):
            x_min=max(0,x_count-nb_conv)
            x_max=min(x_block-1,x_count+nb_conv)
            y_min = max(0, y_count - nb_conv)
            y_max = min(y_block-1, y_count + nb_conv)
            ratio=np.mean(disk_mask[y_min:y_max,x_min:x_max])
            if(ratio>nb_ratio_threshold and disk_mask[y_count,x_count]==0):
                disk_mask[y_count,x_count]=1

#fix bound by just inside node
    for x_count in range(0, x_block):
        for y_count in range(0, y_block):
            if(disk_mask[y_count,x_count]==1):
                bound_mask[y_count,x_count]=0

    return (disk_mask,bound_mask)


def detect_bk_para(image,step,x_block,y_block,b_threshold,nb_threshold,b_point):

    bound_range=15;
    #b_threshold=10;
    #b_threshold = 10;
    #nb_threshold = 5;
    #nb_threshold=15;
    nb_conv=40;
    nb_ratio_threshold=0.6
    y_range, x_range, deepth = image.shape

    #disk_image = np.zeros((y_range, x_range, 3), dtype="uint8")  # back point image
    disk_mask = np.zeros((y_block, x_block), dtype="uint8")  # back point mask

    bound_mask = np.zeros((y_block, x_block), dtype="uint8")  # back point mask

    b_color=b_point

    #print("hi")
    #print(b_color)
    #print(image[0,0])

    #b_color=image[0,0]
    #b_color[0] = np.mean(image[0:step-1, 0:step-1,0])
    #b_color[1] = np.mean(image[0:step - 1, 0:step - 1, 1])
    #b_color[2] = np.mean(image[0:step - 1, 0:step - 1, 2])


    last_y1 = 1
    last_y2 = y_block - 1

    for x_count in range(0, x_block):
        x=x_count*step
        last_t_color = image[0, x]

        #last_t_color[0] = np.mean(image[y:y+step - 1, x:+step - 1, 0])
        #last_t_color[1] = np.mean(image[y:y + step - 1, x:+step - 1, 1])
        #last_t_color[2] = np.mean(image[y:y + step - 1, x:+step - 1, 2])

        disk_mask[0:last_y1 - 1, x_count] = 1

        for y_count in range(last_y1, y_block):
            y=y_count*step
            t_color = image[y, x]
            # this_r=this_color[0]
            # this_g=this_color[1]
            # this_b=this_color[2]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_y_min=y_count;
                bound_y_max=max(0,y_count+bound_range)
                bound_mask[bound_y_min:bound_y_max, x_count] = 1

                if (y_count > y_block / 2):
                    last_y1 = 1
                else:
                    last_y1 = max(1, y_count - 10)
                # print(last_y1)
                break

        last_t_color = image[y_range-1, x]
        disk_mask[last_y2:y_block - 1, x_count] = 1

        for y_count in range(last_y2, 0, -1):
            y=y_count*step
            t_color = image[y, x]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_y_max=y_count;
                bound_y_min=max(0,y_count-bound_range)
                bound_mask[bound_y_min:bound_y_max, x_count] = 1
                if (y_count < y_block / 2):
                    last_y2 = y_block - 2
                else:
                    last_y2 = min(y_block - 2, y_count + 10)
                break

    last_x1 = 1
    last_x2 = x_block - 1
    for y_count in range(0, y_block):
        y=y_count*step
        last_t_color = image[y, 0]
        disk_mask[y_count, 0:last_x1 - 1] = 1
        for x_count in range(last_x1, x_block):
            x=x_count*step
            # print(y)
            # this_color=list(image[y,x])
            t_color = image[y, x]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_x_min=x_count;
                bound_x_max=max(0,x_count+bound_range)
                bound_mask[y_count, bound_x_min:bound_x_max] = 1

                if (x_count > x_block / 2):
                    last_x1 = 1
                else:
                    last_x1 = max(1, x_count - 10)
                # print(last_y1)
                break

        last_t_color = image[y, x_range - 1]
        disk_mask[y_count, last_x2:x_block - 1] = 1

        for x_count in range(last_x2, 0, -1):
            x=x_count*step
            t_color = image[y, x]
            # print(y)
            # this_r=this_color[0]
            # this_g=this_color[1]
            # this_b=this_color[2]

            if (max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))) <= b_threshold or max(
                    abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= nb_threshold):
                # if (max(abs(t_color.astype(np.int32) - last_t_color.astype(np.int32))) <= 10):
                # if(max(abs(this_r-basic_r),abs(this_g-basic_g),abs(this_b-this_b))<=5):
                disk_mask[y_count, x_count] = 1
                last_t_color = t_color
                continue
            else:
                #print(x,y,max(abs(t_color.astype(np.int32) - b_color.astype(np.int32))),t_color,last_t_color,b_color)
                bound_x_max=x_count;
                bound_x_min=max(0,x_count-bound_range)
                bound_mask[y_count, bound_x_min:bound_x_max] = 1
                if (x_count < x_block / 2):
                    last_x2 = x_block - 2
                else:
                    last_x2 = min(x_block - 2, x_count + 10)
                break

#fix background by average

    for x_count in range(0, x_block):
        for y_count in range(0, y_block):
            x_min=max(0,x_count-nb_conv)
            x_max=min(x_block-1,x_count+nb_conv)
            y_min = max(0, y_count - nb_conv)
            y_max = min(y_block-1, y_count + nb_conv)
            ratio=np.mean(disk_mask[y_min:y_max,x_min:x_max])
            if(ratio>nb_ratio_threshold and disk_mask[y_count,x_count]==0):
                disk_mask[y_count,x_count]=1

#fix bound by just inside node
    for x_count in range(0, x_block):
        for y_count in range(0, y_block):
            if(disk_mask[y_count,x_count]==1):
                bound_mask[y_count,x_count]=0

    return (disk_mask,bound_mask)