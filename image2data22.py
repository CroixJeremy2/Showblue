#!/usr/bin/env python
# -*- coding: utf-8 -*-


' image2data22'

__author__ = 'Jun Zhang'

import numpy as np
import math
import cv2
import queue
import os
import detect_bk as db
import findshape
import loadcfg

def image2data2(image_file,out_point=1,out_bound=1,out_hull=1,b_threshold=10,nb_threshold=5):
    #out_point=1 => compute blue_point
    #out_bound=1=> compute bound_image, disk_image
    #out_hull=1> compute disk_imageout_bound
    # block_per_dim=40 #for fig 3
    # block_per_dim=200 #for fig 2
    # block_per_dim=300 #for fig 2
    # global block_per_dim, conv_dist, boud_dist, back_ratio, boudn_ratio, limit, blue_range, red_range

    min_block_size=25;

    block_per_dim = 800  # for fig 1

    conv_dist = 7;
    bound_dist = 10;
    back_ratio = 0.95
    bound_ratio = 0.92

    limit = 25;  # threshold for detecting background
    blue_range = 100  # upper bound for blue points in blue domain
    red_range = 90  # low bound for blue points in red domain
    blue_red_dst=15; # minimal gap between blue and red for blue points

    image_name=os.path.splitext(image_file)[0]
    out_file="Out"+os.path.splitext(image_file)[0]+".txt"

    if not os.path.exists('process'):
        os.makedirs('process')

    blue_file="process/blue_"+image_file
    back_file="process/bound_"+image_file
    snow_file = "process/hull_" + image_file

    #load image
    canvas = cv2.imread(image_file)
    #canvas = cv2.imread("test1.jpg")
    #canvas = cv2.imread("test2.jpg")
    #canvas = cv2.imread("test3.jpg")
    y_range,x_range,deepth=canvas.shape

    #print(x_range,y_range)
    front = np.zeros((y_range, x_range, 3), dtype="uint8") #background image
    blue_point = np.zeros((y_range, x_range, 3), dtype="uint8") #blue point image

    disk_image = np.zeros((y_range, x_range, 3), dtype="uint8") #back point image
    bound_image = np.zeros((y_range, x_range, 3), dtype="uint8")  # boundary point image

    #cv2.namedWindow("Canvas",0); #show background
    #cv2.namedWindow("Canvas_show",0); #show blue points
    #cv2.namedWindow("Original",0); #orignal image
    #cv2.namedWindow("blur",0);
    #cv2.namedWindow("filter",0);
    #cv2.namedWindow("disk",0);

    #this is the default names of colors
    blacks = (0, 0, 0)
    whites = (255, 255, 255)
    greens = (0, 255, 0)
    blues = (100, 100, 255)
    #################################33

    # filter the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    sharped = cv2.filter2D(canvas, -1, kernel=kernel)
    ###############333

    # blur the image
    kernel_size = (5, 5);
    sigma = 1.5;

    dst = cv2.GaussianBlur(sharped,kernel_size,sigma)
    ########################3333





    #statistics
    count=0;
    count_point=0;
    count_blue=0;
    ####################3

    #paramet setting
    #step=10;
    #

    step=math.ceil(max(x_range,y_range)/block_per_dim)
    #######################3

    #blue_set = np.zeros((block_per_dim, x_range, 3), dtype="uint8") #3


    #block size in x axis
    x_block=len(range(0, x_range,step))
    #block size in y axis
    y_block=len(range(0, y_range,step))

    #whether it is a blue block matrix
    blue_set = np.zeros((x_block,y_block), dtype="uint8")
    #whether it is a background block matrix
    back_set = np.zeros((y_block,x_block), dtype="uint8")
    bound_set = np.zeros((y_block,x_block), dtype="uint8")



    # sum_diff_ab=0;
    # sum_diff_ac=0;
    # for back_x in range(0,x_range,x_range-1):
    #    for back_y in range(0,y_range,y_range-1):
    #         f = list(dst[back_y,back_x])
    #         a = f[0]
    #         b = f[1]
    #         c = f[2]
    #         sum_diff_ab=abs(int(a)-int(b))+sum_diff_ab
    #         sum_diff_ac = abs(int(a)-int(c)) + sum_diff_ac
    #
    # limit=max(sum_diff_ab/4,sum_diff_ac/4)*5

    #b_threshold=10
    #nb_threshold = 5
    back_set,bound_set = db.detect_bk(canvas, step, x_block, y_block)
    #back_set, bound_set = db.detect_bk_para(canvas, step, x_block, y_block,b_threshold, nb_threshold)


    space_ratio=round(1-np.sum(back_set)/(x_block*y_block),4)



    #detecting blue points and background points
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            count=count+1;
            f=list(dst[y,x])
            a=f[0]
            b=f[1]
            c=f[2]
            red = int(a)
            green = int(b)
            blue = int(c)
            ##print(r,g,b)
            #print(r,g,b)
            #print(abs(g-255),abs(b-255))

            # if max(abs(red-green),abs(red-blue))<=limit:
            #     #if abs(g-b)<=limit and abs(b-r)<=limit and abs(r-b)<=limit:
            #     #print(r, g, b)
            #     cv2.rectangle(front, (x, y), (x + step, y + step), whites, -1)
            #     back_set[x_count, y_count] = 1
            #     count_point=count_point+1;

            if blue<blue_range and red>blue+blue_red_dst:
                #   print(r,g,b,x,y)
                #print(x,y,red,green,blue)
                if(bound_set[y_count,x_count]==0 and back_set[y_count,x_count]==0):
                    if(out_point==1):
                        cv2.rectangle(blue_point, (x, y), (x + step, y + step), blues, -1);
                    blue_set[x_count,y_count]=1
                #count_blue=count_blue+1;

            y_count=y_count+1
        x_count=x_count+1

    #recheck background blocks



    #print(back_count,blue_count,bound_count,other_count)
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            if(back_set[y_count,x_count]==1 and (out_hull==1 or out_bound==1)):
                cv2.rectangle(disk_image, (x, y), (x + step, y + step), whites, -1)
            if (bound_set[y_count, x_count] == 1 and out_bound==1):
                cv2.rectangle(bound_image, (x, y), (x + step, y + step), blues, -1)
            y_count=y_count+1
        x_count=x_count+1


    #set the list of blue points and  construct neighboring table
    id_x,id_y = np.where(blue_set==1)
    blue_set_len=len(id_x)

    blue_set_nb = np.zeros((blue_set_len,blue_set_len), dtype="uint8")

    blue_label = np.zeros((blue_set_len,1), dtype="uint32")

    for index1 in range (0,blue_set_len):
        index1_x = id_x[index1]
        index1_y = id_y[index1]
        for index2 in range (0,blue_set_len):
            index2_x = id_x[index2]
            index2_y = id_y[index2]
            if (index1==index2):
                continue
            if ((abs(index1_x-index2_x)<=1) and  (abs(index1_y-index2_y)<=1)):
                blue_set_nb[index1,index2]=1;
                blue_set_nb[index2,index1]=1;



    #assign number to blue points
    list_x=id_x
    list_y=id_y
    count_index=1;

    size_list=[];

    for index in range (0,blue_set_len):
        index_x=id_x[index]
        index_y=id_y[index]
        if(blue_label[index]>0):
            continue
        blue_label[index]=count_index
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        # 将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
        font_size = int(max(x_range, y_range) / 1600)
        #cv2.putText(blue_point,str(count_index), (step*index_x,step*index_y),font, font_size, (255, 255, 255), 5)
        count_index = count_index + 1
        q = queue.Queue()
        q.put(index)
        this_size=1;
        while not q.empty():
            head = q.get()
            for nb in range(0, blue_set_len):
                if(blue_set_nb[head, nb]==0):
                    continue
                if(blue_label[nb]>0):
                    continue
                this_size=this_size+1;
                blue_label[nb] = count_index-1
                nb_x = id_x[nb]
                nb_y = id_y[nb]
                cv2.line(blue_point, (step * nb_x, step * nb_y), (step * index_x, step * index_y), blues, 5)
                q.put(nb)
    size_list.append(this_size);


    count_blue=count_index-1;


    #for x in range(1, x_range,step):
    #     for y in range (1,y_range,step):
    #        cv2.rectangle(front, (x, y), (x+step, y+step), green, -1)

    #show background
    #two_fig = cv2.add(dst, front)
    two_fig = cv2.addWeighted(dst, 0.7, disk_image, 0.3, 0)

    #cv2.imshow("Canvas", two_fig) #13

    #cv2.resizeWindow("Canvas", 400, 400);


    #show blue points
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    font_size=int(max(x_range,y_range)/300)
    cv2.putText(blue_point,str(count_index-1), (0,int(y_range/6)),font, font_size, (255, 255, 0), 20)

    cv2.putText(blue_point,str(int(space_ratio*x_range*y_range)), (int(x_range/3),int(y_range/6)),font, font_size, (255, 255, 0), 20)


    #mix_blue = cv2.add(dst, blue_point)
    mix_blue=cv2.addWeighted(dst,0.7,blue_point,0.3,0)

    mix_black = cv2.addWeighted(two_fig, 0.7, bound_image, 0.3, 0)

    if(out_point==1):
        cv2.imwrite(blue_file, mix_blue)
    if(out_bound==1):
        cv2.imwrite(back_file, mix_black)
    if(out_hull==1):
        cv2.imwrite(snow_file, two_fig)


    #cv2.imshow("Canvas_show", blue_point) #13
    #cv2.imshow("Canvas_show", mix_blue) #13

    #cv2.resizeWindow("Canvas_show", 400, 400);

    #cv2.imshow("Original", canvas) #13

    #cv2.resizeWindow("Original", 400, 400);

    #cv2.imshow("blur", blurred) #13

    #cv2.resizeWindow("blur", 400, 400);

    #cv2.imshow("filter", dst) #13

    #cv2.resizeWindow("filter", 400, 400);

    #cv2.imshow("disk", disk_image) #13

    #cv2.resizeWindow("disk", 400, 400);

    #print(count_point/count)
    #print(1-back_count/sum_count)
    #print(count_blue)
    #cv2.waitKey(0) #14

    blue_point_num=count_index-1
    cell_size=int(x_range*y_range*space_ratio)

    fo = open("data.txt", "a")

    fo.writelines([image_name, ":", str(blue_point_num), ":", str(cell_size), ":", str(x_range), ":", str(y_range), ":",
                   str(space_ratio), '\n']);

    fo.close()
    return (blue_point_num, cell_size, x_range, y_range,space_ratio )


def image2data22_para(image_file,bk_cfg_list, point_cfg_list, inside_cfg_list,out_point=1,out_bound=1,out_hull=1,alpha=1,shape_threshold=0):
    #out_point=1 => compute blue_point
    #out_bound=1=> compute bound_image, disk_image
    #out_hull=1> compute disk_image
    # block_per_dim=40 #for fig 3
    # block_per_dim=200 #for fig 2
    # block_per_dim=300 #for fig 2
    # global block_per_dim, conv_dist, boud_dist, back_ratio, boudn_ratio, limit, blue_range, red_range

    block_per_dim = 800  # for fig 1

    [avg_r_bk,avg_g_bk,avg_b_bk,std_r_bk,std_g_bk,std_b_bk]=bk_cfg_list
    [avg_r_pt, avg_g_pt, avg_b_pt, std_r_pt, std_g_pt, std_b_pt] = point_cfg_list
    [avg_r_in, avg_g_in, avg_b_in, std_r_in, std_g_in, std_b_in] = inside_cfg_list

    conv_dist = 7;
    bound_dist = 10;
    back_ratio = 0.95
    bound_ratio = 0.92

    limit = 25;  # threshold for detecting background
    blue_range = 100  # upper bound for blue points in blue domain
    red_range = 90  # low bound for blue points in red domain
    blue_red_dst=15; # minimal gap between blue and red for blue points

    image_name=os.path.splitext(image_file)[0]
    out_file="Out"+os.path.splitext(image_file)[0]+".txt"

    if not os.path.exists('process'):
        os.makedirs('process')

    blue_file="process/blue_"+image_file
    back_file="process/bound_"+image_file
    snow_file = "process/hull_" + image_file

    #load image
    canvas = cv2.imread(image_file)
    #canvas = cv2.imread("test1.jpg")
    #canvas = cv2.imread("test2.jpg")
    #canvas = cv2.imread("test3.jpg")
    y_range,x_range,deepth=canvas.shape

    #print(x_range,y_range)
    front = np.zeros((y_range, x_range, 3), dtype="uint8") #background image
    blue_point = np.zeros((y_range, x_range, 3), dtype="uint8") #blue point image

    disk_image = np.zeros((y_range, x_range, 3), dtype="uint8") #back point image
    bound_image = np.zeros((y_range, x_range, 3), dtype="uint8")  # boundary point image

    #cv2.namedWindow("Canvas",0); #show background
    #cv2.namedWindow("Canvas_show",0); #show blue points
    #cv2.namedWindow("Original",0); #orignal image
    #cv2.namedWindow("blur",0);
    #cv2.namedWindow("filter",0);
    #cv2.namedWindow("disk",0);

    #this is the default names of colors
    blacks = (0, 0, 0)
    whites = (255, 255, 255)
    greens = (0, 255, 0)
    blues = (100, 100, 255)
    #################################33



    # blur the image
    #################################33

    # filter the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    sharped = cv2.filter2D(canvas, -1, kernel=kernel)
    ###############333

    # blur the image
    kernel_size = (5, 5);
    sigma = 1.5;

    dst = cv2.GaussianBlur(sharped, kernel_size, sigma)
    ########################3333
    ###############333


    #statistics
    count=0;
    count_point=0;
    count_blue=0;
    ####################3

    #paramet setting
    #step=10;
    #

    step=math.ceil(max(x_range,y_range)/block_per_dim)
    #######################3

    #blue_set = np.zeros((block_per_dim, x_range, 3), dtype="uint8") #3


    #block size in x axis
    x_block=len(range(0, x_range,step))
    #block size in y axis
    y_block=len(range(0, y_range,step))

    #whether it is a blue block matrix
    blue_set = np.zeros((x_block,y_block), dtype="uint8")
    #whether it is a background block matrix
    back_set = np.zeros((y_block,x_block), dtype="uint8")
    bound_set = np.zeros((y_block,x_block), dtype="uint8")



    # sum_diff_ab=0;
    # sum_diff_ac=0;
    # for back_x in range(0,x_range,x_range-1):
    #    for back_y in range(0,y_range,y_range-1):
    #         f = list(dst[back_y,back_x])
    #         a = f[0]
    #         b = f[1]
    #         c = f[2]
    #         sum_diff_ab=abs(int(a)-int(b))+sum_diff_ab
    #         sum_diff_ac = abs(int(a)-int(c)) + sum_diff_ac
    #
    # limit=max(sum_diff_ab/4,sum_diff_ac/4)*5

    #b_threshold=10
    #nb_threshold = 5

    #back_set,bound_set = db.detect_bk(canvas, step, x_block, y_block)

    #back_set, bound_set = db.detect_bk_para(canvas, step, x_block, y_block,b_threshold, nb_threshold)

    bk_ratio = loadcfg.loadalpha("cfg/cfg_bk_ratio.txt")

    diff_r = max(0, abs(abs(float(avg_r_in) - float(avg_r_bk)))*bk_ratio)
    diff_g = max(0, abs(abs(float(avg_g_in) - float(avg_g_bk)))*bk_ratio)
    diff_b = max(0, abs(abs(float(avg_b_in) - float(avg_b_bk)))*bk_ratio)

    b_threshold = max(diff_r, diff_g, diff_b)
    nb_threshold = max(std_r_bk, std_b_bk, std_g_bk)



    bk_point=np.zeros((3), dtype="uint8")
    bk_point[0]=avg_r_bk;
    bk_point[1] = avg_g_bk;
    bk_point[2] = avg_b_bk;
    back_set, bound_set = db.detect_bk_para(canvas, step, x_block, y_block, b_threshold, nb_threshold,bk_point)


    space_ratio=round(1-np.sum(back_set)/(x_block*y_block),4)



    #detecting blue points and background points
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            count=count+1;
            f=list(dst[y,x])
            a=f[0]
            b=f[1]
            c=f[2]
            red = int(a)
            green = int(b)
            blue = int(c)
            ##print(r,g,b)
            #print(r,g,b)
            #print(abs(g-255),abs(b-255))

            # if max(abs(red-green),abs(red-blue))<=limit:
            #     #if abs(g-b)<=limit and abs(b-r)<=limit and abs(r-b)<=limit:
            #     #print(r, g, b)
            #     cv2.rectangle(front, (x, y), (x + step, y + step), whites, -1)
            #     back_set[x_count, y_count] = 1
            #     count_point=count_point+1;

            #if blue<blue_range and red>blue+blue_red_dst:
            if (red > avg_r_pt - alpha * std_r_pt and red < avg_r_pt + alpha * std_r_pt and green > avg_g_pt - alpha * std_g_pt and green < avg_g_pt + alpha * std_g_pt and blue > avg_b_pt - alpha * std_b_pt and blue < avg_b_pt + alpha * std_b_pt):
                #   print(r,g,b,x,y)
                #print(x,y,red,green,blue)
                 if (bound_set[y_count, x_count] == 0 and back_set[y_count, x_count] == 0):
                     if (shape_threshold == 0):
                        blue_set[x_count, y_count] = 1
                        if (out_point == 1):
                          cv2.rectangle(blue_point, (x, y), (x + step*3, y + step*3), blues, -1);
                     else:
                        #shape_ratio = findshape.findshape_v2(dst, x, y, dst[y, x], 20, 70)
                        #print(shape_ratio)
                        if (shape_ratio >= shape_threshold):
                            #print(shape_ratio)
                            blue_set[x_count, y_count] = 1
                            if (out_point == 1):
                             cv2.rectangle(blue_point, (x, y), (x + step*3, y + step*3), whites, -1);
                            # cv2.rectangle(blue_point, (x-15, y-15), (x + 15, y + 15), whites, -1);
                            #print("shape ratio", shape_ratio)
                #count_blue=count_blue+1;

            y_count=y_count+1
        x_count=x_count+1

    #recheck background blocks


    # print(back_count,blue_count,bound_count,other_count)
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            if(back_set[y_count,x_count]==1 and (out_hull==1 or out_bound==1)):
                cv2.rectangle(disk_image, (x, y), (x + step, y + step), whites, -1)
            if (bound_set[y_count, x_count] == 1 and out_bound==1):
                cv2.rectangle(bound_image, (x, y), (x + step, y + step), blues, -1)
            y_count=y_count+1
        x_count=x_count+1


    #set the list of blue points and  construct neighboring table
    id_x,id_y = np.where(blue_set==1)
    blue_set_len=len(id_x)

    print(blue_set_len);

    #blue_set_nb = np.zeros((blue_set_len,blue_set_len), dtype="uint8")

    nb_range = 1
    blue_set_4nb = np.zeros((blue_set_len, (2*nb_range+1)**2,3), dtype="uint32")
    blue_set_4nb_len = np.zeros((blue_set_len, 1), dtype="uint32")
    # left, right, up, down
    blue_index_set=np.ones((x_block,y_block), dtype="int32")*(-1)


    blue_label = np.zeros((blue_set_len,1), dtype="uint32")

    #for index1 in range (0,blue_set_len):
    #    index1_x = id_x[index1]
    #    index1_y = id_y[index1]
    #    for index2 in range (0,blue_set_len):
    #        index2_x = id_x[index2]
    #        index2_y = id_y[index2]
    #        if (index1==index2):
    #            continue
    #        if ((abs(index1_x-index2_x)<=1) and  (abs(index1_y-index2_y)<=1)):
    #            blue_set_nb[index1,index2]=1;
    #            blue_set_nb[index2,index1]=1;




    for index1 in range (0,blue_set_len):
        index1_x = id_x[index1]
        index1_y = id_y[index1]
        blue_index_set[index1_x,index1_y]=index1;


    for index1 in range (0,blue_set_len):
        index1_x = id_x[index1]
        index1_y = id_y[index1]

        nb_list_x=[]
        nb_list_y=[]
        nb_list_index=[]



        for nb_x in range(index1_x-nb_range,index1_x+nb_range+1):
            for nb_y in range(index1_y-nb_range, index1_y+nb_range+1):
                if(nb_x==index1_x and nb_y==index1_y):
                    continue
                if(nb_x>=0 and nb_x <=x_block-1 and nb_y>=0 and nb_y <= y_block-1):
                    if(blue_index_set[nb_x,nb_y]>-1):
                        nb_list_x.append(nb_x)
                        nb_list_y.append(nb_y)
                        nb_list_index.append(blue_index_set[nb_x, nb_y])





        blue_set_4nb_len[index1]=len(nb_list_x)
        #print(index1,blue_set_4nb_len[index1])
        #print(x_block)
        #print(y_block)
        #print(nb_list_x)
        #print(nb_list_y)
        #print(nb_list_index)
        for i, item in enumerate(nb_list_x):
          #print(i)

           blue_set_4nb[index1,i,0]=nb_list_x[i]
           blue_set_4nb[index1, i, 1] = nb_list_y[i]
           blue_set_4nb[index1, i, 2] = nb_list_index[i]
           #if(index1<=100):
           #  print(index1,i,nb_list_index[i])




    #assign number to blue points
    list_x=id_x
    list_y=id_y
    count_index=1;

    size_list=[];
    for index in range (0,blue_set_len):
        index_x=id_x[index]
        index_y=id_y[index]
        if(blue_label[index]>0):
            continue
        blue_label[index]=count_index
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        # 将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
        font_size = int(max(x_range, y_range) / 1600)
        #cv2.putText(blue_point,str(count_index), (step*index_x,step*index_y),font, font_size, (255, 255, 255), 5)
        count_index = count_index + 1
        q = queue.Queue()
        q.put(index)
        this_size=1;
        x_list=[];
        y_list=[];
        x_list.append(index_x);
        y_list.append(index_y);


        while not q.empty():
            head = q.get()
            for i in range(0,blue_set_4nb_len[head,0]):
                #print(blue_set_4nb_len[head,0])
                nb=blue_set_4nb[head,i,2];
                if(blue_label[nb]>0):
                    #if(index<=400):
                    #  print("show blue label",index,head,nb,blue_label[nb])
                    continue
                this_size=this_size+1

                blue_label[nb] = count_index-1
                nb_x = id_x[nb]
                nb_y = id_y[nb]
                x_list.append(nb_x);
                y_list.append(nb_y);
                #cv2.line(blue_point, (step * nb_x, step * nb_y), (step * index_x, step * index_y), blues, 5)
                q.put(nb)
        cv2.rectangle(blue_point,
                              (step * int(sum(x_list) / len(x_list)), step * int(sum(y_list) / len(y_list))), (
                              step * int(sum(x_list) / len(x_list)) + step,
                              step + step * int(sum(y_list) / len(y_list))), whites, -1);
        size_list.append(this_size);
        #print(this_size);
        #if(index<=500):
        # print(index,blue_set_4nb_len[index,0],this_size)
        #if(this_size>=10):
            #this_shape=findshape.findshape_v4(min(x_list), max(x_list), min(y_list), max(y_list));
            #if(this_shape>shape_threshold):
              #cv2.putText(blue_point,str(this_size), (step * int(sum(x_list)/len(x_list)),step * int(sum(y_list)/len(y_list))),font, font_size, (255, 255, 255), 10)
              #cv2.rectangle(blue_point, (step * int(sum(x_list)/len(x_list)),step * int(sum(y_list)/len(y_list))), (step * int(sum(x_list)/len(x_list))+step,step+step * int(sum(y_list)/len(y_list))), whites, -1);
              #size_list.append(this_size);


    count_blue=count_index-1;


    #for x in range(1, x_range,step):
    #     for y in range (1,y_range,step):
    #        cv2.rectangle(front, (x, y), (x+step, y+step), green, -1)

    #show background
    #two_fig = cv2.add(dst, front)
    two_fig = cv2.addWeighted(dst, 0.7, disk_image, 0.3, 0)

    #cv2.imshow("Canvas", two_fig) #13

    #cv2.resizeWindow("Canvas", 400, 400);


    #show blue points
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    font_size=int(max(x_range,y_range)/300)
    cv2.putText(blue_point,str(count_index-1), (0,int(y_range/6)),font, font_size, (255, 255, 0), 20)

    cv2.putText(blue_point,str(int(space_ratio*x_range*y_range)), (int(x_range/3),int(y_range/6)),font, font_size, (255, 255, 0), 20)


    #mix_blue = cv2.add(dst, blue_point)
    mix_blue=cv2.addWeighted(dst,0.7,blue_point,0.3,0)

    mix_black = cv2.addWeighted(two_fig, 0.7, bound_image, 0.3, 0)

    if(out_point==1):
        cv2.imwrite(blue_file, mix_blue)
    if(out_bound==1):
        cv2.imwrite(back_file, mix_black)
    if(out_hull==1):
        cv2.imwrite(snow_file, two_fig)


    #cv2.imshow("Canvas_show", blue_point) #13
    #cv2.imshow("Canvas_show", mix_blue) #13

    #cv2.resizeWindow("Canvas_show", 400, 400);

    #cv2.imshow("Original", canvas) #13

    #cv2.resizeWindow("Original", 400, 400);

    #cv2.imshow("blur", blurred) #13

    #cv2.resizeWindow("blur", 400, 400);

    #cv2.imshow("filter", dst) #13

    #cv2.resizeWindow("filter", 400, 400);

    #cv2.imshow("disk", disk_image) #13

    #cv2.resizeWindow("disk", 400, 400);

    #print(count_point/count)
    #print(1-back_count/sum_count)
    #print(count_blue)
    #cv2.waitKey(0) #14

    blue_point_num=count_index-1
    cell_size=int(x_range*y_range*space_ratio)

    fo = open("data.txt", "a")

    fo.writelines([image_name, ":", str(blue_point_num), ":", str(cell_size), ":", str(x_range), ":", str(y_range), ":",
                   str(space_ratio), '\n']);

    fo.close()

    #fo = open("size_dist.txt", "w")
    #print(size_list)
    #for item in size_list:
    #    fo.writelines([str(item),"\n"]);
    #fo.close();
    return (blue_point_num, cell_size, x_range, y_range,space_ratio )



def image2data2_para_cluster(image_file,bk_cfg_list, point_cfg_list, inside_cfg_list,out_point=1,out_bound=1,out_hull=1,alpha=1,shape_threshold=0):
    #out_point=1 => compute blue_point
    #out_bound=1=> compute bound_image, disk_image
    #out_hull=1> compute disk_image
    # block_per_dim=40 #for fig 3
    # block_per_dim=200 #for fig 2
    # block_per_dim=300 #for fig 2
    # global block_per_dim, conv_dist, boud_dist, back_ratio, boudn_ratio, limit, blue_range, red_range

    block_per_dim = 800  # for fig 1

    [avg_r_bk,avg_g_bk,avg_b_bk,std_r_bk,std_g_bk,std_b_bk]=bk_cfg_list
    [avg_r_pt, avg_g_pt, avg_b_pt, std_r_pt, std_g_pt, std_b_pt] = point_cfg_list
    [avg_r_in, avg_g_in, avg_b_in, std_r_in, std_g_in, std_b_in] = inside_cfg_list

    #bk_cluster_num=len(avg_r_bk)
    pt_cluster_num = len(avg_r_pt)
    #print(avg_r_pt,avg_g_pt,avg_b_pt,std_r_pt,std_g_pt,std_b_pt)
    #in_cluster_num = len(avg_r_in)

    conv_dist = 7;
    bound_dist = 10;
    back_ratio = 0.95
    bound_ratio = 0.92

    limit = 25;  # threshold for detecting background
    blue_range = 100  # upper bound for blue points in blue domain
    red_range = 90  # low bound for blue points in red domain
    blue_red_dst=15; # minimal gap between blue and red for blue points

    image_name=os.path.splitext(image_file)[0]
    out_file="Out"+os.path.splitext(image_file)[0]+".txt"

    if not os.path.exists('process'):
        os.makedirs('process')

    blue_file="process/blue_"+image_file
    back_file="process/bound_"+image_file
    snow_file = "process/hull_" + image_file

    #load image
    canvas = cv2.imread(image_file)
    #canvas = cv2.imread("test1.jpg")
    #canvas = cv2.imread("test2.jpg")
    #canvas = cv2.imread("test3.jpg")
    y_range,x_range,deepth=canvas.shape

    #print(x_range,y_range)
    front = np.zeros((y_range, x_range, 3), dtype="uint8") #background image
    blue_point = np.zeros((y_range, x_range, 3), dtype="uint8") #blue point image

    disk_image = np.zeros((y_range, x_range, 3), dtype="uint8") #back point image
    bound_image = np.zeros((y_range, x_range, 3), dtype="uint8")  # boundary point image

    #cv2.namedWindow("Canvas",0); #show background
    #cv2.namedWindow("Canvas_show",0); #show blue points
    #cv2.namedWindow("Original",0); #orignal image
    #cv2.namedWindow("blur",0);
    #cv2.namedWindow("filter",0);
    #cv2.namedWindow("disk",0);

    #this is the default names of colors
    blacks = (0, 0, 0)
    whites = (255, 255, 255)
    greens = (0, 255, 0)
    blues = (100, 100, 255)
    #################################33



    # blur the image
    kernel_size = (5, 5);
    sigma = 1.5;

    blurred = cv2.GaussianBlur(canvas,kernel_size,sigma)
    ########################3333


    #filter the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(blurred, -1, kernel=kernel)
    ###############333


    #statistics
    count=0;
    count_point=0;
    count_blue=0;
    ####################3

    #paramet setting
    #step=10;
    #

    step=math.ceil(max(x_range,y_range)/block_per_dim)
    #######################3

    #blue_set = np.zeros((block_per_dim, x_range, 3), dtype="uint8") #3


    #block size in x axis
    x_block=len(range(0, x_range,step))
    #block size in y axis
    y_block=len(range(0, y_range,step))

    #whether it is a blue block matrix
    blue_set = np.zeros((x_block,y_block), dtype="uint8")
    #whether it is a background block matrix
    back_set = np.zeros((y_block,x_block), dtype="uint8")
    bound_set = np.zeros((y_block,x_block), dtype="uint8")



    # sum_diff_ab=0;
    # sum_diff_ac=0;
    # for back_x in range(0,x_range,x_range-1):
    #    for back_y in range(0,y_range,y_range-1):
    #         f = list(dst[back_y,back_x])
    #         a = f[0]
    #         b = f[1]
    #         c = f[2]
    #         sum_diff_ab=abs(int(a)-int(b))+sum_diff_ab
    #         sum_diff_ac = abs(int(a)-int(c)) + sum_diff_ac
    #
    # limit=max(sum_diff_ab/4,sum_diff_ac/4)*5

    #b_threshold=10
    #nb_threshold = 5

    #back_set,bound_set = db.detect_bk(canvas, step, x_block, y_block)

    #back_set, bound_set = db.detect_bk_para(canvas, step, x_block, y_block,b_threshold, nb_threshold)

    bk_ratio = loadcfg.loadalpha("cfg/cfg_bk_ratio.txt")

    diff_r = max(0, abs(abs(float(avg_r_in) - float(avg_r_bk))) * bk_ratio)
    diff_g = max(0, abs(abs(float(avg_g_in) - float(avg_g_bk))) * bk_ratio)
    diff_b = max(0, abs(abs(float(avg_b_in) - float(avg_b_bk))) * bk_ratio)

    b_threshold = max(diff_r, diff_g, diff_b)
    nb_threshold = max(std_r_bk, std_b_bk, std_g_bk)



    bk_point=np.zeros((3), dtype="uint8")
    bk_point[0]=avg_r_bk;
    bk_point[1] = avg_g_bk;
    bk_point[2] = avg_b_bk;
    back_set, bound_set = db.detect_bk_para(canvas, step, x_block, y_block, b_threshold, nb_threshold,bk_point)


    space_ratio=round(1-np.sum(back_set)/(x_block*y_block),4)



    #detecting blue points and background points
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            count=count+1;
            f=list(dst[y,x])
            a=f[0]
            b=f[1]
            c=f[2]
            red = int(a)
            green = int(b)
            blue = int(c)
            ##print(r,g,b)
            #print(r,g,b)
            #print(abs(g-255),abs(b-255))

            # if max(abs(red-green),abs(red-blue))<=limit:
            #     #if abs(g-b)<=limit and abs(b-r)<=limit and abs(r-b)<=limit:
            #     #print(r, g, b)
            #     cv2.rectangle(front, (x, y), (x + step, y + step), whites, -1)
            #     back_set[x_count, y_count] = 1
            #     count_point=count_point+1;

            #if blue<blue_range and red>blue+blue_red_dst:
            pass_flag=0
            for test_index in range(pt_cluster_num):
                if (red > avg_r_pt[test_index] - alpha * std_r_pt[test_index] and red < avg_r_pt[test_index] + alpha * std_r_pt[test_index] and green > avg_g_pt[test_index] - alpha * std_g_pt[test_index] and green < avg_g_pt[test_index] + alpha * std_g_pt[test_index] and blue > avg_b_pt[test_index] - alpha * std_b_pt[test_index] and blue < avg_b_pt[test_index] + alpha * std_b_pt[test_index]):
                   pass_flag=pass_flag+1
            if(pass_flag>0):
            #if (red > avg_r_pt - alpha * std_r_pt and red < avg_r_pt + alpha * std_r_pt and green > avg_g_pt - alpha * std_g_pt and green < avg_g_pt + alpha * std_g_pt and blue > avg_b_pt - alpha * std_b_pt and blue < avg_b_pt + alpha * std_b_pt):
                #   print(r,g,b,x,y)
                #print(x,y,red,green,blue)
                 if (bound_set[y_count, x_count] == 0 and back_set[y_count, x_count] == 0):
                     if (shape_threshold == 0):
                        blue_set[x_count, y_count] = 1
                        if (out_point == 1):
                         cv2.rectangle(blue_point, (x, y), (x + step*3, y + step*3), blues, -1);
                     else:
                        shape_ratio = findshape.findshape_v2(dst, x, y, dst[y, x], 20, 50)
                        if (shape_ratio >= shape_threshold):
                            #print(shape_ratio)
                            blue_set[x_count, y_count] = 1
                            if (out_point == 1):
                             cv2.rectangle(blue_point, (x, y), (x + step*3, y + step*3), whites, -1);
                            # cv2.rectangle(blue_point, (x-15, y-15), (x + 15, y + 15), whites, -1);
                            #print("shape ratio", shape_ratio)
                #count_blue=count_blue+1;

            y_count=y_count+1
        x_count=x_count+1

    #recheck background blocks


    # print(back_count,blue_count,bound_count,other_count)
    x_count=0;
    y_count=0;
    for x in range(0, x_range,step):
        y_count=0
        for y in range (0,y_range,step):
            if(back_set[y_count,x_count]==1 and (out_hull==1 or out_bound==1)):
                cv2.rectangle(disk_image, (x, y), (x + step, y + step), whites, -1)
            if (bound_set[y_count, x_count] == 1 and out_bound==1):
                cv2.rectangle(bound_image, (x, y), (x + step, y + step), blues, -1)
            y_count=y_count+1
        x_count=x_count+1


    #set the list of blue points and  construct neighboring table
    id_x,id_y = np.where(blue_set==1)
    blue_set_len=len(id_x)

    blue_set_nb = np.zeros((blue_set_len,blue_set_len), dtype="uint8")

    blue_label = np.zeros((blue_set_len,1), dtype="uint32")

    for index1 in range (0,blue_set_len):
        index1_x = id_x[index1]
        index1_y = id_y[index1]
        for index2 in range (0,blue_set_len):
            index2_x = id_x[index2]
            index2_y = id_y[index2]
            if (index1==index2):
                continue
            if ((abs(index1_x-index2_x)<=1) and  (abs(index1_y-index2_y)<=1)):
                blue_set_nb[index1,index2]=1;
                blue_set_nb[index2,index1]=1;



    #assign number to blue points
    list_x=id_x
    list_y=id_y
    count_index=1;
    for index in range (0,blue_set_len):
        index_x=id_x[index]
        index_y=id_y[index]
        if(blue_label[index]>0):
            continue
        blue_label[index]=count_index
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        # 将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
        font_size = int(max(x_range, y_range) / 1600)
        #cv2.putText(blue_point,str(count_index), (step*index_x,step*index_y),font, font_size, (255, 255, 255), 5)
        count_index = count_index + 1
        q = queue.Queue()
        q.put(index)
        while not q.empty():
            head = q.get()
            for nb in range(0, blue_set_len):
                if(blue_set_nb[head, nb]==0):
                    continue
                if(blue_label[nb]>0):
                    continue
                blue_label[nb] = count_index-1
                nb_x = id_x[nb]
                nb_y = id_y[nb]
                cv2.line(blue_point, (step * nb_x, step * nb_y), (step * index_x, step * index_y), blues, 5)
                q.put(nb)


    count_blue=count_index-1;


    #for x in range(1, x_range,step):
    #     for y in range (1,y_range,step):
    #        cv2.rectangle(front, (x, y), (x+step, y+step), green, -1)

    #show background
    #two_fig = cv2.add(dst, front)
    two_fig = cv2.addWeighted(dst, 0.7, disk_image, 0.3, 0)

    #cv2.imshow("Canvas", two_fig) #13

    #cv2.resizeWindow("Canvas", 400, 400);


    #show blue points
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    font_size=int(max(x_range,y_range)/300)
    cv2.putText(blue_point,str(count_index-1), (0,int(y_range/6)),font, font_size, (255, 255, 0), 20)

    cv2.putText(blue_point,str(int(space_ratio*x_range*y_range)), (int(x_range/3),int(y_range/6)),font, font_size, (255, 255, 0), 20)


    #mix_blue = cv2.add(dst, blue_point)
    mix_blue=cv2.addWeighted(dst,0.7,blue_point,0.3,0)

    mix_black = cv2.addWeighted(two_fig, 0.7, bound_image, 0.3, 0)

    if(out_point==1):
        cv2.imwrite(blue_file, mix_blue)
    if(out_bound==1):
        cv2.imwrite(back_file, mix_black)
    if(out_hull==1):
        cv2.imwrite(snow_file, two_fig)


    #cv2.imshow("Canvas_show", blue_point) #13
    #cv2.imshow("Canvas_show", mix_blue) #13

    #cv2.resizeWindow("Canvas_show", 400, 400);

    #cv2.imshow("Original", canvas) #13

    #cv2.resizeWindow("Original", 400, 400);

    #cv2.imshow("blur", blurred) #13

    #cv2.resizeWindow("blur", 400, 400);

    #cv2.imshow("filter", dst) #13

    #cv2.resizeWindow("filter", 400, 400);

    #cv2.imshow("disk", disk_image) #13

    #cv2.resizeWindow("disk", 400, 400);

    #print(count_point/count)
    #print(1-back_count/sum_count)
    #print(count_blue)
    #cv2.waitKey(0) #14

    blue_point_num=count_index-1
    cell_size=int(x_range*y_range*space_ratio)

    fo = open("data.txt", "a")

    fo.writelines([image_name, ":", str(blue_point_num), ":", str(cell_size), ":", str(x_range), ":", str(y_range), ":",
                   str(space_ratio), '\n']);

    fo.close()
    return (blue_point_num, cell_size, x_range, y_range,space_ratio )