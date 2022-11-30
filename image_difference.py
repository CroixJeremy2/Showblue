#!/usr/bin/env python
# -*- coding: utf-8 -*-


' image2data2'

__author__ = 'Jun Zhang'

import numpy as np
import math
import cv2
import queue
import os
import detect_bk as db
import kmean_fun as kf
import kmean_fun


def diff_list(list_1, list_2):
    diff_list=[abs(float(list_1[i])-float(list_2[i])) for i in range(0,len(list_1))]
    return max(diff_list)

def image_differene(image_file1,image_file2):
    image_o = cv2.imread(image_file1)
    image_p = cv2.imread(image_file2)


    y_range, x_range, deepth = image_o.shape

    y2_range, x2_range, deepth = image_p.shape
    out = np.zeros((y_range, x_range, 3), dtype="uint8")
    #test_file="test.jpg"

    #print(y_range, x_range, y2_range, x2_range)
    #this is the default names of colors
    blacks = (0, 0, 0)
    whites = (255, 255, 255)
    greens = (0, 255, 0)
    blues = (100, 100, 255)

    r_list=[]
    g_list=[]
    b_list=[]

    r2_list=[]
    g2_list=[]
    b2_list=[]

    for x in range(0, x_range):
        for y in range(0, y_range):
            f=list(image_p[y,x])
            a=f[0]
            b=f[1]
            c=f[2]
            red = int(a)
            green = int(b)
            blue = int(c)

            f2 = list(image_o[y, x])

            if(diff_list(f,f2)>=40):
                r_list.append(f2[0])
                g_list.append(f2[1])
                b_list.append(f2[2])
                #cv2.rectangle(out, (x, y), (x, y), whites, -1)
            else:
                r2_list.append(f2[0])
                g2_list.append(f2[1])
                b2_list.append(f2[2])
                #cv2.rectangle(out, (x, y), (x + 1, y + 1), (int(f2[0]),int(f2[1]),int(f2[2])), -1)




    #print(r_list)
    #print(g_list)
    #print(b_list)

    #print(min(r_list),max(r_list))
    #print(min(g_list), max(g_list))
    #print("b")
    #print(min(b_list), max(b_list))
    #print("b")
    diff_v = [float(r_list[i]) - float(g_list[i]) for i in range(len(r_list))]
    #print("rg")
    c = [int(abs(ele)) for ele in diff_v]
    #print(c)
    #print( min(c),max(c))
    diff_v = [float(r_list[i]) - float(b_list[i]) for i in range(len(r_list))]
    #print("rb")
    c = [int(abs(ele)) for ele in diff_v]
    #print(c)
    #print(min(c), max(c))
    diff_v = [float(b_list[i]) - float(g_list[i]) for i in range(len(b_list))]
    #print("bg")
    c = [int(abs(ele)) for ele in diff_v]
    #print(c)
    #print(min(c), max(c))

    #print(sum(r_list) / len(r_list),sum(g_list) / len(g_list),sum(b_list) / len(b_list))
    #print(math.sqrt(np.var(r_list)),math.sqrt(np.var(g_list)),math.sqrt(np.var(b_list)))

    #print(sum(r2_list) / len(r2_list), sum(g2_list) / len(g2_list), sum(b2_list) / len(b2_list))
    #print(math.sqrt(np.var(r2_list)), math.sqrt(np.var(g2_list)), math.sqrt(np.var(b2_list)))

    #cv2.imwrite(test_file, out)
    return (r_list, g_list, b_list, r2_list, g2_list, b2_list)


def mark_image(image_in, image_out,r_list,g_list,b_list,alpha=1):
    image_o = cv2.imread(image_in)
    y_range, x_range, deepth = image_o.shape
    out = np.zeros((y_range, x_range, 3), dtype="uint8")
    whites = (255, 255, 255)
    blacks = (0, 0, 0)
    for x in range(0, x_range):
        for y in range(0, y_range):
            f = list(image_o[y, x])
            a = f[0]
            b = f[1]
            c = f[2]
            red = int(a)
            green = int(b)
            blue = int(c)

            if(red>=sum(r_list) / len(r_list)-alpha*math.sqrt(np.var(r_list)) and red<=sum(r_list) / len(r_list)+alpha*math.sqrt(np.var(r_list))
                and green>=sum(g_list) / len(g_list)-alpha*math.sqrt(np.var(g_list)) and green<=sum(g_list) / len(g_list)+alpha*math.sqrt(np.var(g_list))
                and blue>=sum(b_list) / len(b_list)-alpha*math.sqrt(np.var(b_list)) and blue<=sum(b_list) / len(b_list)+alpha*math.sqrt(np.var(b_list))):
                cv2.rectangle(out, (x, y), (x + 1, y + 1), whites, -1)
            else:
                cv2.rectangle(out, (x, y), (x + 1, y + 1), blacks, -1)
    cv2.imwrite(image_out, out)


def mark_image_3(image_ref,image_in, image_out,r_list,g_list,b_list,alpha=1):
    image_o = cv2.imread(image_in)
    image_r=  cv2.imread(image_ref)
    y_range, x_range, deepth = image_o.shape
    out = np.zeros((y_range, x_range, 3), dtype="uint8")
    whites = (255, 255, 255)
    blacks = (0, 0, 0)
    for x in range(0, x_range):
        for y in range(0, y_range):
            f = list(image_r[y, x])
            a = f[0]
            b = f[1]
            c = f[2]
            red = int(a)
            green = int(b)
            blue = int(c)

            f2 = list(image_o[y, x])
            a2 = f2[0]
            b2 = f2[1]
            c2 = f2[2]
            red2 = int(a2)
            green2 = int(b2)
            blue2 = int(c2)

            if(red>=sum(r_list) / len(r_list)-alpha*math.sqrt(np.var(r_list)) and red<=sum(r_list) / len(r_list)+alpha*math.sqrt(np.var(r_list))
                and green>=sum(g_list) / len(g_list)-alpha*math.sqrt(np.var(g_list)) and green<=sum(g_list) / len(g_list)+alpha*math.sqrt(np.var(g_list))
                and blue>=sum(b_list) / len(b_list)-alpha*math.sqrt(np.var(b_list)) and blue<=sum(b_list) / len(b_list)+alpha*math.sqrt(np.var(b_list))):
                cv2.rectangle(out, (x, y), (x + 1, y + 1), blacks, -1)
            else:
                cv2.rectangle(out, (x, y), (x + 1, y + 1), (red2,green2,blue2), -1)
    cv2.imwrite(image_out, out)


def get_avg_std(r_list, g_list, b_list):
    avg_r=sum(r_list) / len(r_list)
    avg_g = sum(g_list) / len(g_list)
    avg_b = sum(b_list) / len(b_list)
    std_r=math.sqrt(np.var(r_list))
    std_g = math.sqrt(np.var(g_list))
    std_b = math.sqrt(np.var(b_list))

    return (int(avg_r), int(avg_g), int(avg_b), int(std_r), int(std_g), int(std_b))


def get_cluster_avg_std(r_list, g_list, b_list,cluster_num=7, cutoff_threshold=0.65):
    dataSet = []
    counts=len(r_list)
    max_count=20000
    for index in range(counts):
        dataSet.append([r_list[index],g_list[index],b_list[index]])
        if (index >= max_count):
            break;

    total_count=index

    dataSet = np.mat(dataSet)
    cluster_num = 7
    centroids, clusterAssment = kmean_fun.kmeans(dataSet, cluster_num)
    cluster_array = np.squeeze(np.asarray(clusterAssment[:, 0]))
    cluster_list=[]
    cluster_ratio = []


    for num in range(cluster_num):
        #c = [1, 2, 2, 1, 1, 4, 5, 1]
        #idx = [idx for (idx, val) in enumerate(c) if val == 1]
        #print(idx)
        idx = [idx for (idx, val) in enumerate(cluster_array) if val == num]
        cluster_list.append(idx)
        cluster_ratio.append(len(cluster_list[num])/total_count)
        #print(cluster_list[num])

    sorted_index=sorted(range(len(cluster_ratio)), key=cluster_ratio.__getitem__, reverse=True)

    current_ratio=0
    avg_r= []
    avg_g = []
    avg_b = []
    std_r = []
    std_g = []
    std_b = []
    for index_ in range(cluster_num):
        hit_index=sorted_index[index_]
        current_ratio=current_ratio+cluster_ratio[hit_index]
        #print(current_ratio)


        avg_r.append(int(sum([r_list[i]  for i in cluster_list[hit_index]]) / len([r_list[i]  for i in cluster_list[hit_index]])))
        avg_g.append(int(sum([g_list[i] for i in cluster_list[hit_index]]) / len([g_list[i] for i in cluster_list[hit_index]])))
        avg_b.append(int(sum([b_list[i] for i in cluster_list[hit_index]]) / len([b_list[i] for i in cluster_list[hit_index]])))

        std_r.append(int(math.sqrt(np.var([r_list[i] for i in cluster_list[hit_index]]))))
        std_g.append(int(math.sqrt(np.var([g_list[i] for i in cluster_list[hit_index]]))))
        std_b.append(int(math.sqrt(np.var([b_list[i] for i in cluster_list[hit_index]]))))
        if(current_ratio>=cutoff_threshold):
            break

    return (avg_r, avg_g, avg_b, std_r, std_g, std_b)


def image_differene_quick(image_file1,image_file2):
    image_o = cv2.imread(image_file1)
    image_p = cv2.imread(image_file2)


    y_range, x_range, deepth = image_o.shape

    y2_range, x2_range, deepth = image_p.shape
    out = np.zeros((y_range, x_range, 3), dtype="uint8")
    #test_file="test.jpg"

    #print(y_range, x_range, y2_range, x2_range)
    #this is the default names of colors
    blacks = (0, 0, 0)
    whites = (255, 255, 255)
    greens = (0, 255, 0)
    blues = (100, 100, 255)

    r_list=[]
    g_list=[]
    b_list=[]

    r2_list=[]
    g2_list=[]
    b2_list=[]

    for x in range(0, x_range):
        for y in range(0, y_range):
            f=list(image_p[y,x])
            a=f[0]
            b=f[1]
            c=f[2]
            red = int(a)
            green = int(b)
            blue = int(c)

            f2 = list(image_o[y, x])

            if(diff_list(f,f2)>=40):
                r_list.append(f2[0])
                g_list.append(f2[1])
                b_list.append(f2[2])
                #cv2.rectangle(out, (x, y), (x, y), whites, -1)


    return (r_list, g_list, b_list)