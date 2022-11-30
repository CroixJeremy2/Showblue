#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Jun Zhang'

def loadcfg(bk_file):
    f = open(bk_file, "r")
    #skip txt for mean (r,g,b)
    read_data=f.readline()
    #read mean (r,g,b)
    read_data=f.readline()
    line=read_data.split(' ')
    #print(line)
    avg_r=int(line[0])
    avg_g=int(line[1])
    avg_b=int(line[2])


    read_data=f.readline()
    #read std_r, std_g, std_b
    read_data=f.readline()
    line=read_data.split(' ')
    #print(line)
    std_r=int(line[0])
    std_g=int(line[1])
    std_b=int(line[2])
    f.close()

    return (avg_r,avg_g,avg_b,std_r,std_g,std_b)

def savecfg(bk_file, avg_r, avg_g, avg_b, std_r, std_g, std_b, info, info2):
    f = open(bk_file, "w")
    f.write('%s\n' % info)
    f.write('%s %s %s\n' % (avg_r, avg_g, avg_b))
    f.write('%s\n' % info2)
    f.write('%s %s %s\n' % (std_r, std_g, std_b))
    f.close()

def loadalpha(alpha_file):
    f = open(alpha_file, "r")
    # skip txt for mean (r,g,b)
    read_data = f.readline()
    # read mean (r,g,b)
    read_data = f.readline()
    line = read_data.split(' ')
    # print(line)
    alpha = float(line[0])
    f.close()

    return alpha


def savecfg_cluster(bk_file, avg_r, avg_g, avg_b, std_r, std_g, std_b, info, info2):
    cluster_to_save=len(avg_r)
    f = open(bk_file, "w")
    f.write('color_cluster_number \n')
    f.write('%d\n' % cluster_to_save)
    f.write('%s\n' % info)
    for index in range(cluster_to_save):
     f.write('%s %s %s\n' % (avg_r[index], avg_g[index], avg_b[index]))
    f.write('%s\n' % info2)
    for index in range(cluster_to_save):
     f.write('%s %s %s\n' % (std_r[index], std_g[index], std_b[index]))
    f.close()


def loadcfg_cluster(bk_file):
    avg_r=[]
    avg_g=[]
    avg_b=[]
    std_r=[]
    std_g=[]
    std_b=[]
    f = open(bk_file, "r")
    #skip txt for mean (r,g,b)
    read_data=f.readline()
    read_data = f.readline()
    line=read_data.split('\n')
    cluster_num=int(line[0])
    read_data = f.readline()

    for num in range(cluster_num):

        #read mean (r,g,b)
        read_data=f.readline()
        line=read_data.split(' ')
        #print(line)
        avg_r.append(int(line[0]))
        avg_g.append(int(line[1]))
        avg_b.append(int(line[2]))


    read_data=f.readline()
    #read std_r, std_g, std_b
    for num in range(cluster_num):
     read_data=f.readline()
     line=read_data.split(' ')
     #print(line)
     std_r.append(int(line[0]))
     std_g.append(int(line[1]))
     std_b.append(int(line[2]))

    f.close()

    return (avg_r,avg_g,avg_b,std_r,std_g,std_b)