import os, fnmatch
import time

from functools import cmp_to_key
import image2data2 as ig
#import image2data3 as ig
import loadcfg

DIR = os.getcwd()

def compare(x, y):
    stat_x = os.stat(DIR + "/" + x)

    stat_y = os.stat(DIR + "/" + y)

    if stat_x.st_ctime < stat_y.st_ctime:

        return -1

    elif stat_x.st_ctime > stat_y.st_ctime:

        return 1

    else:

        return 0


#iterms = os.listdir(DIR)


iterms_jpg=fnmatch.filter(os.listdir('.'), '*.jpg')
iterms_jpeg=fnmatch.filter(os.listdir('.'), '*.jpeg')
iterms_tif=fnmatch.filter(os.listdir('.'), '*.tif')
iterms=iterms_jpg+iterms_jpeg+iterms_tif
print(iterms)

iterms.sort(key = cmp_to_key(compare))

count=0;
print("output format: figurename, blue_points,cell_size,x_range,y_range,ratio")

avg_r, avg_g, avg_b, std_r, std_g, std_b=loadcfg.loadcfg("cfg/cfg_bk.txt")
cfg_bk=[avg_r, avg_g, avg_b, std_r, std_g, std_b]
avg_r, avg_g, avg_b, std_r, std_g, std_b=loadcfg.loadcfg("cfg/cfg_point.txt")
cfg_pt=[avg_r, avg_g, avg_b, std_r, std_g, std_b]
avg_r, avg_g, avg_b, std_r, std_g, std_b=loadcfg.loadcfg("cfg/cfg_in.txt")
cfg_in=[avg_r, avg_g, avg_b, std_r, std_g, std_b]

alpha=loadcfg.loadalpha("cfg/cfg_alpha.txt")
shape_ratio=loadcfg.loadalpha("cfg/cfg_shape.txt")

#new for time cmputing
last_time=time.time()
#
num_files=len(iterms)
avg_time=120
num_processed=0
print("number of files to treat: ", num_files)
for iterm in iterms:
    print("estimation of resting time: ", (num_files-num_processed)*avg_time/60.0, "minutes")
    print("treating file:"+iterm)
    #blue_points, cell_size, x_range, y_range, ratio = ig.image2data(iterm)
    #blue_points, cell_size, x_range, y_range, ratio = ig.image2data3_para(iterm, cfg_bk, cfg_pt, cfg_in,1, 1, 1,alpha,shape_ratio)
    blue_points, cell_size, x_range, y_range, ratio = ig.image2data2_para(iterm, cfg_bk, cfg_pt, cfg_in, 1, 1, 1, alpha,
                                                                          shape_ratio)
    count=count+1
    localtime = time.asctime(time.localtime(time.time()))
    #new for time estimation
    pass_time=time.time()-last_time
    num_processed=num_processed+1
    avg_time=pass_time/num_processed
    #
    print("Curreen time is:", localtime)
    print(str(count)+" files has been processed, check folder process and the file data.txt")