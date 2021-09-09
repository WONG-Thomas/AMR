# -*- coding: utf-8 -*-
"""
 @Author  : Huang Yuefeng
"""
import os
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

#find dark circle by density, dark circle is used to calculate angle
#this is kept for futer use, current version ring is fixed as a parameter
def detect_density_circle_by_center(img, x, y, max_d, distance, img_name):
    w,h = img.shape
    r = min(x,y,abs(h-x),abs(w-y)) 
    print (x,y,r,w,h)
    density = np.zeros((max(w,h)*2,4))
    for i in range(w):
        for j in range(h):
            d = round(((i-y)**2+(j-x)**2)**0.5)
            density[d][0] += 1
            density[d][1] += img[i][j]
    density = density[:r,]
    density[:,0][density[:,0]==0] = -1 
    density[:,2] = density[:,1]/density[:,0]
    print ('circle statistics:', max_d, density[:,0], density[:,1], density[:,2])
    if max_d > len(density):
        print ('error not enough circle ', len(density), 'at ', max_d)
        return None, None

    width = 10
    for i,s in enumerate(density):
        if i-width/2 < 0 or i+width/2 >= len(density):
            s[3] = s[2]
        else:
            s[3] = sum (density[i-int(width/2):i+int(width/2),2])/width
    offset = np.argmin(density[max_d-distance:max_d+distance, 3])
    max_d = max_d - distance + offset
    print ('circle diameter is:', max_d, density[max_d], density[:,3])

    for i in range(w):
        for j in range(h):
            if max_d == round(((i-y)**2+(j-x)**2)**0.5):
                if img[i][j] >= 250:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.circle(img,(x,y),3,(0,0,0),-1)
    cv2.imwrite('../result/'+img_name+'.circle.bmp', img)

    return max_d, img

#atan calculate angle, x and y is the center of circle, i and j is the image pixel coordinates
def cal_angle(x,y,i,j):
    dx = j-y
    dy = x-i
    if dx != 0:
        angle = math.atan(dy/dx)
        angle = math.degrees(angle)
        if dx >= 0 and dy >= 0:
            angle = angle
        elif dx >= 0 and dy < 0:
            angle = 360+angle
        elif dx <0 and dy >=0:
            angle = 180+angle
        else:
            angle = 180+angle
    else:
        if dy >=0:
            angle = 90
        else:
            angle = 270
    angle = round(angle)
    if angle ==  360:
        angle = 0

    return angle

#draw found angle in origianl image, x and y is center, found_a is found_angle, gt is ground truth, max_d is circle radius
def draw_result_by_center(img, y, x, found_a, gt, max_d, img_name):
    gt = round(gt)
    w,h = img.shape
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a or angle == gt:
                if img[i][j] >= 127:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.circle(img,(y,x),3,(0,0,0),-1)
    cv2.circle(img,(y,x),max_d,(255,255,255),1)
    cv2.imwrite('../result/'+img_name+'.result.bmp', img)

def find_density_angle_by_center(img, y, x, max_d, img_name):
    w,h = img.shape
    print (x,y, max_d)
    #demension: 1st is pixel count, 2nd is pixel sum, 3rd is pixel average, 4th is pixel smooth of 3rd by window
    density = np.zeros((360,4))
    for i in range(w):
        for j in range(h):
            d = round(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)

            if angle <0 or angle >= 360:
                print (angle, dy, dx, i, j)
                exit(1)

            density[angle][0] += 1
            density[angle][1] += img[i][j]

    density[:,2] = density[:,1]/density[:,0]
    #print ('angle ', density[:,2])

    #smooth angle density by window of 10 in cycle when near 0 and 360
    width = 10
    for i,s in enumerate(density):
        if i-int(width/2) < 0:
            s[3] = sum (density[i-int(width/2):,2])
            s[3] += sum (density[:i+int(width/2),2])
            s[3] /= width
        elif i+int(width/2) >= len(density):
            s[3] = sum (density[i-int(width/2):,2])
            s[3] += sum (density[:i+int(width/2)-len(density),2])
            s[3] /= width
        else:
            s[3] = sum (density[i-int(width/2):i+int(width/2),2])
            s[3] /= width
    found_a = np.argmax(density[:, 3])
    print ('found angle ', found_a, density[found_a])
    #print ('average ', density[:,3])

    #this is used to deal with the same value, when there is the same value the middle one is chosen.
    i_start = -1
    i_length = 0
    max_start = -1
    max_length = 0
    for i,s in enumerate(density[:,3]):
        if s == density[:,3][found_a]:
            if i_start == -1:
                i_start = i
            i_length += 1
        else:
            if i_length > max_length:
                max_length  = i_length
                max_start = i_start
                i_start = -1
                i_length = 0

    max_a = max_start + round(max_length/2)
    print ('adjust ', found_a, max_a, max_start, max_length )
    found_a = max_a

    #show direction result
    for i in range(w):
        for j in range(h):
            d = round(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a:
                    img[i][j] = 128 

    cv2.circle(img,(y,x),3,(0,0,0),-1)
    cv2.circle(img,(y,x),max_d,(128,128,128),1)
    cv2.imwrite('../result/'+img_name+'.direction.bmp', img)
    return found_a

#before calculate center, image should be enhanced.
def preprocess_for_center(im_in_path, im_out_path, ratio, im_name):
    img = cv2.imread(im_in_path+im_name,0)
    img = np.clip(img*ratio, 0,255).astype(np.uint8)
    im_path = '../result/'+im_name+'.center_processed.bmp'
    cv2.imwrite(im_out_path+im_name, img)

    return True

#this is used only for find good template, not be used in running time.
def preprocess_for_sbm(im_in_path, im_out_path, ratio, im_name):
    img = cv2.imread(im_in_path+im_name,0)
    img = np.clip(img*ratio, 0,255).astype(np.uint8)
    im_path = '../result/'+im_name+'.center_processed.bmp'
    cv2.imwrite(im_out_path+im_name+str(ratio)+'.png', img)

    return True

#before find angle, image also should be enhanced, this could be different from enhancement for find center
def preprocess_for_angle(im_in_path, ratio, im_name):
    print (im_in_path,im_name, ratio)
    img = cv2.imread(im_in_path+im_name,0)
    ori_img = img.copy()

    #img = cv2.equalizeHist(img)
    #img = cv2.medianBlur(img, 5)
    #img = cv2.GaussianBlur(img, (5, 5), 10)

    #grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 使用CV_32F防止数据溢出
    #grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    #gradx = cv2.convertScaleAbs(grad_x)
    #grady = cv2.convertScaleAbs(grad_y)
    #gradxy = cv2.addWeighted(gradx,0.5, grady, 0.5, 0)
    #gradxy = cv2.equalizeHist(gradxy)
    #_, gradxy = cv2.threshold(gradxy, 0, 255, cv2.THRESH_OTSU)
    #cv2.imwrite(im_edge_path, gradxy)

    img = np.clip(img*ratio, 0,255).astype(np.uint8)
    im_path = '../result/'+im_name+'.processed.bmp'
    cv2.imwrite(im_path, img)

    return ori_img, img

#get label
def check_label(name , dic_gt):
    if name not in dic_gt:
        print ("key error in dict label")
        print (name)
        return None
    else:
        return dic_gt[name] 

#call c++ program to find the template center
def find_center(dir_path, im_name, template_id):
    im_path = dir_path+im_name
    #im_path = preprocess(im_path, im_name)
    print (im_path)
    result = os.popen('./shape_based_matching/shape_based_matching_test test '+template_id+' '+im_path).readlines()
    os.system('cp ./shape_based_matching/test/AMR/test_result.png  '+'../result/'+im_name+'.sbm_result.png')
    print (result)
    result = result[-1].split(',')
    x = eval(result[0])
    y = eval(result[1])
    r = eval(result[2])
    angle  = eval(result[3])

    return x,y,r
   
#find direction in two steps:
#step1: find center of object. It has two cases.  
#####Case1: the image has obvious center. find object center directly.
#####Case2: the image has not obvious center, but its previous image (without object) has obvious center. Find object center in its previous image, and use it as current image center.
#step2: find lightest line from center to a circle with some radius. This mostly is the highlight part of object. This is the direction.
def deal_with_direction(dir_path, label_path):
    print (dir_path, label_path)
    dict_label = {}
    with open(label_path) as f_label:
        while(True):
            line = f_label.readline()
            items = line.split('\t')
            if len(items) != 2:
                print (line)
                break
            else:
                dict_label[items[0]] = items[1]
    i_total = 0
    i_tp5 = 0
    i_tp10 = 0
    max_delta = -1
    max_name = 0
    min_delta = 10000
    min_name = 0
    over_deltas = 0
    i_over = 0
    below_deltas = 0
    i_below = 0
    #sometimes center can't be found, they are recorded.
    no_center = {}
    #the iamges with big error of angle bigger than 10, are recorded.
    big_error = {}
    i_no_center = 0
    #parameter for algirhtm:
    #1st and 2nd are center offset which should be added to predict result.
    #3rd radius to calculate hightlight of object.
    #4th not used now
    #5th image enhancement parameter by Yi Zuotian, for finding center.
    #6th image enhancement parameter by Yi Zuotian, for highlight calculation.
    #5th and 6th must be float, and they maybe different.
    dict_radius = {
                   '00':[-1,1,100,10,1.0,2.5], 
                   '01':[-3,0,100,10,1.5,2.5], 
                   '02':[4,0,100,10,2.0,2.0], 
                   '03':[-4,1,100,10,1.5,2.0], 
                   '04':[-7,2,100,10,2.5,2.5],
                   '05':[-6,4,70,10,2.5,3.4], 
                   '06':[-1,0,100,10,2.0,2.0], 
                   '07':[-7,0,100,10,1.5,2.0], 
                   '08':[-3,-3,100,10,2.0,2.0]
                    }
    dict_result = {}
    for im_name in os.listdir(dir_path):
        #if im_name !=  '07_00576_Video_2021_08_03_103726_2.avi.jpg':
            #continue
        if im_name.endswith('.jpg'):
            arr = im_name.split('_')
            template_id = arr[0]
            if template_id != '08':
                continue
            if arr[1] == '00000':
                continue
            else:
                print(im_name)
                gt = check_label(im_name, dict_label)
                print ('gt', gt)
                #some images has not label
                if gt == None:
                    continue
                gt = eval(gt)

                #find center step
                im_out_path = '../../AMR_data/0908_processed/';
                ratio = dict_radius[template_id][4]
                preprocess_for_center(dir_path, im_out_path, ratio, im_name)
                if template_id in ['02', '06', '05', '00', '04']:
                    arr = im_name.split('_')
                    arr[1] = '00000'
                    secondly_im_name = '_'.join(arr)
                    preprocess_for_center(dir_path, im_out_path, ratio, secondly_im_name)
                    x,y,r = find_center(im_out_path, secondly_im_name, template_id)
                elif template_id in ['03', '01', '07', '08']:
                    x,y,r = find_center(im_out_path, im_name, template_id)
                else:
                    print ('error', im_name)
                    exit(1)
                if x == 0 and y == 0:
                    if template_id not in no_center:
                        no_center[template_id] = [im_name]
                    else:
                        no_center[template_id].append(im_name)
                    i_total += 1
                    i_no_center += 1
                    continue

                #find highlight direction step
                ratio = dict_radius[template_id][5]
                ori_img, img = preprocess_for_angle(dir_path, ratio, im_name)
                x += dict_radius[template_id][0]
                y += dict_radius[template_id][1]
                max_d = dict_radius[template_id][2]
                distantce = dict_radius[template_id][3]
                #max_d, img = detect_density_circle_by_center(img, x, y, max_d, distantce, im_name)
                print ('adjust ', max_d)
                if max_d == None:
                    angle = -1
                else:
                    angle = find_density_angle_by_center(img, x, y, max_d, im_name)
                draw_result_by_center(ori_img, x, y, angle, gt, max_d, im_name)
                if arr[0] in dict_result:
                    dict_result[arr[0]].append([im_name, angle, gt])
                else:
                    dict_result[arr[0]] = [[im_name, angle, gt]]

                #give a summary to result
                i_total += 1
                if angle == -1:
                    continue
                delta = angle-gt
                if delta > 180 or 180-delta < delta:
                    delta -= 180
                if delta >= -5 and delta <= 5:
                    i_tp5 += 1
                if delta >= -10 and delta <= 10:
                    i_tp10 += 1
                else:
                    if template_id not in big_error:
                        big_error[template_id] = [im_name]
                    else:
                        big_error[template_id].append(im_name)
                if delta > max_delta:
                    max_delta = delta 
                    max_name = im_name
                if delta < min_delta:
                    min_delta = delta
                    min_name = im_name
                if delta > 20 or delta < -20:
                    print ("error ", delta, " ", im_name)
                else:
                    if delta >= 0:
                        over_deltas += delta
                        i_over += 1
                    else:
                        below_deltas += delta
                        i_below += 1
            #break

    for key in sorted(dict_result):
        print (dict_result[key])
    print ("delta 5 precision is:", i_tp5/(i_total+0.000001), "details: ", i_tp5, " ", i_total)
    print ("delta 10 precision is:", i_tp10/(i_total+0.000001), "details: ", i_tp10, " ", i_total)
    print ("max error is:", max_delta, max_name)
    print ("min error is:", min_delta, min_name)
    print ("average over error is: ", over_deltas/(i_over+0.000001), " ", over_deltas, " ", i_over)
    print ("average below error is: ", below_deltas/(i_below+0.000001), " ", below_deltas, " ", i_below)
    print ("no center is: ", i_no_center)
    for key in sorted(no_center):
        print (no_center[key])
    print ("big error is: ", len(big_error))
    for key in sorted(big_error):
        print (big_error[key])

#this is used to check enhancement result in offline step
def preprocess_folder(src_folder_name, dst_folder_name):
    i = 0
    for im_name in os.listdir(src_folder_name):
        if im_name.endswith('.jpg'):
            for ratio in [1.0,1.5,2.0,2.5,3.0]:
                preprocess_for_sbm(src_folder_name, dst_folder_name, ratio, im_name)
                i += 1
                if i%100 == 0:
                    print (i)

def main():
    dict_center = {}
    dir_path = '../../AMR_data/0908/'
    label_path = '../../AMR_data/0902.txt'
    #preprocess_folder(dir_path, '../../AMR_data/0908_processed/')
    deal_with_direction(dir_path, label_path)

if __name__ == '__main__':
    main()
