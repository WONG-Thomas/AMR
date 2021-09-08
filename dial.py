# -*- coding: utf-8 -*-
"""
 @File    : dm_05.py
 @Time    : 2021/7/26 下午2:27
 @Author  : Huang Yuefeng
 @Description    :
 1. 原图变换到模板图像，而非模板图像变换到原图(done)
 2. 变换角度太大时，不做变换(done)
 3. 最终角度与标准角度比较(done)
 4. delta < 15改为delta < 20 （done)
"""
import os
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_process(img_path, img_name):
    ori_img = cv2.imread(img_path,0)
    cv2.imwrite('../result/'+img_name+'ori.bmp', ori_img)
    img = ori_img.copy()
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('../result/'+img_name+'grey.bmp', img)

    return ori_img, img

def detect_circle(img, img_name, ori_img):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,300,param1=50,param2=30,minRadius=50,maxRadius=200)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
        cv2.circle(img,(i[0],i[1]),2,(255,0,0),2)
    cv2.imwrite('../result/'+img_name+'circle.bmp', img)
    if len(circles[0]) == 0:
        print('no circle detect')
        exit(1)
    x,y,r = circles[0][0]

    roi = np.zeros(img.shape, np.uint8) 
    cv2.circle(roi, (x,y), r, (1,0,0), -1)
    img = ori_img * (roi.astype(img.dtype))
    img = img[max(int(y)-r, 0):min(int(y)+r, img.shape[0]), max(int(x)-r, 0):min(int(x)+r, img.shape[1])]
    cv2.imwrite('../result/'+img_name+'crop.bmp', img)
    img = cv2.resize(img, (501,501),cv2.INTER_AREA)
    cv2.imwrite('../result/'+img_name+'.resize.bmp', img)

    return img

def detect_circle(img, img_name, ori_img):
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,300,param1=50,param2=30,minRadius=50,maxRadius=200)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
        cv2.circle(img,(i[0],i[1]),2,(255,0,0),2)
    cv2.imwrite('../result/'+img_name+'circle.bmp', img)
    if len(circles[0]) == 0:
        print('no circle detect')
        exit(1)
    x,y,r = circles[0][0]

    roi = np.zeros(img.shape, np.uint8) 
    cv2.circle(roi, (x,y), r, (1,0,0), -1)
    img = ori_img * (roi.astype(img.dtype))
    img = img[max(int(y)-r, 0):min(int(y)+r, img.shape[0]), max(int(x)-r, 0):min(int(x)+r, img.shape[1])]
    cv2.imwrite('../result/'+img_name+'crop.bmp', img)
    img = cv2.resize(img, (501,501),cv2.INTER_AREA)
    cv2.imwrite('../result/'+img_name+'.resize.bmp', img)

    return img

def detect_line(img_path, img_name):
    img = cv2.imread(img_path,0)
    print (img.shape)
    cv2.imwrite('../result/grey.bmp', img)
    img = cv2.GaussianBlur(img,(5,5),3)
    #img = cv2.medianBlur(img, 3)
    cv2.imwrite('../result/blur.bmp', img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('../result/'+img_name+'binary.bmp', img)
    img = cv2.Canny(img, 50, 150, apertureSize=3)
    cv2.imwrite('../result/border.bmp', img)
    #img = cv2.Canny(img, 50, 150, apertureSize=3)
    minLineLength = 50
    maxLineGap = 4
    #lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, 10,
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 10,
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imwrite('../result/'+img_name+'line.bmp', img)

def find_small_big(start, arr, result):
    small,big = min(arr),max(arr)
    while(True):
        L = int(len(arr)/2)
        if L <= 1:
            result.append(['small', start, start+len(arr), len(arr), small, big, big-small])
            return
        small1,big1 = min(arr[:L]),max(arr[:L])
        small2,big2 = min(arr[L:]),max(arr[L:])
        #print ('L', len(arr), L, 'totoal ', small, big, 'seg1 ', small1, big1, 'seg2', small2, big2)
        if   (small == small1 and big == big2 and big1 <= small2):
            result.append(['increase', start, start+len(arr), len(arr), small, big, big-small])
            break
        elif (small == small2 and big == big1 and small1 >= big2):
            result.append(['decrease', start, start+len(arr), len(arr), small, big, big-small])
            break
        else:
            find_small_big(start, arr[:L], result)
            find_small_big(start+L, arr[L:], result)
            return 

def connect_seg(arr, source):
    result = []
    print (arr)
    now = None
    for a in arr:
        if now == None:
            now = a.copy()
            continue
        else:
            if now[0] == a[0]:
                now[2] = a[2]
                now[3] += a[3]
                now[4] = min(source[now[1]:now[2]])
                now[5] = max(source[now[1]:now[2]])
                now[6] = now[5]-now[4]
            else:
                result.append(now)
                now = a.copy()
    result.append(now)
    print ('after connected ', result)

    for i,r in enumerate(result):
        if r[0] == 'small':
            if i == 0:
                r[0] = result[1][0]
            elif i == len(result)-1:
                r[0] = result[-2][0]
            elif result[i-1][0] == result[i+1][0]:
                r[0] = result[i-1][0]
            elif result[i-1][-1] >= result[i+1][-1]:
                r[0] = result[i-1][0]
            else:
                r[0] = result[i+1][0]
    print ('small connected ', result)

    temp = result.copy()
    result = []
    now = None
    for a in temp:
        if now == None:
            now = a.copy()
            continue
        else:
            if now[0] == a[0]:
                now[2] = a[2]
                now[3] += a[3]
                now[4] = min(source[now[1]:now[2]])
                now[5] = max(source[now[1]:now[2]])
                now[6] = now[5]-now[4]
            else:
                result.append(now)
                now = a.copy()
    result.append(now)
    print ('again connected ', result)

    for i,r in enumerate(result):
        if i == 0 or i == len(result)-1:
            continue
        if r[-1] < result[i-1][-1]*0.3 and r[-1] < result[i+1][-1]*0.3:
            r[0] = result[i-1][0]
            print ('remove ', i)
    print ('again small connected ', result)

    temp = result.copy()
    result = []
    now = None
    for a in temp:
        if now == None:
            now = a.copy()
            continue
        else:
            if now[0] == a[0]:
                now[2] = a[2]
                now[3] += a[3]
                now[4] = min(source[now[1]:now[2]])
                now[5] = max(source[now[1]:now[2]])
                now[6] = now[5]-now[4]
            else:
                result.append(now)
                now = a.copy()
    result.append(now)
    print ('final connected ', result)

    return result

def detect_dark_circle(img, img_name):
    w,h = img.shape
    x,y = int(w/2),int(h/2)
    r = x
    print (x,y)
    statics = np.zeros((int(x)*2,3))
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            statics[d][0] += 1
            if img[i][j] == 0:
                statics[d][1] += 1
    statics[:,0][statics[:,0]==0] = -1 
    statics[:,2] = statics[:,1]/statics[:,0]
    result = []
    find_small_big(0, statics[:r,2], result)
    segs = connect_seg(result, statics[:r, 2])
    start,end = segs[0][1],segs[0][2]
    max_d = np.argmax(statics[start:end, 2])
    test_d = np.argmax(statics[:r,2])
    max_d = test_d
    print (statics[:r,2])
    print (max_d, test_d, statics[max_d])
    for i in range(w):
        for j in range(h):
            if max_d == int(((i-x)**2+(j-y)**2)**0.5):
                if img[i][j] >= 250:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.imwrite('../result/'+img_name+'dark_ring.bmp', img)

def detect_dark_circle2(img, img_name):
    w,h = img.shape
    x,y = int(w/2),int(h/2)
    r = x
    print (x,y)
    statics = np.zeros((int(x)*2,4))
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            statics[d][0] += 1
            if img[i][j] == 0:
                statics[d][1] += 1
    statics = statics[:r,]
    statics[:,0][statics[:,0]==0] = -1 
    statics[:,2] = statics[:,1]/statics[:,0]
    print (statics[:,2])

    width = 50
    for i,s in enumerate(statics):
        if i-width/2 < 0 or i+width/2 >= len(statics):
            continue
        s[3] = sum (statics[i-int(width/2):i+int(width/2),2])/width
    max_d = np.argmax(statics[:, 3])
    print (max_d, statics[:,3])

    for i in range(w):
        for j in range(h):
            if max_d == int(((i-x)**2+(j-y)**2)**0.5):
                if img[i][j] >= 250:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.imwrite('../result/'+img_name+'dark_ring2.bmp', img)

    return max_d, img

def detect_dark_circle_by_center(img, x, y, img_name):
    w,h = img.shape
    r = min(x,y,abs(h-x),abs(w-y)) 
    print (x,y,r,w,h)
    statics = np.zeros((max(w,h)*2,4))
    for i in range(w):
        for j in range(h):
            d = int(((i-y)**2+(j-x)**2)**0.5)
            statics[d][0] += 1
            if img[i][j] == 0:
                statics[d][1] += 1
    statics = statics[:r,]
    statics[:,0][statics[:,0]==0] = -1 
    statics[:,2] = statics[:,1]/statics[:,0]
    print (statics[:,2])

    width = 10
    for i,s in enumerate(statics):
        if i-width/2 < 0 or i+width/2 >= len(statics):
            continue
        s[3] = sum (statics[i-int(width/2):i+int(width/2),2])/width
    max_d = np.argmax(statics[:, 3])
    print (max_d, statics[:,3])

    for i in range(w):
        for j in range(h):
            if max_d == int(((i-y)**2+(j-x)**2)**0.5):
                if img[i][j] >= 250:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.circle(img,(x,y),3,(0,0,0),-1)
    cv2.imwrite('../result/'+img_name+'.circle.bmp', img)

    return max_d, img

def detect_density_circle_by_center(img, x, y, max_d, distance, img_name):
    w,h = img.shape
    r = min(x,y,abs(h-x),abs(w-y)) 
    print (x,y,r,w,h)
    statics = np.zeros((max(w,h)*2,4))
    for i in range(w):
        for j in range(h):
            d = round(((i-y)**2+(j-x)**2)**0.5)
            statics[d][0] += 1
            statics[d][1] += img[i][j]
    statics = statics[:r,]
    statics[:,0][statics[:,0]==0] = -1 
    statics[:,2] = statics[:,1]/statics[:,0]
    print ('circle statistics:', max_d, statics[:,0], statics[:,1], statics[:,2])
    if max_d > len(statics):
        print ('error not enough circle ', len(statics), 'at ', max_d)
        return None, None

    width = 10
    for i,s in enumerate(statics):
        if i-width/2 < 0 or i+width/2 >= len(statics):
            s[3] = s[2]
        else:
            s[3] = sum (statics[i-int(width/2):i+int(width/2),2])/width
    offset = np.argmin(statics[max_d-distance:max_d+distance, 3])
    max_d = max_d - distance + offset
    print ('circle diameter is:', max_d, statics[max_d], statics[:,3])

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

def find_angle(img, max_d, img_name):
    w,h = img.shape
    x,y = int(w/2),int(h/2)
    print (x,y, max_d)
    statics = np.zeros((360,4))
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)

            if angle <0 or angle >= 360:
                print (angle, dy, dx, i, j)
                exit(1)

            statics[angle][0] += 1
            if img[i][j] == 0:
                statics[angle][1] += 1

    statics[:,2] = statics[:,1]/statics[:,0]
    found_a = np.argmin(statics[:, 2])
    print ('angle ', statics[:,2])
    print ('found angle ', found_a)

    width = 10
    for i,s in enumerate(statics):
        if i-int(width/2) < 0:
            s[3] = sum (statics[i-int(width/2):,2])
            print (i, i-width/2, s[3])
            s[3] += sum (statics[:i+int(width/2),2])
            print (i, i+width/2, s[3])
            s[3] /= width
        elif i+int(width/2) >= len(statics):
            s[3] = sum (statics[i-int(width/2):,2])
            s[3] += sum (statics[:i+int(width/2)-len(statics),2])
            s[3] /= width
        else:
            s[3] = sum (statics[i-int(width/2):i+int(width/2),2])
            s[3] /= width
    found_a = np.argmin(statics[:, 3])
    print ('average ', statics[:,3])
    print (found_a )

    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a:
                if img[i][j] >= 200:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 

    cv2.circle(img,(x,y),3,(0,0,0),-1)
    cv2.imwrite('../result/'+img_name+'light_direction.bmp', img)
    return found_a

def draw_result(img, found_a, max_d, img_name):
    w,h = img.shape
    x,y = int(w/2),int(h/2)
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a:
                if img[i][j] >= 127:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.circle(img,(x,y),3,(0,0,0),-1)
    cv2.imwrite('../result/'+img_name+'.result.bmp', img)

def draw_result_by_center(img, y, x, found_a, gt, max_d, img_name):
    gt = round(gt)
    w,h = img.shape
    #x,y = int(w/1),int(h/2)
    for i in range(w):
        for j in range(h):
            d = int(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a or angle == gt:
            #if angle == gt:
                if img[i][j] >= 127:
                    img[i][j] = 0
                else:
                    img[i][j] = 255 
    cv2.circle(img,(y,x),3,(0,0,0),-1)
    cv2.circle(img,(y,x),max_d,(255,255,255),1)
    cv2.imwrite('../result/'+img_name+'.result.bmp', img)

def detect_direction(img, img_name):
    ori_img = img.copy()
    img = cv2.GaussianBlur(img,(5,5),3)
    #cv2.imwrite('./'+img_name+'blur.bmp', img)
    img = cv2.equalizeHist(img)
    #cv2.imwrite('./'+img_name+'equal.bmp', img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    #cv2.imwrite('./'+img_name+'binary.bmp', img)
    max_d, img = detect_dark_circle2(img, img_name)
    angle = find_angle(img, max_d, img_name)
    draw_result(ori_img, angle, max_d, img_name)

def detect(dir_path):
    i = 0
    for im_name in os.listdir(dir_path):
        if im_name.endswith('.jpg'):
            if im_name != '01_00884.jpg':
                continue
            print(im_name)
            im_path = os.path.join(dir_path, im_name)
            ori_img,img = read_process(im_path, im_name)
            sub_img = detect_circle(img, im_name, ori_img)
            detect_direction(sub_img, im_name)
            i += 1
            if i == 50:
                exit(1)


def find_angle_by_center(img, y, x, max_d, img_name):
    w,h = img.shape
    #x,y = int(w/2),int(h/2)
    print (x,y, max_d)
    statics = np.zeros((360,4))
    for i in range(w):
        for j in range(h):
            d = round(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)

            if angle <0 or angle >= 360:
                print (angle, dy, dx, i, j)
                exit(1)

            statics[angle][0] += 1
            if img[i][j] == 0:
                statics[angle][1] += 1

    statics[:,2] = statics[:,1]/statics[:,0]
    found_a = np.argmin(statics[:, 2])
    print ('angle ', statics[:,2])
    #print ('found angle ', found_a)

    width = 10
    for i,s in enumerate(statics):
        if i-int(width/2) < 0:
            s[3] = sum (statics[i-int(width/2):,2])
            print (i, i-width/2, s[3])
            s[3] += sum (statics[:i+int(width/2),2])
            print (i, i+width/2, s[3])
            s[3] /= width
        elif i+int(width/2) >= len(statics):
            s[3] = sum (statics[i-int(width/2):,2])
            s[3] += sum (statics[:i+int(width/2)-len(statics),2])
            s[3] /= width
        else:
            s[3] = sum (statics[i-int(width/2):i+int(width/2),2])
            s[3] /= width
    found_a = np.argmin(statics[:, 3])
    i_start = -1
    i_length = 0
    max_start = -1
    max_length = 0
    for i,s in enumerate(statics[:,3]):
        if s == statics[:,3][found_a]:
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
    print ('average ', statics[:,3])
    print ('adjust ', found_a, max_a, max_start, max_length )
    found_a = max_a

    for i in range(w):
        for j in range(h):
            d = round(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)
            if angle == found_a:
                    img[i][j] = 128 

    cv2.circle(img,(y,x),3,(0,0,0),-1)
    #cv2.circle(img,(y,x),max_d,(128,128,128),1)
    cv2.imwrite('../result/'+img_name+'.direction.bmp', img)
    return found_a

def find_density_angle_by_center(img, y, x, max_d, img_name):
    w,h = img.shape
    #x,y = int(w/2),int(h/2)
    print (x,y, max_d)
    statics = np.zeros((360,4))
    for i in range(w):
        for j in range(h):
            d = round(((i-x)**2+(j-y)**2)**0.5)
            if d >= max_d:
                continue
            angle = cal_angle(x,y,i,j)

            if angle <0 or angle >= 360:
                print (angle, dy, dx, i, j)
                exit(1)

            statics[angle][0] += 1
            statics[angle][1] += img[i][j]

    statics[:,2] = statics[:,1]/statics[:,0]
    #print ('angle ', statics[:,2])

    width = 10
    for i,s in enumerate(statics):
        if i-int(width/2) < 0:
            s[3] = sum (statics[i-int(width/2):,2])
            print (i, i-width/2, s[3])
            s[3] += sum (statics[:i+int(width/2),2])
            print (i, i+width/2, s[3])
            s[3] /= width
        elif i+int(width/2) >= len(statics):
            s[3] = sum (statics[i-int(width/2):,2])
            s[3] += sum (statics[:i+int(width/2)-len(statics),2])
            s[3] /= width
        else:
            s[3] = sum (statics[i-int(width/2):i+int(width/2),2])
            s[3] /= width
    found_a = np.argmax(statics[:, 3])
    print ('found angle ', found_a, statics[found_a])
    i_start = -1
    i_length = 0
    max_start = -1
    max_length = 0
    for i,s in enumerate(statics[:,3]):
        if s == statics[:,3][found_a]:
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
    #print ('average ', statics[:,3])
    print ('adjust ', found_a, max_a, max_start, max_length )
    found_a = max_a

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


def preprocess_for_center(im_in_path, im_out_path, ratio, im_name):
    img = cv2.imread(im_in_path+im_name,0)
    img = np.clip(img*ratio, 0,255).astype(np.uint8)
    im_path = '../result/'+im_name+'.center_processed.bmp'
    cv2.imwrite(im_out_path+im_name, img)

    return True

def preprocess_for_sbm(im_in_path, im_out_path, ratio, im_name):
    img = cv2.imread(im_in_path+im_name,0)
    img = np.clip(img*ratio, 0,255).astype(np.uint8)
    im_path = '../result/'+im_name+'.center_processed.bmp'
    cv2.imwrite(im_out_path+im_name+str(ratio)+'.png', img)

    return True

def preprocess_for_direction(im_in_path, ratio, im_name):
    print (im_in_path,im_name, ratio)
    img = cv2.imread(im_in_path+im_name,0)
    ori_img = img.copy()
    #w,h = img.shape
    #if w > h:
    #    img = img[w-h:,:]
    #else:
    #    img = img[:,h-w:]
    #img = cv2.resize(img, (512,512),cv2.INTER_CUBIC)

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

    img = np.clip(img*ratio, 0,255).astype(np.uint8)

    im_path = '../result/'+im_name+'.processed.bmp'
    #im_edge_path = '../result/'+im_name+'.edge.bmp'
    #cv2.imwrite(im_path, img)
    cv2.imwrite(im_path, img)
    #cv2.imwrite(im_edge_path, gradxy)

    return ori_img, img

def find_center(im_path, im_name):
    im_path = preprocess(im_path, im_name)
    print (im_path)
    result = os.popen('../shape_based_matching/shape_based_matching_test test '+im_path).readlines()
    print (result)
    result = result[-1].split(',')
    x = eval(result[0])
    y = eval(result[1])
    r = eval(result[2])

    return x,y,r

def find_max_d_by_center(im_path, im_name, x, y):
    img = cv2.imread(im_path,0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite('../result/'+im_name+'.binary.bmp', img)
    max_d, img = detect_dark_circle_by_center(img, x, y, im_name)
    return max_d

def deal_with_bg(dir_path, dict_center):
    for im_name in os.listdir(dir_path):
        if im_name.endswith('.jpg'):
            arr = im_name.split('_')
            if arr[1] != '00000':
            #if im_name != '08_00884.jpg':
                continue
            else:
                print ('ok')
            im_path = dir_path+im_name
            x,y,r = find_center(dir_path, im_name)
            if x == -1:
                print ('cant find center ', im_name)
                exit(1)
            img = cv2.imread(im_path,0)
            cv2.circle(img,(x,y),2,(0,0,0),-1)
            cv2.circle(img,(x,y),r,(0,0,0),2)
            cv2.imwrite('../result/'+im_name+'.bgcenter.bmp', img)
            max_d = find_max_d_by_center(im_path, im_name, x, y)

            key = dir_path+arr[0]
            dict_center[key] = (x,y,max_d)

def deal_with_label(name, value):
    i_total = 0
    i_right = 0
    if name == '00':
        i_total += 1
        if value != 110:
            print (name, "should be 110, but it is ", value)
        else:
            i_right += 1
    elif name == '01':
        i_total += 1
        if value != 110:
            print (name, "should be 110, but it is ", value)
        else:
            i_right += 1
    elif name == '02':
        i_total += 1
        if value != 130:
            print (name, "should be 130, but it is ", value)
        else:
            i_right += 1
    elif name == '03':
        i_total += 1
        if value != 140:
            print (name, "should be 140, but it is ", value)
        else:
            i_right += 1
    elif name == '04':
        i_total += 1
        if value != 110:
            print (name, "should be 110, but it is ", value)
        else:
            i_right += 1
    elif name == '05':
        i_total += 1
        if value != 110:
            print (name, "should be 110, but it is ", value)
        else:
            i_right += 1
    elif name == '06':
        i_total += 1
        if value != 110:
            print (name, "should be 110, but it is ", value)
        else:
            i_right += 1
    elif name == '07':
        i_total += 1
        if value != 65:
            print (name, "should be 65, but it is ", value)
        else:
            i_right += 1
    elif name == '08':
        i_total += 1
        if value != 80:
            print (name, "should be 80, but it is ", value)
        else:
            i_right += 1
    else:
        print ("error there is not such imgae considered:", name)

def check_label(name, predict, dic_gt):
    if name not in dic_gt:
        print ("key error in dict label")
        print (name)
        print (dic_gt)
        exit(1)
    else:
        return dic_gt[name] 

def deal_with_fg(dir_path, dict_center):
    dict_result = {}
    for im_name in os.listdir(dir_path):
    #    if im_name !=  '07_00664.jpg':
    #        continue
        if im_name.endswith('.jpg'):
            arr = im_name.split('_')
            position = arr[0]
            if arr[1] == '00000':
                continue
            else:
                print(im_name)
                img = cv2.imread(dir_path+im_name,0)
                ori_img = img.copy()
                #img = preprocess(img, im_name)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
                im_path = dir_path+im_name
                if position in ['02', '06', '05', '00', '04']:
                    new_x,new_y,new_r = find_center(im_path, im_name)
                    key = dir_path+position
                    x,y,max_d = dict_center[key]
                elif position in ['08', '03', '01', '07']:
                    x,y,r = find_center(im_path, im_name)
                    max_d = find_max_d_by_center(im_path, im_name, x, y)
                    key = dir_path+position
                    new_x,new_y,new_r = dict_center[key]
                else:
                    print ('error', im_name)
                    exit(1)

                angle = find_angle_by_center(img, x, y, max_d, im_name)
                draw_result_by_center(ori_img, x, y, angle, max_d, im_name)
                if arr[0] in dict_result:
                    dict_result[arr[0]].append([im_name, angle])
                else:
                    dict_result[arr[0]] = [[im_name, angle]]
                deal_with_label(arr[0], angle)

                cv2.circle(ori_img,(new_x,new_y),3,(0,0,0),-1)
                cv2.circle(ori_img,(new_x,new_y),new_r,(0,0,0),1)
                cv2.imwrite('../result/'+im_name+'.center.bmp', ori_img)
        print (dict_result)
        for key in sorted(dict_result):
            print (dict_result[key])


def find_center_directly(dir_path, im_name, template_id):
    im_path = dir_path+im_name
    #im_path = preprocess(im_path, im_name)
    print (im_path)
    result = os.popen('../shape_based_matching/shape_based_matching_test test '+template_id+' '+im_path).readlines()
    os.system('cp ../shape_based_matching/test/AMR/test_result.png  '+'../result/'+im_name+'.sbm_result.png')
    print (result)
    result = result[-1].split(',')
    x = eval(result[0])
    y = eval(result[1])
    r = eval(result[2])
    angle  = eval(result[3])

    return x,y,r

def find_center_secondly(dir_path, im_name, template_id):
    print ('secondly input image ', dir_path+im_name)
    arr = im_name.split('_')
    arr[1] = '00000'
    im_path = dir_path+'_'.join(arr)
    #im_path = preprocess(im_path, template_image_name)
    print ('secondly real image ', im_path)
    result = os.popen('../shape_based_matching/shape_based_matching_test test '+template_id+' '+im_path).readlines()
    os.system('cp ../shape_based_matching/test/AMR/test_result.png  '+'../result/'+im_name+'.sbm_result.png')
    print (result)
    result = result[-1].split(',')
    x = eval(result[0])
    y = eval(result[1])
    r = eval(result[2])
    angle  = eval(result[3])

    return x,y,r
    
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
    no_center = {}
    big_error = {}
    i_no_center = 0
    dict_radius = {
                   '00':[-2,0,100,10,1.0,2.0], 
                   '01':[-3,0,100,10,1.5,2.5], 
                   '02':[3,0,100,10,2.0,2.0], 
                   '03':[-5,1,100,10,1.5,2.0], 
                   '04':[-7,0,100,10,2.5,2.5],
                   '05':[-9,1,70,10,2.5,3.0], 
                   '06':[-1,0,100,10,2.0,2.0], 
                   '07':[-5,0,100,10,1.5,2.0], 
                   '08':[-3,-3,100,10,2.0,2.0]
                    }
    dict_result = {}
    for im_name in os.listdir(dir_path):
        #if im_name !=  '01_00912_Video_2021_08_25_103524_3.avi.jpg':
            #continue
        if im_name.endswith('.jpg'):
            arr = im_name.split('_')
            template_id = arr[0]
            if template_id != '01':
                continue
            if arr[1] == '00000':
                continue
            else:
                print(im_name)
                im_out_path = '../../AMR_data/0902_processed/';
                ratio = dict_radius[template_id][4]
                preprocess_for_center(dir_path, im_out_path, ratio, im_name)
                if template_id in ['02', '06', '05', '00', '04']:
                    arr = im_name.split('_')
                    arr[1] = '00000'
                    secondly_im_name = '_'.join(arr)
                    preprocess_for_center(dir_path, im_out_path, ratio, secondly_im_name)
                    x,y,r = find_center_directly(im_out_path, secondly_im_name, template_id)
                elif template_id in ['03', '01', '07', '08']:
                    x,y,r = find_center_directly(im_out_path, im_name, template_id)
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

                #im_path = dir_path+im_name
                ratio = dict_radius[template_id][5]
                ori_img, img = preprocess_for_direction(dir_path, ratio, im_name)

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
                gt = check_label(im_name, angle, dict_label)
                gt = eval(gt)
                draw_result_by_center(ori_img, x, y, angle, gt, max_d, im_name)
                if arr[0] in dict_result:
                    dict_result[arr[0]].append([im_name, angle, gt])
                else:
                    dict_result[arr[0]] = [[im_name, angle, gt]]

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
    print ("delta 5 precision is:", i_tp5/i_total, "details: ", i_tp5, " ", i_total)
    print ("delta 10 precision is:", i_tp10/i_total, "details: ", i_tp10, " ", i_total)
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
    dir_path = '../../AMR_data/0902/'
    #dir_path = '../../AMR_data/0902_processed/'
    label_path = '../../AMR_data/0902.txt'
    #preprocess_folder(dir_path, '../../AMR_data/0902_processed/')
    deal_with_direction(dir_path, label_path)

if __name__ == '__main__':
    main()
