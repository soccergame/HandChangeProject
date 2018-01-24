# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np
import SkinColorExtract
import matplotlib.pyplot as plt
import HoughTransform
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
from itertools import product, combinations
import json

def find_max_contours(image, isSecondFind=False):
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_area = image.shape[0] * image.shape[1]
    max_area = -1
    hand_idx = -1
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= 0.7 * image_area and isSecondFind:
            continue

        if isSecondFind:
            min_x_cnt = np.min(cnt[:, :, 0])
            max_x_cnt = np.max(cnt[:, :, 0])
            min_y_cnt = np.min(cnt[:, :, 1])
            if (min_x_cnt < 0.05 * image.shape[1] and 
                max_x_cnt > 0.95 * image.shape[1] and
                min_y_cnt < 0.05 * image.shape[0]):
                continue

        #imgSkin = np.zeros_like(image, dtype=np.uint8)
        #cv2.drawContours(imgSkin, [cnt], 0, 255, -1)
        #if imgSkin[int(0.5*image.shape[0]), int(0.5*image.shape[1])] != 255:
        #    continue

        if max_area < area:
            max_area = area
            hand_idx = i

    if hand_idx >= 0:
        imgSkin = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(imgSkin, contours, hand_idx, 255, -1)
        imgSkin[(imgSkin < 255)] = 0
        if imgSkin[int(0.5*image.shape[0]), int(0.5*image.shape[1])] != 255:
            imgSkin = None
    else:
        imgSkin = None

    return contours[hand_idx], max_area, imgSkin

def detect_contour(image, imageMask=None, isCvtColor=False, isEqHist=False, isForCut=True):
    if isCvtColor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if isEqHist:
        image = cv2.equalizeHist(image)
    if imageMask is not None:
        imageMask[imageMask > 0] = 1
        hand_img = image * imageMask
    else:
        hand_img = image
    hand_img = cv2.GaussianBlur(hand_img,(3, 3), 0) 
    hand_img = cv2.Canny(hand_img, 50, 150)
    cv2.dilate(hand_img, np.ones([3, 3]), hand_img)
    if isForCut:
        cv2.dilate(hand_img, np.ones([3, 3]), hand_img, iterations=5)
        cv2.erode(hand_img, np.ones([3, 3]), hand_img, iterations=5)

    return hand_img

def find_mser_region(img, centerY, centerX, start, end, step, ave_color):
    base_area = 0
    min_area = 0
    max_area = 0
    change_delta = 1
    q_ratio = []
    start = max(change_delta, start)
    for i in range(start, end, step):
        # 先求Qi的面积
        if i == start:
            base_img = copy.deepcopy(img)
            base_img[base_img < (ave_color-i)] = 0
            base_img[base_img > (ave_color+i)] = 0
            base_img[base_img > 0] = 255
            #plt.imshow(base_img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
            #plt.show()
            label_image = measure.label(base_img, connectivity = 2)
            label = label_image[centerY, centerX]
            base_area = np.where(label_image == label)[0].shape[0]

            # 再求Q(i-delta)的面积
            base_img = copy.deepcopy(img)
            base_img[base_img < (ave_color-(i-change_delta))] = 0
            base_img[base_img > (ave_color+(i-change_delta))] = 0
            base_img[base_img > 0] = 255
            label_image = measure.label(base_img, connectivity = 2)
            label = label_image[centerY, centerX]
            min_area = np.where(label_image == label)[0].shape[0]

            # 再求Q(i+delta)的面积
            base_img = copy.deepcopy(img)
            base_img[base_img < (ave_color-(i+change_delta))] = 0
            base_img[base_img > (ave_color+(i+change_delta))] = 0
            base_img[base_img > 0] = 255
            label_image = measure.label(base_img, connectivity = 2)
            label = label_image[centerY, centerX]
            max_area = np.where(label_image == label)[0].shape[0]
        else:
            # 直接求Q(i+delta)的面积
            base_img = copy.deepcopy(img)
            base_img[base_img < (ave_color-(i+change_delta))] = 0
            base_img[base_img > (ave_color+(i+change_delta))] = 0
            base_img[base_img > 0] = 255
            label_image = measure.label(base_img, connectivity = 2)
            label = label_image[centerY, centerX]
            max_area = np.where(label_image == label)[0].shape[0]

        q_ratio.append((max_area - min_area) / base_area)
        if q_ratio[-1] < 0:
            q_ratio[-1] = 1.0
        min_area = base_area
        base_area = max_area

    idx = np.argmin(q_ratio)
    base_img = copy.deepcopy(img)
    base_img[base_img < (ave_color-(idx+start))] = 0
    base_img[base_img > (ave_color+(idx+start))] = 0
    base_img[base_img > 0] = 255
    label_image = measure.label(base_img, connectivity = 2)
    label = label_image[centerY, centerX]
    label_image[label_image != label] = 0
    label_image[label_image == label] = 1
    label_image = label_image.astype(np.uint8)
    #cv2.erode(label_image, np.ones([3, 3]), label_image, iterations=2)
    #cv2.dilate(label_image, np.ones([3, 3]), label_image, iterations=5)
    mser_region = label_image

    return mser_region

def get_MSER_region(img, centerY, centerX, start, end, step, iterations=1):
    ave_color = img[centerY, centerX]
    for iter in range(iterations):
        mser_region = find_mser_region(img, centerY, centerX, start, end, step, ave_color)
        ave_color = int(np.sum(img * mser_region) / np.sum(mser_region) + 0.5)
        plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    mser_region[mser_region == 1] = 255
    #plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    cv2.erode(mser_region, np.ones([3, 3]), mser_region, iterations=5)
    #plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    label_image = measure.label(mser_region, connectivity = 2)
    label = label_image[centerY, centerX]
    label_image[label_image != label] = 0
    label_image[label_image == label] = 255
    mser_region = label_image.astype(np.uint8)
    cv2.dilate(mser_region, np.ones([3, 3]), mser_region, iterations=5)

    return mser_region

def rotate_image(image, degree):
    (h, w) = image.shape[:2]
    center = (0.5 * w, 0.5 * h)
    # 计算二维旋转的仿射变换矩阵  
    M = cv2.getRotationMatrix2D(center, degree, 1)  
    # 变换图像，并用黑色填充其余值  
    rotated = cv2.warpAffine(image, M, (w, h))  

    return rotated

def get_finger_width(hist_width):
    hist_width[hist_width <= 0.2 * np.max(hist_width)] = 0
    hist_width[hist_width > 0] = 1
    hist_width = list(hist_width)
    flag = False
    finger_width = 0
    tmp_width = 0
    start_idx = 0
    end_idx = 0
    for i in range(len(hist_width)):
        if hist_width[i] == 1:
            if flag == False:
                flag = True
            tmp_width = tmp_width + 1

        if hist_width[i] == 0 and flag == True:
            if finger_width < tmp_width:
                finger_width = tmp_width
                end_idx = i-1
                start_idx = end_idx - finger_width + 1
            tmp_width = 0
            flag = False
    if flag == True:
        finger_width = tmp_width
        end_idx = len(hist_width)-1
        start_idx = end_idx - finger_width + 1

    return finger_width, start_idx, end_idx

def get_region(img, centerY, centerX, threshold, iterations=5):
    key_area1 = 0
    key_area2 = 0
    key_ratio = []
    flag = False
    color = img[centerY, centerX]
    for i in range(iterations):
        huituimg1 = copy.deepcopy(gray_finger_img)
        huituimg1[huituimg1 < (color-threshold)] = 0
        huituimg1[huituimg1 > (color+threshold)] = 0
        huituimg1[huituimg1 > 0] = 255
        label_image1 = measure.label(huituimg1, connectivity = 2)
        label1 = label_image1[centerY, centerX]
        label_image1[label_image1 != label1] = 0
        label_image1[label_image1 == label1] = 255
        label_image1 = label_image1.astype(np.uint8)
        plt.imshow(label_image1), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.erode(label_image1, np.ones([3, 3]), label_image1)
        cv2.dilate(label_image1, np.ones([3, 3]), label_image1)
        label_image1[label_image1 == 255] = 1
        color = int(np.sum(img * label_image1) / np.sum(label_image1))
        plt.imshow(label_image1), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        plt.show() 

    mser_region = label_image1

    return mser_region


#temp = 'W:/FaceDBTrain/ImmigrationDataBase/000014.json'
#with open(temp, 'r') as json_file:
#    facial_encodings = json.load(json_file)
#    json_file.close()

#name_list = []
#for key in facial_encodings:
#    if ('000002\\' in key) or ('000010\\' in key) or ('000017\\' in key):
#        name_list.append(key)

#hzx = len(name_list)

#for key in name_list:
#    del facial_encodings[key]

#with open(temp, 'w') as json_file:
#    json.dump(facial_encodings, json_file)
#    json_file.close()

#hzx = 'W:/FaceDBTrain/ImmigrationDataBase/000000/36167/36167_B.jpg'
#jj = cv2.imread(hzx)
#plt.imshow(jj), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
#plt.show() 
#cv2.imwrite(hzx, jj)
################################################################################
def HandChange(imgFile, maskFile, outFile):
    #imgFile = 'I:/OtherProject/HandChange/Mask.png'
    finger_mask = cv2.imread(maskFile)[:,:,0]
    finger_mask_center_x = 51
    finger_mask_center_y = 60
    finger_mask_width = 51
    finger_mask_height = 101
    
    # load an original image and normalize
    #imgFile = 'I:/OtherProject/HandChange/test2.jpg'
    img = cv2.imread(imgFile)
    scale_ratio = 780 / min(img.shape[0], img.shape[1])
    img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio)
    
    ################################################################################
    
    print('YCbCr Skin Model')
    rows,cols,channels = img.shape
    ################################################################################
    # light compensation
    
    gamma = 0.95
    img = np.floor(img ** gamma).astype(np.uint8)
    
    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # convert color space from bgr to rgb                        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    huituimg = copy.deepcopy(img)
    imgSkinA = SkinColorExtract.extract_skin_color_YCbCr(imgYcc)
    #imgSkinB = SkinColorExtract.extract_skin_color_RGB(img)
    imgSkin = np.reshape(imgSkinA, [rows, cols])
    #plt.imshow(imgSkin), plt.title('Hand area of the first search'), plt.xticks([]), plt.yticks([])
    #plt.show()
    # 膨胀，去除一些小的空洞
    cv2.dilate(imgSkin, np.ones([3, 3]), imgSkin, iterations=5)
    # cv2.erode(imgSkin, np.ones([3, 3]), imgSkin, iterations=5)
    
    # 寻找最大的连通域作为手的区域
    hand_contour, max_area, imgSkin = find_max_contours(imgSkin)
    if hand_contour is None:
        print(u'手部区域与背景太接近或者图像质量太差')
    # 获取手部区域
    #plt.imshow(imgSkin), plt.title('Hand area of the first search'), plt.xticks([]), plt.yticks([])
    #plt.show()  
    
    # 利用canny算子进行边缘检测
    contour_image = detect_contour(imgYcc[:,:,0], imgSkin, isCvtColor=False, isForCut=True)
    #plt.imshow(contour_image), plt.title('Hand contour by canny'), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    imgSkin[(imgSkin == 1)] = 255
    imgSkin = imgSkin - contour_image
    imgSkin[(imgSkin < 255)] = 0
    #plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    
    hand_contour, max_area, imgSkin = find_max_contours(imgSkin, isSecondFind=True)
    if hand_contour is None:
        print(u'手部区域与背景太接近或者图像质量太差')
    else:
        #plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        #plt.show() 
        cv2.medianBlur(imgSkin, 5, imgSkin)
        #plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        #plt.show() 
        cv2.dilate(imgSkin, np.ones([5, 5]), imgSkin)
        #plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        #plt.show() 
        hand_contour, max_area, imgSkin = find_max_contours(imgSkin)
        moments = cv2.moments(hand_contour, True)
        center_hand = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        cv2.circle(huituimg, tuple(center_hand), 5, (255, 0, 0), -1)
        
        dists = []
        min_dist = 100000
        start_idx = 0
        for i, pnt in enumerate(hand_contour):
            pnt = np.reshape(pnt, -1)
            # 靠近边缘的点排除，该步骤可能应当放到后面？
            if (pnt[0] < 0.05 * cols or pnt[0] > 0.95 * cols
                or pnt[1] < 0.05 * rows or pnt[1] > 0.95 * rows):
                dists.append(0)
                continue
            # 将所有距离点加入一维数组
            dist = np.sqrt(np.dot((pnt-center_hand),(pnt-center_hand)))
            dists.append(dist)
            if min_dist > dist:
                min_dist = dist
                start_idx = i
        
        dists = dists[start_idx:] + dists[0:start_idx]
        hand_contour = np.concatenate((hand_contour[start_idx:],hand_contour[0:start_idx]))
        #plt.plot(range(len(dists)), dists)
        #plt.show()
        
        ave_dist = np.average(dists)
        finger_width = ave_dist / 7
        
        # 进行高斯滑动平均
        # 这一段究竟需要不需要？
        # 需要，而且必须进行平滑，最好根据指头的平均宽度平滑
        # 此处给了一个经验值
        kernel = np.exp(-1 * (np.arange(int(0.5*finger_width)*2+1)-int(0.5*finger_width))**2 / (finger_width))
        kernel = kernel / np.sum(kernel)
        dists = np.convolve(dists, kernel, mode='same')
        #dists = dists[50:len(hand_contour)+50]
        #plt.plot(range(len(dists)), dists)
        #plt.show()
        
        kernel = np.array([1, -1])
        grad = np.sign(np.convolve(dists, kernel, mode='same'))
        grad = np.convolve(grad, kernel, mode='same')
        idx = np.where(grad == -2)
        
        # 开始处理指尖点
        prop_pnt = hand_contour[idx]
        prop_dist = np.array(dists)[idx]
        idx = np.argsort(prop_dist)
        prop_pnt = prop_pnt[idx]
        prop_dist = prop_dist[idx]
        max_dist = np.max(prop_dist)
        finger_pnt = []
        dist_pnt = []
        for pnt, dist in zip(prop_pnt, prop_dist):
            pnt = np.reshape(pnt, -1)
            if pnt[1] > center_hand[1]:
                continue
            if dist < 0.5 * max_dist:
                continue
            finger_pnt.append(pnt)
            dist_pnt.append(dist)
        
        # 接下来精细的筛选
        if len(dist_pnt) > 4:
            ave_dist = np.average(dist_pnt)
            del_id = []
            for i, pnt in enumerate(finger_pnt):
                if pnt[1] > center_hand[1]-0.5*ave_dist:
                    del_id.append(i)
                    continue
    
            for i, id in enumerate(del_id):
                del finger_pnt[id-i]
                del dist_pnt[id-i]
    
        if len(dist_pnt) > 4:
            imgSkin[imgSkin < 255] = 0
            imgSkin[imgSkin == 255] = 1
            minh = np.min(np.where(imgSkin==1)[0])
            minw = np.min(np.where(imgSkin==1)[1])
            maxh = np.max(np.where(imgSkin==1)[0])
            maxw = np.max(np.where(imgSkin==1)[1])
            new_img = (img[minh:maxh+1, minw:maxw+1, :] * np.expand_dims(imgSkin[minh:maxh+1, minw:maxw+1], axis=2))
            #plt.imshow(img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
            #plt.show() 
            color_ave = (np.sum(np.sum(new_img, axis=0), axis=0) / np.sum(imgSkin)).astype(np.float32)
            center_color = new_img[center_hand[1]-minh, center_hand[0]-minw, :].astype(np.float32)
            color_ave = 0.7 * color_ave + 0.3 * center_color
            new_img = np.sqrt(np.sum(np.square(new_img-color_ave), axis=2)).astype(np.float32)
            new_img = new_img * imgSkin[minh:maxh+1, minw:maxw+1]
            del_id = []
            for i, pnt in enumerate(finger_pnt):
                
                zz = new_img[max(0, pnt[1]-minh):max(0, pnt[1]-minh)+20, 
                             max(0, pnt[0]-minw-10):max(0, pnt[0]-minw)+10]
    
                zeros_num = len(np.where(zz == 0)[0])
                zz[zz < 70] = 0
                zz[zz >= 70] = 1
                prop = 1 - np.sum(zz) / (np.size(zz) - zeros_num)
                
                if prop < 0.5:
                    del_id.append(i)
    
            for i, id in enumerate(del_id):
                del finger_pnt[id-i]
                del dist_pnt[id-i]
    
        #finger_pnt.append([496,592])
        for pnt in finger_pnt:
            cv2.circle(huituimg, tuple(pnt), 5, (0, 255, 0), -1)
            cv2.line(huituimg, tuple(center_hand), tuple(pnt), (0, 0, 255), 2) 
    
        #plt.imshow(huituimg), plt.title('finger point'), plt.xticks([]), plt.yticks([])
        #plt.show() 
    
    
        # 至此，4个指尖点已经全部检测出来，接下来要开始形状匹配啦
        # 提取指尖点附近领域
        dist_ave = np.average(dist_pnt)
        H_radius = max(36, int(0.2 * dist_ave)) #此处0.1是一个经验的阈值
        W_radius = max(36, int(0.12 * dist_ave))
        for pnt in finger_pnt[::-1]:
            finger_img = img[max(0, pnt[1]-int(0.5*H_radius)):min(rows-1, pnt[1]+int(1.5*H_radius)), 
                             max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
                             :]
            gray_finger_img = imgYcc[max(0, pnt[1]-int(0.5*H_radius)):min(rows-1, pnt[1]+int(1.5*H_radius)), 
                                     max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
                                     0]
            finger_skin = imgSkin[max(0, pnt[1]-int(0.5*H_radius)):min(rows-1, pnt[1]+int(1.5*H_radius)), 
                                  max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius)]
            finger_img = (finger_img * np.expand_dims(finger_skin/255, axis=2)).astype(np.uint8)
            #gray_finger_img = (gray_finger_img * (finger_skin/255)).astype(np.uint8)
            #mser_finger_img = imgHsv[max(0, pnt[1]-int(0.7*H_radius)):min(rows-1, pnt[1]+int(1.3*H_radius)), 
            #                         max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
            #                         1]
            plt.imshow(gray_finger_img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
            plt.show() 
            # 裁剪图像后，指尖点的位置：
            crop_pnt = np.array((int(0.5*H_radius),W_radius))
            #cv2.circle(finger_img, (crop_pnt[1], crop_pnt[0]+int(0.5*W_radius)), 5, (0, 0, 255))
            #plt.imshow(finger_img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
            #plt.show()
            # 抽取边缘太low了，从现在开始，要尝试一下MSER+HOG
            # 抽取HOG特征
            hand_contour = detect_contour(gray_finger_img, finger_skin, isForCut=False)
            plt.imshow(hand_contour), plt.title('finger image'), plt.xticks([]), plt.yticks([])
            plt.show() 
            gx = cv2.Sobel(hand_contour, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(hand_contour, cv2.CV_32F, 0, 1, ksize=1)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            #mag = mag[crop_pnt[1]-36:crop_pnt[1]+36, crop_pnt[0]-36:crop_pnt[0]+36]
            #angle = angle[crop_pnt[1]-36:crop_pnt[1]+36, crop_pnt[0]-36:crop_pnt[0]+36]
            #plt.imshow(mag), plt.title('finger image'), plt.xticks([]), plt.yticks([])
            #plt.show()
         
            # 全局统计梯度直方图
            grad_bins = np.zeros([181])
            for i in range(gray_finger_img.shape[0]):
                for j in range(gray_finger_img.shape[1]):
                    if mag[i,j] == 0:
                        continue
                    if angle[i,j] >= 180:
                        angle[i,j] = angle[i,j] - 180
                    bin_id = int(angle[i,j])
                    if abs(bin_id - angle[i,j]) < 0.0001:
                        grad_bins[(bin_id % 180)] = grad_bins[bin_id] + mag[i,j]
                    else:
                        grad_bins[((bin_id+1) % 180)] = (
                            grad_bins[((bin_id+1) % 180)] + mag[i,j] * (angle[i,j] - bin_id))
                        grad_bins[bin_id] = (
                            grad_bins[bin_id] + mag[i,j] * (1 - angle[i,j] + bin_id))
            
            # 统计梯度直方图
            grad_bins = grad_bins / np.max(grad_bins)
            idx = np.where(grad_bins > 0.1 * np.max(grad_bins))[0]
            grad_ave = 0
            for index_bin in list(idx):
                grad_angle = index_bin
                if index_bin >= 90:
                    grad_angle = 90 - grad_angle
                grad_ave = grad_ave + grad_bins[index_bin] * grad_angle
            grad_ave = grad_ave / np.sum(grad_bins[idx])
            rotated = rotate_image((gray_finger_img * (finger_skin)).astype(np.uint8), grad_ave)
            plt.imshow(rotated), plt.title('finger image'), plt.xticks([]), plt.yticks([])
            plt.show()
            # 抽取MSER区域
            keyY = crop_pnt[0]+int(0.5*W_radius)
            keyX = crop_pnt[1]
            #hzx = imgSkin[keyY, keyX]
            #finger_color = gray_finger_img[keyY, keyX]
            mser_region = get_MSER_region(rotated[0:int(1.5*H_radius), :], keyY, keyX, 1, 40, 1, 2)
            mser_region[mser_region == 255] = 1
            #plt.imshow(rotated[0:int(1.5*H_radius), :] * mser_region), 
            #plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
            #plt.show()
            # 计算指甲宽度
            hist_width = np.sum(mser_region, axis=0)
            finger_width, start_idx, end_idx = get_finger_width(hist_width)
            center_x = int(0.5 + 0.5 * (start_idx + end_idx))
            center_y = np.where(mser_region[:, center_x] > 0)[0][0]
            # 接下来对模板图像进行变换然后贴图
            scale_width = int(finger_width / finger_mask_width * finger_mask.shape[1] * 0.85)
            scale_height = int(finger_width / finger_mask_height * finger_mask.shape[0])
            scale_center_x = int(finger_width / finger_mask_width * finger_mask_center_x * 0.85)
            scale_center_y = int(finger_width / finger_mask_height * finger_mask_center_y)
            resize_mask = cv2.resize(finger_mask, (scale_width, scale_height))
            idx = np.array(np.where(resize_mask == 255))
            shift_x = center_x - scale_center_x
            shift_y = center_y - scale_center_y
            idx[0, :] = idx[0, :] + shift_y
            idx[1, :] = idx[1, :] + shift_x
    
            rotate_mask = np.zeros((rotated.shape[0],rotated.shape[1]), dtype=np.uint8)
            for jj in range(len(idx[0, :])):
                if idx[0, jj] < 0 or idx[0, jj] >= rotated.shape[0]:
                    continue
                if idx[1, jj] < 0 or idx[1, jj] >= rotated.shape[1]:
                    continue
                rotate_mask[idx[0, jj], idx[1, jj]] = 255
            rotate_mask = rotate_image(rotate_mask, -grad_ave)
            idx = np.array(np.where(rotate_mask == 255))
            idx[0, :] = idx[0, :] + max(0, pnt[1]-int(0.5*H_radius))
            idx[1, :] = idx[1, :] + max(0, pnt[0]-W_radius)
            img[idx[0, :], idx[1, :], 0] = 255
            img[idx[0, :], idx[1, :], 1] = 0
            img[idx[0, :], idx[1, :], 2] = 0
            #plt.imshow(img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
            #plt.show()

    cv2.imwrite(outFile, img)


if __name__ == '__main__':
    """ Entry point """
    HandChange('I:/OtherProject/HandChange/test2.jpg', 
               'I:/OtherProject/HandChange/Mask.png',
               'I:/OtherProject/HandChange/test2_mask.jpg')