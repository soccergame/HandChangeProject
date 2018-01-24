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

#def _point_distance(face_encodings, face_to_compare):
#    """
#    计算特征之间的余弦距离
#    输入:
#    [face_encodings] 人脸特征序列
#    [face_to_compare] 待比较的人脸特征
#    输出: 
#    一个numpy向量，包含待比较的人脸与序列中所有人脸的余弦距离
#    """
#    if len(face_encodings) == 0:
#        return np.empty((0))
#
#    corr = face_encodings - face_to_compare
#    corr = corr * corr
#    corr = np.sqrt(np.sum(corr, axis=1))
#    #corr = np.sum(face_encodings*face_to_compare,axis=1)
#    ##face_encodings_norm = np.sum(np.square(face_encodings), axis = 1)
#    ##face_to_compare_norm = np.sum(np.square(face_to_compare))
#    ##corr = corr / np.sqrt(face_encodings_norm * face_to_compare_norm + 0.0000001);
#    #corr = (1.0 + corr) / 2.0
#
#    # 需要用欧氏距离
#
#    #print(corr)
#    return corr
#
#def _chinese_whispers(encodings, 
#                      threshold=5, 
#                      iterations=20):
#    """ Chinese Whispers Algorithm
#    
#    Modified from Alex Loveless' implementation,
#    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
#    
#    Inputs:
#        encoding_list: a list of facial encodings from face_recognition
#        threshold: facial match threshold,default 0.6
#        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
#    
#    Outputs:
#        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
#            sorted by largest cluster to smallest
#    """
#    
#    #from face_recognition.api import _face_distance
#    from random import shuffle
#    import networkx as nx
#    # Create graph
#    nodes = []
#    edges = []
#    
#    if len(encodings) <= 1:
#        print (u'没有足够的人脸加入节点!')
#        return []
#    
#    print(u'计算特征两两之间的距离')
#    for idx, face_encoding_to_check in enumerate(encodings):
#        # 加入一个节点
#        node_id = idx+1
#    
#        # 用一个节点初始化一个聚类（作为聚类中心）
#        node = (node_id, {'cluster': idx, 'path': idx})
#        nodes.append(node)
#    
#        # Facial encodings to compare
#        if (idx+1) >= len(encodings):
#            # 若是最后一个节点，则不建立边缘
#            break
#    
#        # 每个人脸都计算到该聚类中心的距离
#        # print(image_paths[idx])
#        compare_encodings = encodings[idx+1:]
#        distances = _point_distance(compare_encodings, face_encoding_to_check)
#        # print(distances)
#        encoding_edges = []
#        for i, distance in enumerate(distances):
#            if distance < threshold:
#                # 将人脸与该中心的距离作为权重加入以该点为中心的类
#                edge_id = idx+i+2
#                encoding_edges.append((node_id, edge_id, {'weight': distance}))
#    
#        edges = edges + encoding_edges
#    
#    print(u'开始图匹配')
#    G = nx.Graph()
#    G.add_nodes_from(nodes)
#    G.add_edges_from(edges)
#    
#    # 循环迭代
#    for _ in range(0, iterations):
#        cluster_nodes = G.nodes()
#        shuffle(cluster_nodes)
#        for node in cluster_nodes:
#            neighbors = G[node]
#            clusters = {}
#    
#            # 考察中心点每个相邻节点
#            for ne in neighbors:
#                if isinstance(ne, int):
#                    # 将该相邻节点所属类别加入中心点邻域
#                    if G.node[ne]['cluster'] in clusters:
#                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
#                    else:
#                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']
#    
#            # 找到邻域中距离该中心点最近的相邻节点（类别）
#            edge_weight_sum = 0
#            max_cluster = G.node[node]['cluster'] 
#            for cluster in clusters:
#                if clusters[cluster] > edge_weight_sum:
#                    edge_weight_sum = clusters[cluster]
#                    max_cluster = cluster
#    
#            # 认为该中心点与距离最近的相邻节点构成一个类
#            # 并以该相邻节点的名称作为类别名称
#            G.node[node]['cluster'] = max_cluster
#    
#    clusters = {}
#    
#    # 准备聚类输出
#    print(u'准备聚类输出')
#    for (_, data) in G.node.items():
#        cluster = data['cluster']
#        path = data['path']
#    
#        if cluster:
#            if cluster not in clusters:
#                clusters[cluster] = []
#            clusters[cluster].append(path)
#    
#    # 按照类别包含点的数目，排列聚类输出
#    print(u'按照类别包含点的数目，排列聚类输出')
#    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
#    
#    return sorted_clusters

def find_max_contours(image, isSecondFind=False):
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_area = image.shape[0] * image.shape[1]
    max_area = 0
    hand_contour = None
    for cnt in contours:
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

        imgSkin = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(imgSkin, [cnt], 0, 255, -1)
        if imgSkin[int(0.5*image.shape[0]), int(0.5*image.shape[1])] != 255:
            continue

        if max_area < area:
            max_area = area
            hand_contour = cnt

    if hand_contour is not None:
        imgSkin = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(imgSkin, [hand_contour], 0, 255, -1)
        imgSkin[(imgSkin < 255)] = 0
    else:
        imgSkin = None

    return hand_contour, max_area, imgSkin

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
    base_img[base_img < (ave_color-idx)] = 0
    base_img[base_img > (ave_color+idx)] = 0
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
        #plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
        #plt.show()
    
    mser_region[mser_region == 1] = 255
    cv2.erode(mser_region, np.ones([3, 3]), mser_region, iterations=2)
    cv2.dilate(mser_region, np.ones([3, 3]), mser_region, iterations=2)
    label_image = measure.label(mser_region, connectivity = 2)
    label = label_image[centerY, centerX]
    label_image[label_image != label] = 0
    label_image[label_image == label] = 255
    mser_region = label_image.astype(np.uint8)
    #plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

        #huituimg2 = copy.deepcopy(gray_finger_img)
        #huituimg2[huituimg2 < (color-i-1)] = 0
        #huituimg2[huituimg2 > (color+i+1)] = 0
        #huituimg2[huituimg2 > 0] = 255
        #label_image2 = measure.label(huituimg2, connectivity = 2)
        #label2 = label_image2[centerY, centerX]
        #key_area2 = np.where(label_image2 == label2)[0].shape[0]
        #key_ratio.append(key_area1 / key_area2)
        #key_area1 = key_area2
        #label_image1 = label_image2
        #label1 = label2
        #
        #if key_ratio[-1] > threshold:
        #    flag = True
        #
        #if flag:
        #    if key_ratio[-1] < threshold:
        #        label_image1[label_image1 != label1] = 0
        #        label_image1[label_image1 == label1] = 255
        #        label_image1 = label_image1.astype(np.uint8)
        #        cv2.erode(label_image1, np.ones([3, 3]), label_image1, iterations=2)
        #        cv2.dilate(label_image1, np.ones([3, 3]), label_image1, iterations=5)
        #        mser_region = label_image1
        #        break

    #if mser_region is None:
    #    label_image1[label_image1 != label1] = 0
    #    label_image1[label_image1 == label1] = 255
    #    label_image1 = label_image1.astype(np.uint8)
    #    cv2.erode(label_image1, np.ones([3, 3]), label_image1, iterations=2)
    #    cv2.dilate(label_image1, np.ones([3, 3]), label_image1, iterations=5)
    #    mser_region = label_image1

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
    hist_width[hist_width <= 0.5 * np.max(hist_width)] = 0
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

    #cv2.erode(label_image1, np.ones([3, 3]), label_image1, iterations=2)
    #cv2.dilate(label_image1, np.ones([3, 3]), label_image1, iterations=5)
    mser_region = label_image1
    #plt.imshow(mser_region), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 

    return mser_region

################################################################################
imgFile = 'I:/OtherProject/HandChange/Mask.png'
finger_mask = cv2.imread(imgFile)[:,:,0]
finger_mask_center_x = 51
finger_mask_center_y = 70
finger_mask_width = 51
finger_mask_height = 101
#mask_img = mask_img[:, 75-55:75+55, :]
#cv2.imwrite('I:/OtherProject/HandChange/finger_template1.png', mask_img)

# load an original image and normalize
imgFile = 'I:/OtherProject/HandChange/test2.jpg'
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
#imgYcc[:,:,0] = cv2.equalizeHist(imgYcc[:,:,0])
#plt.imshow(imgHsv[:,:,1]), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
#plt.show()  

# convert color space from bgr to rgb                        
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
huituimg = copy.deepcopy(img)
imgSkinA = SkinColorExtract.extract_skin_color_YCbCr(imgYcc)
#imgSkinB = SkinColorExtract.extract_skin_color_RGB(img)
imgSkin = np.reshape(imgSkinA, [rows, cols])
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
#plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
#plt.show() 

#cv2.erode(imgSkin, np.ones([3, 3]), imgSkin)
#imgSkin[(imgSkin < 255)] = 0  
#
## 第二遍寻找
## 寻找最大的连通域作为手的区域
#_, contours, hierarchy = cv2.findContours(imgSkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#max_area = 0
#hand_contour = None
#for cnt in contours:
#    area = cv2.contourArea(cnt)
#    if max_area < area:
#        max_area = area
#        hand_contour = cnt
#
## 获取手部区域
#imgSkin = np.zeros_like(imgSkin, dtype=np.uint8)
#cv2.drawContours(imgSkin, [hand_contour], 0, 255, -1)
#imgSkin = imgSkin - hand_img
#cv2.erode(imgSkin, np.ones([3, 3]), imgSkin)
#imgSkin[(imgSkin < 255)] = 0
#
#_, contours, hierarchy = cv2.findContours(imgSkin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#max_area = 0
#hand_contour = None
#for cnt in contours:
#    area = cv2.contourArea(cnt)
#    if max_area < area:
#        max_area = area
#        hand_contour = cnt

#hand_img = img * np.expand_dims(imgSkin, axis=2)
#hand_img_hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)[:,:,1]
#ave_pixel = (np.sum(hand_img_hsv) / np.sum(imgSkin)).astype(np.uint8)
#hand_img_hsv[(hand_img_hsv < ave_pixel-15)] = 0
#hand_img_hsv[(hand_img_hsv > ave_pixel+15)] = 0
#hand_img_hsv[(hand_img_hsv > 110)] = 0
#mser = cv2.MSER_create(_delta = 1, _area_threshold = 0.8)
#mser = cv2.MSER_create(_delta = 5, _max_area = int(1.2 * max_area))
#regions, boxes = mser.detectRegions(hand_img_hsv)
#max_area = 0
#hand_region = None
#for region in regions:
#    area = np.shape(region)[0]
#    if max_area < area:
#        max_area = area
#        hand_region = region

#for box in boxes:
#    x, y, w, h = box
#    cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 2)

##imgSkin[hand_region[:,::-1]] = 255
##imgSkin[(imgSkin < 127.5)] = 0
##imgSkin[(imgSkin > 127.5)] = 1
#plt.imshow(hand_img_hsv), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
#plt.show()   

#for pnt in hand_contour:
#    cv2.circle(img, tuple(np.reshape(pnt, -1)), 5, (255, 0, 0), -1)

#plt.imshow(img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
#plt.show()    

# 计算重心
#total_point_num = np.sum(imgSkin)
#idx = np.where(imgSkin == 1)
#center_hand = (int(np.sum(idx[1])/total_point_num+0.5), int(np.sum(idx[0])/total_point_num+0.5))
else:
    cv2.dilate(imgSkin, np.ones([5, 5]), imgSkin)
    #plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    hand_contour, max_area, imgSkin = find_max_contours(imgSkin)
    moments = cv2.moments(hand_contour, True)
    center_hand = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
    cv2.circle(huituimg, tuple(center_hand), 5, (255, 0, 0), -1)
    
    #plt.imshow(img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    # 重新计算边缘
    #cv2.dilate(imgSkin, np.ones([5, 5]), imgSkin)
    #hand_img = detect_contour(np.ones_like(img)*255, imgSkin)
    #plt.imshow(hand_img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    #hand_contour = np.array(np.where(hand_img > 0)).transpose()[:,::-1]
    
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

    for pnt in finger_pnt:
        cv2.circle(huituimg, tuple(pnt), 5, (0, 255, 0), -1)
        cv2.line(huituimg, tuple(center_hand), tuple(pnt), (0, 0, 255), 2) 

    plt.imshow(huituimg), plt.title('finger point'), plt.xticks([]), plt.yticks([])
    plt.show() 


    # 至此，4个指尖点已经全部检测出来，接下来要开始形状匹配啦
    # 提取指尖点附近领域
    dist_ave = np.average(dist_pnt)
    H_radius = max(36, int(0.2 * dist_ave)) #此处0.1是一个经验的阈值
    W_radius = max(36, int(0.1 * dist_ave))
    for pnt in finger_pnt[::-1]:
        finger_img = img[max(0, pnt[1]-H_radius):min(rows-1, pnt[1]+H_radius), 
                         max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
                         :]
        gray_finger_img = imgYcc[max(0, pnt[1]-H_radius):min(rows-1, pnt[1]+H_radius), 
                                 max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
                                 0]
        finger_skin = imgSkin[max(0, pnt[1]-H_radius):min(rows-1, pnt[1]+H_radius), 
                              max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius)]
        finger_img = (finger_img * np.expand_dims(finger_skin/255, axis=2)).astype(np.uint8)
        gray_finger_img = (gray_finger_img * (finger_skin/255)).astype(np.uint8)
        #mser_finger_img = imgHsv[max(0, pnt[1]-int(0.7*H_radius)):min(rows-1, pnt[1]+int(1.3*H_radius)), 
        #                         max(0, pnt[0]-W_radius):min(cols-1, pnt[0]+W_radius),
        #                         1]
        plt.imshow(gray_finger_img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        plt.show() 
        # 裁剪图像后，指尖点的位置：
        crop_pnt = np.array((H_radius,W_radius))
        #cv2.circle(finger_img, (crop_pnt[1], crop_pnt[0]+int(0.5*W_radius)), 5, (0, 0, 255))
        #plt.imshow(finger_img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        #plt.show()
        # 抽取边缘太low了，从现在开始，要尝试一下MSER+HOG
        # 抽取HOG特征
        hand_contour = detect_contour(gray_finger_img, isForCut=False)
        plt.imshow(hand_contour), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        plt.show() 
        gx = cv2.Sobel(hand_contour, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(hand_contour, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #mag = mag[crop_pnt[1]-36:crop_pnt[1]+36, crop_pnt[0]-36:crop_pnt[0]+36]
        #angle = angle[crop_pnt[1]-36:crop_pnt[1]+36, crop_pnt[0]-36:crop_pnt[0]+36]
        plt.imshow(mag), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        plt.show()
        # 分块统计梯度直方图
        #grad_bins = np.zeros([gray_finger_img.shape[0] * gray_finger_img.shape[1], 37])
        #for i in range(gray_finger_img.shape[0]):
        #    for j in range(gray_finger_img.shape[1]):
        #        block_id = j + (i * gray_finger_img.shape[1])
        #        bin_id = int(angle[i,j] / 10)
        #        if mag[i,j] == 0:
        #            grad_bins[block_id, 36] = 255
        #            continue
        #        if abs(bin_id * 10 - angle[i,j]) < 0.0001:
        #            grad_bins[block_id, (bin_id % 36)] = grad_bins[block_id, bin_id] + mag[i,j]
        #        else:
        #            grad_bins[block_id, ((bin_id+1) % 36)] = (
        #                grad_bins[block_id, ((bin_id+1) % 36)] + mag[i,j] * (angle[i,j] - bin_id * 10) / 10)
        #            grad_bins[block_id, bin_id] = (
        #                grad_bins[block_id, bin_id] + mag[i,j] * (10 - angle[i,j] + bin_id * 10) / 10)
        
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
        #grad_bins = np.reshape(grad_bins, [gray_finger_img.shape[0], gray_finger_img.shape[1], 37])
        #small_bins = np.argmax(grad_bins, axis=2)
        #small_bins[small_bins == 0] = -18
        #small_bins[small_bins > 32] = small_bins[small_bins > 32] + 18
        #small_bins[small_bins > 23] = small_bins[small_bins > 23] - 18
        #valid_grad = len(np.where(small_bins < 36)[0])
        #small_bins[small_bins >= 36] = 0
        grad_bins = grad_bins / np.max(grad_bins)
        #grad_bins = grad_bins / np.sum(grad_bins)
        idx = np.where(grad_bins > 0.5 * np.max(grad_bins))[0]
        grad_ave = np.sum(grad_bins[idx]*idx) / np.sum(grad_bins[idx])
        #dir_pnt = (int(crop_pnt[0]-20*np.cos(grad_ave/180*np.pi)), int(crop_pnt[1]+20*np.sin(grad_ave/180*np.pi)))
        #cv2.line(gray_finger_img, (crop_pnt[1], crop_pnt[0]), (dir_pnt[1], dir_pnt[0]), (255,0,0),2)
        #plt.imshow(gray_finger_img), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.show()
        rotated = rotate_image(gray_finger_img, grad_ave)
        #plt.imshow(rotated), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.show()
        #grad_bins = np.reshape(grad_bins, [9, 3, 108])
        #grad_bins = np.reshape(np.transpose(grad_bins, (1, 0, 2)), [3, 3, -1])
        #big_bins = np.mod(np.argmax(grad_bins, axis=2), 36)
        #grad_bins = np.reshape(grad_bins, [3, 3, -1, 36])
        #grad_bins = np.sum(grad_bins, axis=2)
        #big_ave_bins = np.argmax(grad_bins, axis=2)
        #angle_ave = np.average(small_bins[3:,:])
        #print(angle_ave)
        # 抽取MSER区域
        keyY = crop_pnt[0]+int(0.5*W_radius)
        keyX = crop_pnt[1]
        hzx = imgSkin[keyY, keyX]
        finger_color = gray_finger_img[keyY, keyX]
        mser_region = get_MSER_region(rotated, keyY, keyX, 1, 40, 1, 2)
        #mser_region = get_region(gray_finger_img, keyY, keyX, 30)
        mser_region[mser_region == 255] = 1
        plt.imshow(rotated * mser_region), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        plt.show()
        # 计算指甲宽度
        hist_width = np.sum(mser_region, axis=0)
        finger_width, start_idx, end_idx = get_finger_width(hist_width)
        center_x = int(0.5 + 0.5 * (start_idx + end_idx))
        center_y = np.where(mser_region[:, center_x] > 0)[0][0]
        #M = cv2.getRotationMatrix2D(
        #    (0.5 * mser_region.shape[1], 0.5 * mser_region.shape[0]), 
        #    -grad_ave, 1)
        #real_center = np.matmul(M, np.array((center_x, center_y, 1))) #获取真正的指尖点
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

        rotate_mask = np.zeros((mser_region.shape[0],mser_region.shape[1]), dtype=np.uint8)
        for jj in range(len(idx[0, :])):
            if idx[0, jj] < 0 or idx[0, jj] >= mser_region.shape[0]:
                continue
            if idx[1, jj] < 0 or idx[1, jj] >= mser_region.shape[1]:
                continue
            rotate_mask[idx[0, jj], idx[1, jj]] = 255
        rotate_mask = rotate_image(rotate_mask, -grad_ave)
        idx = np.array(np.where(rotate_mask == 255))
        idx[0, :] = idx[0, :] + max(0, pnt[1]-H_radius)
        idx[1, :] = idx[1, :] + max(0, pnt[0]-W_radius)
        #plt.imshow(finger_img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        #plt.show()
        #plt.imshow(rotated), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        #plt.show()
        img[idx[0, :], idx[1, :], 0] = 255
        img[idx[0, :], idx[1, :], 1] = 0
        img[idx[0, :], idx[1, :], 2] = 0
        plt.imshow(img), plt.title('MSER connected'), plt.xticks([]), plt.yticks([])
        plt.show()

        hzx = 0

        # 出于计算方便考虑，不使用C-HOG
        #plt.imshow(mag), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.show() 
        #plt.imshow(gy), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.show() 

        # hough圆检测，效果一般
        #finger_img_contour = detect_contour(gray_finger_img, isForCut=False)
        #hough_img = HoughTransform.hough_trans_circles(int(0.3*W_radius), W_radius, finger_img_contour)
        #depth = W_radius - int(0.1*W_radius) + 1
        #hough_img = np.reshape(hough_img, [-1, gray_finger_img.shape[0], gray_finger_img.shape[1]])
        #idx = np.where(hough_img > 0.9 * np.max(hough_img))
        #r = int(np.average(idx[0])+0.5) # 定位出指宽
        #h = int(np.average(idx[1])+0.5)
        #w = int(np.average(idx[2])+0.5)
        #cv2.circle(finger_img, (w, h), r+int(0.3*W_radius), (0, 0, 255))

        # hough线段检测，效果不好
        #max_len = (int)(np.sqrt(2.0) * np.maximum(finger_img.shape[0], finger_img.shape[1]));
        #hough_img = HoughTransform.hough_trans(360, finger_img_contour)
        #hough_img = np.reshape(hough_img, [-1, 360])
        #hough_idx = hough_img
        #hough_idx[hough_idx < 120] = 0
        #cv2.dilate(hough_idx, np.ones([5, 5]), hough_idx)
        #idx = np.where(hough_idx > 0)
        #rho = 0
        #theta = 0
        #value = 0
        #lines = []
        #for loch, locw in zip(idx[0], idx[1]):
        #    rho = rho + loch * hough_idx[loch, locw]
        #    theta = theta + locw * hough_idx[loch, locw]
        #    value = value + hough_idx[loch, locw]
        #rho = rho / value
        #theta = theta / value
        #rho = rho - max_len
        #theta = theta / 360 * np.pi 
        #if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线    
        #    #该直线与第一行的交点    
        #    pt1 = (int(rho/np.cos(theta)),0)    
        #    #该直线与最后一行的焦点    
        #    pt2 = (int((rho-finger_img.shape[0]*np.sin(theta))/np.cos(theta)),finger_img.shape[0])    
        #    #绘制一条白线    
        #    cv2.line(finger_img, pt1, pt2, (255))    
        #else: #水平直线    
        #    # 该直线与第一列的交点    
        #    pt1 = (0,int(rho/np.sin(theta)))    
        #    #该直线与最后一列的交点    
        #    pt2 = (finger_img.shape[1], int((rho-finger_img.shape[1]*np.cos(theta))/np.sin(theta)))    
        #    #绘制一条直线    
        #    cv2.line(finger_img, pt1, pt2, (255), 1)  
        #verts, faces, _, _ = measure.marching_cubes(hough_img, 0)
        #fig = plt.figure(figsize=(18, 6))
        #ax = fig.add_subplot(111, projection='3d')
        #mesh = Poly3DCollection(verts[faces], alpha=0.70)
        #face_color = [0.45, 0.45, 0.75]
        #mesh.set_facecolor(face_color)
        #ax.add_collection3d(mesh)
        #ax.set_xlim(0, hough_img.shape[2])
        #ax.set_ylim(0, hough_img.shape[1])
        #ax.set_zlim(0, hough_img.shape[0])
        #x_star_end = [0, hough_img.shape[2]]
        #y_star_end = [0, hough_img.shape[1]]
        #z_star_end = [0, hough_img.shape[0]]
        #
        #for s, e in combinations(np.array(list(product(x_star_end,y_star_end,z_star_end))), 2):
        #    ax.plot3D(*zip(s,e), color='r')  
        #
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('z')
        #
        #plt.show()
        #plt.imshow(finger_img_contour[int(0.5*finger_img_contour.shape[0]):int(0.75*finger_img_contour.shape[0]), int(0.5*finger_img_contour.shape[1]-int(0.8*W_radius)):int(0.5*finger_img_contour.shape[1]+int(0.8*W_radius))]), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.imshow(finger_img), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #plt.show() 
        #for i in range(depth):
        #    plt.imshow(hough_img[i, :, :]), plt.title('finger image'), plt.xticks([]), plt.yticks([])
        #    plt.show() 
    
    #cv2.dilate(imgSkin, np.ones([5, 5]), imgSkin)
    # imgSkin[(imgSkin > 0)] = 255
    #cv2.erode(hand_img, np.ones([3, 3]), hand_img)
    
    # 利用霍夫变换进行直线检测
    #再次检测边缘
    #hand_img = detect_contour(np.ones_like(img)*255, imgSkin)
    #hand_img[hand_img > 0] = 255
    #imgSkin = imgSkin - hand_img
    
    # 利用canny算子进行边缘检测
    #imgSkin[imgSkin==255] = 1
    #hand_img = img * np.expand_dims(imgSkin, axis=2)
    #hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    #hand_img = cv2.GaussianBlur(hand_img,(3, 3), 0) 
    #hand_img = cv2.Canny(hand_img, 50, 150)
    #cv2.dilate(hand_img, np.ones([3, 3]), hand_img)
    #plt.imshow(hand_img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    #
    #max_len = (int)(np.sqrt(2.0) * np.maximum(hand_img.shape[0], hand_img.shape[1]));
    #hough_img = HoughTransform.hough_trans(360, hand_img)
    #hough_img = np.reshape(hough_img, [-1, 360])
    #hough_idx = hough_img
    #hough_idx[hough_idx < 120] = 0
    #cv2.dilate(hough_idx, np.ones([5, 5]), hough_idx)
    #plt.imshow(hough_idx), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show() 
    #
    #
    ## 寻找霍夫变换后图像上的连通域
    #_, contours, hierarchy = cv2.findContours(hough_idx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ##hough_idx = np.array(np.where(hough_idx >= 120))
    ##hough_idx = np.transpose(hough_idx)
    ##cluster = _chinese_whispers(hough_idx)
    #max_area = 0
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    if max_area < area:
    #        max_area = area
    #
    #lines = []
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    if area < 10 and area < 0.3 * max_area:
    #        continue
    #
    #    hzx = np.zeros_like(hough_img)
    #    cv2.drawContours(hzx, [cnt], 0, 1, -1)
    #    hough_idx = hough_img * hzx
    #    idx = np.where(hough_idx > 0)
    #    rho = 0
    #    theta = 0
    #    value = 0
    #    for loch, locw in zip(idx[0], idx[1]):
    #        rho = rho + loch * hough_idx[loch, locw]
    #        theta = theta + locw * hough_idx[loch, locw]
    #        value = value + hough_idx[loch, locw]
    #    
    #    rho = rho / value
    #    theta = theta / value
    #    lines.append((rho, theta))
    #    #if theta < 180:
    #    #    lines.append((rho, theta))
    
    ##idx = np.where(hough_img >= 128)
    ##plt.imshow(hough_img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    ##plt.show() 
    ##lines = cv2.HoughLines(hand_img, 1, np.pi/180, 200)
    #
    #for (rho, theta) in lines:   
    #    #rho = line[0, 0] #第一个元素是距离rho    
    #    #theta= line[0, 1] #第二个元素是角度theta 
    #    rho = rho - max_len
    #    theta = theta / 360 * np.pi 
    #    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线    
    #        #该直线与第一行的交点    
    #        pt1 = (int(rho/np.cos(theta)),0)    
    #        #该直线与最后一行的焦点    
    #        pt2 = (int((rho-img.shape[0]*np.sin(theta))/np.cos(theta)),img.shape[0])    
    #        #绘制一条白线    
    #        cv2.line( img, pt1, pt2, (255))    
    #    else: #水平直线    
    #        # 该直线与第一列的交点    
    #        pt1 = (0,int(rho/np.sin(theta)))    
    #        #该直线与最后一列的交点    
    #        pt2 = (img.shape[1], int((rho-img.shape[1]*np.cos(theta))/np.sin(theta)))    
    #        #绘制一条直线    
    #        cv2.line(img, pt1, pt2, (255), 1)    
    
    #cv2.imshow('canny demo',hand_img) 
    #if cv2.waitKey(0) == 27:  
    #    cv2.destroyAllWindows() 
    #def CannyThreshold(lowThreshold):  
    #    detected_edges = cv2.GaussianBlur(hand_img,(3,3),0)  
    #    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)  
    #    #dst = cv2.bitwise_and(hand_img, hand_img, mask = detected_edges)  # just add some colours to edges from original image.  
    #    cv2.imshow('canny demo',detected_edges)  
    #  
    #lowThreshold = 0  
    #max_lowThreshold = 100  
    #ratio = 3  
    #kernel_size = 3  
    #  
    #cv2.namedWindow('canny demo')  
    #  
    #cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)  
    #  
    #CannyThreshold(0)  # initialization  
    #if cv2.waitKey(0) == 27:  
    #    cv2.destroyAllWindows() 
    #
    #hand_img = cv2.GaussianBlur(hand_img,(3,3), 0) 
    #hand_img = cv2.Canny(hand_img, 50, 150)
    
    #plt.imshow(hand_img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show()  
    
    ## 按照边缘走向，逐点计算可能的指尖点
    #max_dist = 0
    #max_pnt = None
    #count = 0
    #fingerTips = []
    #for pnt in hand_contour:
    #    pnt = np.reshape(pnt, -1)
    #    dis = np.sqrt(np.dot((pnt-center_hand),(pnt-center_hand)))
    #    if dis > max_dist:
    #        max_dist = dis
    #        max_pnt = pnt
    #
    #    if dis != max_dist:
    #        count += 1
    #        if count > 400:
    #            count = 0 
    #            max_dist = 0
    #
    #            flag = False
    #
    #            # 靠近边缘的点排除
    #            if max_pnt[0] < 0.05 * cols or max_pnt[0] > 0.95 * cols:
    #                continue
    #            if max_pnt[1] < 0.05 * rows or max_pnt[1] > 0.95 * rows:
    #                continue
    #
    #            # 距离重心太近的点被排除
    #            if len(fingerTips) > 0:
    #                for ft in fingerTips:
    #                    if np.dot((max_pnt-center_hand),(max_pnt-center_hand)) < 200:
    #                        flag = True
    #                        break
    #
    #            if flag:
    #                continue 
    #
    #            fingerTips.append(max_pnt)
    #            cv2.circle(img, tuple(max_pnt), 5, (0, 255, 0), -1)
    #            cv2.line(img, tuple(center_hand), tuple(max_pnt), (0, 0, 255), 2) 
    
    
    
    # 依靠曲率搜索，效果不是很好
    #points_num = len(hand_contour)
    #finger_tips = []
    #for i in range(5, points_num-5):
    #    q = np.reshape(hand_contour[i-5], -1)
    #    p = np.reshape(hand_contour[i], -1)
    #    r = np.reshape(hand_contour[i+5], -1)
    #
    #    dot = np.dot((q-p), (r-p))
    #    if dot < 20 and dot > -20:
    #        cross = np.cross((q-p), (r-p))
    #        if cross > 0:
    #            finger_tips.append(p)
    #            cv2.circle(img, tuple(p), 5, (255, 0, 0), 2)
    
    # 获取手部区域
    # cv2.drawContours(imgSkin, [hand_contour], 0, 255, -1)
    # imgSkin[(imgSkin < 127.5)] = 0
    # imgSkin[(imgSkin > 127.5)] = 1
    #cv2.imshow('hzx', imgSkin)
    # display original image and skin image
    #plt.subplot(1,2,1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1,2,2), 
    #plt.imshow(img), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
    #plt.show()                                                
    #################################################################################
    #
    #print('Goodbye!')