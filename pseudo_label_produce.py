import os
import cv2
import numpy as np
import pickle

current_road = os.getcwd()
save_road = os.path.join(current_road, 'pseudo')
if not os.path.exists(save_road):
    os.mkdir(save_road)
dataset_dir = 'datasets/MF' # 选择不同的数据集生成相应的 pseudo label
list_name = 'list.txt' # 训练数据的list
# dataset_dir = 'datasets/PST900'
# list_name = 'train.txt'
list_road = os.path.join(current_road, dataset_dir, list_name)
def openfile(listname):
    path = os.path.join(current_road, dataset_dir)
    list_read = open(listname).readlines()
    print("Totally {} samples in {}.".format(len(list_read), list_name))
    # list_read = ['01419D', '01451D', '01236N', '01442D', '01443D', '01447D', '01448D', '01445D', '01441D']
    # list_read = ['57_bag20_rect_rgb_frame0000000069', '57_bag20_rect_rgb_frame0000000093','57_bag20_rect_rgb_frame0000000115', \
    #     '57_bag20_rect_rgb_frame0000000145']
    list_read = ['01189N', '01232N', '01243N', '01332N', '01424D', '01335N', '01336N', '01376N', \
        '01425D', '01413D', '01414D', '01418D', '01423D', '01425D', '01534D', '01533D', '01531D', \
            '01525D', '01519D', '01516D', '01501D', '01496D']
    scores = {}
    if list_name == 'list.txt':
        for line in list_read:
            image_name = line.strip()
            # image roads
            image_path = os.path.join(current_road, dataset_dir, 'images/' + image_name + '.png')
            gt_path = os.path.join(current_road, dataset_dir, 'labels/' + image_name + '.png')
            rgb_path = os.path.join(current_road, dataset_dir, 'seperated_images/' + image_name + '_rgb.png')
            thermal_path =  os.path.join(current_road, dataset_dir, 'seperated_images/' + image_name + '_th.png')
            label_path = os.path.join(current_road, dataset_dir, 'visual/' + image_name + '.jpg')

            # read the image
            rgb = cv2.imread(rgb_path)
            inf = cv2.imread(thermal_path, 0)
            gt = cv2.imread(gt_path, 0)
            label = cv2.imread(label_path)
            rgb_iou, inf_iou, iou_rate = visualize(image_name, rgb, inf, gt, label, vis=True)
            scores[image_name] = {"rgb_iou": rgb_iou, "inf_iou": inf_iou, "iou_rate": iou_rate}
    elif list_name == 'train.txt':
        for line in list_read:
            image_name = line.strip()
            flip = False
            if  '_flip' in image_name:
                image_name = image_name.split('_')[0]
                flip = True
            if 'MF' in dataset_dir:
                gt_path = os.path.join(current_road, dataset_dir, 'labels/' + image_name + '.png')
                rgb_path = os.path.join(current_road, dataset_dir, 'seperated_images/' + image_name + '_rgb.png')
                thermal_path =  os.path.join(current_road, dataset_dir, 'seperated_images/' + image_name + '_th.png')
                label_path = os.path.join(current_road, dataset_dir, 'visual/' + image_name + '.jpg')
            else:
                gt_path = os.path.join(current_road, dataset_dir, 'train/labels/' + image_name + '.png')
                rgb_path = os.path.join(current_road, dataset_dir, 'train/rgb/' + image_name + '.png')
                thermal_path =  os.path.join(current_road, dataset_dir, 'train/thermal/' + image_name + '.png')
                label_path = os.path.join(current_road, dataset_dir, 'train/labels/' + image_name + '.png')

            # read the image
            rgb = cv2.imread(rgb_path)
            inf = cv2.imread(thermal_path, 0)
            gt = cv2.imread(gt_path, 0)
            label = cv2.imread(label_path)
            rgb_iou, inf_iou, iou_rate = visualize(image_name, rgb, inf, gt, label, vis=False)
            if flip:
                image_name += '_flip'
            scores[image_name] = {"rgb_iou": rgb_iou, "inf_iou": inf_iou, "iou_rate": iou_rate}
                
    ### 下面注释的代码表示将会将 pseudo label 保存成 score.pkl。
    # with open(os.path.join(path, "score.pkl"), "wb") as fout:
    #     pickle.dump(scores, fout)
    # with open(os.path.join(path, "score.pkl"), "rb") as fin:
    #     njud_data = pickle.load(fin)
    # print(type(njud_data))
    # for k, v in njud_data.items():
    #     if 'flip' not in k:
    #         print(k, '\'s rgb IoU: {:.3f}, thm IoU: {:.3f}, rgb_ratio: {:.3f}, thm_ratio: {:.3f}'\
    #             .format(v['rgb_iou'], v['inf_iou'], v['iou_rate'], 1-v['iou_rate']))
    # print("Done!")


## 可视化 pseudo label 生成的过程。
def visualize(image_name, rgb, inf, gt, label, vis=False):     
    # convert the image to gray
    # rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_ = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # rgb_gray = cv2.normalize(rgb_gray, rgb_gray, 0, 255, cv2.NORM_MINMAX)
    # inf_gray = cv2.cvtColor(inf, cv2.COLOR_BGR2GRAY)
    # inf = cv2.normalize(inf, inf, 0, 255, cv2.NORM_MINMAX)

    # Initialise the more fine-grained saliency detector and compute the saliencyMap
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(rgb_)
    rgb_saliencyMap = (saliencyMap * 255).astype("uint8")
    
    # saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(inf)
    inf_saliencyMap = (saliencyMap * 255).astype("uint8")

    # using OTSU algorithm to convert to binary image
    # _, rgb_binary = cv2.threshold(rgb_saliencyMap, 127, 255, cv2.THRESH_BINARY)
    rgb_ret, rgb_ostu = cv2.threshold(rgb_saliencyMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, rgb_blur_ostu = cv2.threshold(cv2.GaussianBlur(rgb_gray, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('rgb ret: ', rgb_ret)
    # _, inf_binary = cv2.threshold(inf_saliencyMap, 105, 255, cv2.THRESH_BINARY)
    inf_ret, inf_ostu= cv2.threshold(inf_saliencyMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, inf_blur_otsu = cv2.threshold(cv2.GaussianBlur(inf, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('infrared ret: ', inf_ret)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    # kernel = np.ones((5, 5), np.uint8) # Creating kernel
    # rgb_blur_ostu = cv2.dilate(rgb_blur_ostu, kernel, iterations = 2)
    # rgb_blur_ostu = cv2.erode(rgb_blur_ostu, None, iterations = 1)

    # find the rgb contours of gray
    # contours, _ = cv2.findContours(rgb_blur_ostu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(rgb_blur_ostu, contours, -1, (255), cv2.FILLED)

    # find the contours of gt
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("the number of contours: ", len(contours))

    # visual the contours for each type image
    gt[gt>0] = 255
    gt_rect = gt.copy()
    label_rect = label.copy()
    cv2.putText(label_rect, text='number of contours: ' + str(len(contours)), org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
        fontScale=1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    
    rgb_rect = np.zeros_like(rgb_ostu)  # the contours of gt contains parts
    inf_rect = np.zeros_like(inf_ostu) # the contour of gt contains parts
    rgb_overlap = np.zeros_like(gt)
    inf_overlap = np.zeros_like(gt)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(gt_rect, (x, y), (x+w, y+h), (60), 3)
        cv2.rectangle(label_rect, (x, y), (x+w, y+h), (0, 0, 255), 3)
        rgb_rect[y: y+h, x: x+w] = rgb_ostu[y: y+h, x: x+w]  # axis x and y are changed in OpenCV
        inf_rect[y: y+h, x: x+w] =  inf_ostu[y: y+h, x: x+w]
        rgb_overlap[y: y+h, x: x+w] = gt[y: y+h, x: x+w] + rgb_ostu[y: y+h, x: x+w]
        inf_overlap[y: y+h, x: x+w] = gt[y: y+h, x: x+w] + inf_ostu[y: y+h, x: x+w]
    rgb_overlap[rgb_overlap > 255] = 255
    inf_overlap[inf_overlap > 255] = 255

    rgb_iou, inf_iou, iou_rate = cal_score(image_name, gt, rgb_rect, inf_rect)

    if vis:  # Whether visualize the compared image
        rgb_saliencyMap = cv2.merge((rgb_saliencyMap, rgb_saliencyMap, rgb_saliencyMap))
        # rgb_binary = cv2.merge((rgb_binary, rgb_binary, rgb_binary))
        rgb_ostu = color_change(rgb_ostu, 0, 75)
        rgb_ostu = cv2.merge((rgb_ostu, rgb_ostu, rgb_ostu))
        # rgb_blur_ostu = cv2.merge((rgb_blur_ostu, rgb_blur_ostu, rgb_blur_ostu))
        rgb_rect = cv2.merge((rgb_rect, rgb_rect, rgb_rect))
        rgb_overlap = cv2.merge((rgb_overlap, rgb_overlap, rgb_overlap))
        cv2.putText(rgb_overlap, text='rgb iou: %.2f, rate: %.2f'%(rgb_iou, iou_rate), org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
            fontScale=1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        horizontal_rgb = cv2.hconcat([rgb, rgb_saliencyMap, rgb_ostu, rgb_rect, rgb_overlap])

        inf = cv2.merge((inf, inf, inf))
        inf_saliencyMap = cv2.merge((inf_saliencyMap, inf_saliencyMap, inf_saliencyMap))
        # inf_binary = cv2.merge((inf_binary, inf_binary, inf_binary))
        inf_ostu = color_change(inf_ostu, 0, 75)
        inf_ostu = cv2.merge((inf_ostu, inf_ostu, inf_ostu))
        # inf_blur_otsu = cv2.merge((inf_blur_otsu, inf_blur_otsu, inf_blur_otsu))
        inf_rect = cv2.merge((inf_rect, inf_rect, inf_rect))
        inf_overlap = cv2.merge((inf_overlap, inf_overlap, inf_overlap))
        cv2.putText(inf_overlap, text='inf iou: %.2f, rate: %.2f'%(inf_iou, 1-iou_rate), org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
            fontScale=1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        horizontal_inf = cv2.hconcat([inf, inf_saliencyMap, inf_ostu, inf_rect, inf_overlap])

        gt_rect = cv2.merge((gt_rect, gt_rect, gt_rect))
        gt = color_change(gt, 0, 75)
        gt = cv2.merge((gt, gt, gt))
        horizontal_label = cv2.hconcat([label, gt_rect, label_rect, gt, gt_rect])
        
        whole = cv2.vconcat([horizontal_rgb, horizontal_inf, horizontal_label])
        
        save_path = os.path.join(save_road, image_name)
        cv2.imwrite(save_path + '_binary.png', whole)
        cv2.imwrite(save_path + '_rgb_smap.png', rgb_saliencyMap)
        cv2.imwrite(save_path + '_th_smap.png', inf_saliencyMap)
        cv2.imwrite(save_path + '_rgb_ostu.png', rgb_ostu)
        cv2.imwrite(save_path + '_th_ostu.png', inf_ostu)
        cv2.imwrite(save_path + '_gt.png', gt)
        
    return rgb_iou, inf_iou, iou_rate
    

def cal_score(image_name, gt, rgb_rect, inf_rect, beta=0.3):
    gt_rect = gt.copy()
    gt_rect[gt_rect>0] = 255 # visual the gt
    gt_rect = np.float32(gt_rect)
    rgb_rect = np.float32(rgb_rect)
    inf_rect = np.float32(inf_rect)
    gt_rect *= 1/255.0
    rgb_rect *= 1/255.0
    inf_rect *= 1/255.0
    gt_rect[gt_rect >= 0.5] = 1.
    gt_rect[gt_rect < 0.5] = 0.
    rgb_rect[rgb_rect >= 0.5] = 1.
    rgb_rect[rgb_rect < 0.5] = 0.
    inf_rect[inf_rect >= 0.5] = 1.
    inf_rect[inf_rect < 0.5] = 0.

    rgb_over = (rgb_rect * gt_rect).sum()
    rgb_union = ((rgb_rect + gt_rect) >= 1).sum()
    inf_over = (inf_rect * gt_rect).sum()
    inf_union = ((inf_rect + gt_rect) >= 1).sum()
    sum_gt = gt_rect.sum()

    rgb_iou = rgb_over / (1e-7 + rgb_union)
    inf_iou = inf_over / (1e-7 + inf_union)
    rgb_cover = rgb_over / (1e-7 + sum_gt)
    inf_cover = inf_over / (1e-7 + sum_gt)
    iou_rate = rgb_iou / (1e-7 + rgb_iou + inf_iou)
    # if 'D' in image_name: # if image in day-time, use the inf iou to calculate the corresponding rate
    #     iou_rate = 1 - inf_iou # iou_rate is for rgb
    #     # iou_rate = 1 - ((1.0+beta) * inf_iou * inf_cover / (1e-7 + inf_iou + beta * inf_cover))
    # else: # else if image in night-time, use the rgb iou
    #     iou_rate = rgb_iou # iou_rate is for rgb
    #     # iou_rate = (1.0+beta) * rgb_iou * rgb_cover / (1e-7 + rgb_iou + beta * rgb_cover)
    return rgb_iou, inf_iou, iou_rate


def color_change(one_channel_image, original_value=0, modified_value=10):
    for i in range(one_channel_image.shape[0]):
        for j in range(one_channel_image.shape[1]):
            if one_channel_image[i,j] == (original_value):
                one_channel_image[i,j] = (modified_value)
    return one_channel_image

openfile(list_road)
