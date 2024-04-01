import cv2
import sys
import os
import glob
import math
#import utils
import numpy as np
import argparse
import ntpath
import pickle

from utils import get_img_anno_pairs, Circle
from anno_processor import p_of_multiple_normal_distributions, avg_pool_2d


def parse_args():
    parser = argparse.ArgumentParser(description='Process some data collector args')
    parser.add_argument('--img_dir', type=str, default='./',
                        help='')
    parser.add_argument('--start_point', type=int, default=0,
                        help='')
    parser.add_argument('--anno_radius', type=int, default=64,
                        help='')
    args = parser.parse_args()
    return args


def mouse_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        '''
        # print('Left Button Down!')
        x1 = max(0, x - 20)
        y1 = max(0, y - 20)

        x2 = x1 + 40
        y2 = y1 + 40

        if x2 > param[2] or y2 > param[3]:
            return
        '''

        #cv2.rectangle(param[4], (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.circle(param[4], (x, y), param[6], color=(0, 255, 0), thickness=2)
        # param[1].append([x - 20, y - 20, x + 20, y + 20])
        param[1].append(Circle(center_x=x, center_y=y, anno_radius=param[6], valid=True))
        param[0] = True

    # Cancel the annotation
    if event == cv2.EVENT_RBUTTONDOWN:

        # print('Right Button Down!')
        for circle in param[1]:
            if circle.valid == True:
                dx = circle.center_x - x
                dy = circle.center_y - y
                if math.sqrt(dx * dx + dy * dy) < circle.anno_radius:
                    circle.valid = False
                    param[0] = True

        if param[0] == True:
            param[4][...] = param[5][...]
            for circle in param[1]:
                if circle.valid == True:
                    cv2.circle(param[4], (circle.center_x, circle.center_y), circle.anno_radius, (0,255,0), 2)


def write_anno(anno_path, w, h, c, circles, anno_radius, show_maps=False):

    text = ''
    for circle in circles:
        if circle.valid == False:
            continue
        text = text + str(circle.center_x) + ' ' + str(circle.center_y) + ' ' + str(circle.anno_radius) + '\n'

    with open(anno_path, 'w') as f:
        f.write(text)

    # Write anno of probability format
    p = p_of_multiple_normal_distributions(circles, h, w)
    p_avg_pool = avg_pool_2d(p, anno_radius * 2)
    
    with open(anno_path + '.pkl', 'wb') as f:
        pickle.dump(p_avg_pool, f)

    if show_maps:
        show_anno_maps(p, p_avg_pool, anno_radius)

    print('Writing annotation done!')


def show_anno_maps(p, p_avg_pool, anno_radius, scale=0.2):

    if p_avg_pool is not None:
        w = int(p_avg_pool.shape[1] * anno_radius * 2 * scale)
        h = int(p_avg_pool.shape[0] * anno_radius * 2 * scale)
        p_avg_pool_to_show = cv2.resize(p_avg_pool, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow('probability map avg pool', p_avg_pool_to_show)
    else:
        cv2.imshow('probability map avg pool', np.zeros((10, 10)))

    if p is not None:
        w = int(p.shape[1] * scale)
        h = int(p.shape[0] * scale)
        p_to_show = cv2.resize(p, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('probability map', p_to_show)
    else:
        cv2.imshow('probability map', np.zeros((10, 10)))


def read_anno(anno_path, circles, anno_radius, show_maps=False):

    lines = []
    with open(anno_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        text_items = line.strip().split()
        circles.append(
            Circle(
                center_x=int(text_items[0]),
                center_y=int(text_items[1]),
                anno_radius=int(text_items[2]),
                valid=True))
        
    if os.path.exists(anno_path + '.pkl'):
        with open(anno_path + '.pkl', 'rb') as f:
            p_avg_pool = pickle.load(f)
            if show_maps:
                show_anno_maps(None, p_avg_pool, anno_radius)
    else:
        if show_maps:
            show_anno_maps(None, None, anno_radius)

    print('Reading annotation done!')


def view_images_and_update_anno(img_dir, start_point, anno_radius):

    # Read image list
    img_list = glob.glob(os.path.join(img_dir, '**/*.png'), recursive=True)
    img_list.sort()

    # Load existing annotation files
    img_anno_dict = dict()
    img_anno_list = get_img_anno_pairs(img_dir)
    for (img_path, anno_path) in img_anno_list:
        img_anno_dict[img_path] = anno_path

    cv2.namedWindow("image")

    position = start_point % len(img_list)
    while True:
        img_path = img_list [position]
        img = cv2.imread(img_path)
        img_copy = np.copy(img)
       
        print()
        print(position + 1, 'of', len(img_list))
        print(img_path)
        print(img.shape)

        mouse_param = [False, [], img.shape[1], img.shape[0], img, img_copy, anno_radius]
        #############    0     1        2            3          4     5        6

        circles = mouse_param[1]

        if img_path in img_anno_dict:
            anno_path = img_anno_dict[img_path]
            read_anno(anno_path, circles, anno_radius, True)
        else:
            # If there is no corresponding anno file for this image, create it
            anno_path = img_path.replace('.png', '.txt')
            img_anno_dict[img_path] = anno_path
            write_anno(anno_path, img.shape[1], img.shape[0], img.shape[2], circles, anno_radius)
            read_anno(anno_path, circles, anno_radius, True)
        
        for circle in circles:
            if circle.valid == True:
                cv2.circle(img, (circle.center_x, circle.center_y), circle.anno_radius, (0,255,0), 2)

             
        cv2.setMouseCallback("image", mouse_event, mouse_param)

        while True:
            # print('Refresh image!!!')
            cv2.imshow('image', img)
            #key = cv2.waitKey(1) & 0xFF
            key = cv2.waitKeyEx(10)

 
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img[...] = img_copy[...]
                circles.clear()
                mouse_param[0] = True
        
            elif key in {ord('q'), ord('Q')} or key == 27:
                sys.exit()
            
            elif key in {ord('a'), ord('A')} or key == 2424832 or key == 2490368:
                if mouse_param[0]:
                    write_anno(anno_path, img.shape[1], img.shape[0], img.shape[2], circles, anno_radius)
                position -= 1
                position %= len(img_list)
                break
            
            elif key in {ord('d'), ord('D')} or key == 2555904 or key == 2621440:
                if mouse_param[0]:
                    write_anno(anno_path, img.shape[1], img.shape[0], img.shape[2], circles, anno_radius)
                position += 1
                position %= len(img_list)
                break

            elif key in {ord('p'), ord('P')}:
                write_anno(anno_path, img.shape[1], img.shape[0], img.shape[2], circles, anno_radius, True)
                mouse_param[0] = False

            elif key != -1:
                    print('unkown key', key)

# Command example:
#python ./edit_anno.py --img_dir ./images_collected_0 --start_point 0

# Build exe example:
#pyinstaller --onefile ./edit_anno.py

if __name__ == "__main__":

    args = parse_args()

    view_images_and_update_anno(args.img_dir, args.start_point, args.anno_radius)