import numpy as np
from utils import Circle

def p_of_normal_distribution(center_x, center_y, sigma, img_height, img_width):

    x = np.arange(img_width).reshape((1, img_width)).repeat(img_height, axis=0)
    y = np.arange(img_height).reshape((img_height, 1)).repeat(img_width, axis=1)

    distance_square = np.square(x - center_x) + np.square(y - center_y)
    p = np.exp(-0.5 * distance_square / (sigma * sigma))

    return p

def p_of_multiple_normal_distributions(circles : list, img_height, img_width):

    sum_of_p = np.zeros((img_height, img_width))
    for circle in circles:
        if not circle.valid:
            continue
        p = p_of_normal_distribution(circle.center_x, circle.center_y, circle.anno_radius, img_height, img_width)
        sum_of_p = sum_of_p + (1.0 - sum_of_p) * p

    sum_of_p = np.clip(sum_of_p, 0.0, 1.0)

    return sum_of_p

def avg_pool_2d(p, reduce_filter):
    h = p.shape[0]
    w = p.shape[1]
    h_remain = h % reduce_filter
    w_remain = w % reduce_filter
    left_h_remain = h_remain // 2; right_h_remain = h_remain - left_h_remain
    left_w_remain = w_remain // 2; right_w_remain = w_remain - left_w_remain

    p = p[left_h_remain : h - right_h_remain, left_w_remain : w - right_w_remain]

    p = np.reshape(p, (p.shape[0] // reduce_filter, reduce_filter, p.shape[1] // reduce_filter, reduce_filter))
    p_avg_pool = p.mean(axis=(1, 3))

    return p_avg_pool


#p = p_of_normal_distribution(5, 5, 11, 11, 5)
#print(p)