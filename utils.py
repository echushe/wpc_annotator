import os
import glob
import ntpath


def get_img_anno_pairs(img_dir):
    anno_paths = glob.glob(os.path.join(img_dir, '**/*.txt'), recursive=True)
    # print(img_files)

    img_anno_list = []
    for anno_path in anno_paths:
        
        if anno_path.endswith('.txt'):
            img_path = anno_path.replace('.txt', '.jpg')

            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')
            
            if os.path.exists(img_path):

                #print(img_path)
                #print(anno_path)

                img_anno_list.append((img_path, anno_path))


    return img_anno_list


class Circle:

    def __init__(self, center_x, center_y, anno_radius, valid=True):
        self.center_x = center_x
        self.center_y = center_y
        self.anno_radius = anno_radius
        self.valid = valid