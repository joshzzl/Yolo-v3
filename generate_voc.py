
from pycocotools.coco import COCO
import shutil
import random
import os
import wget


_ANNO_PATH = './coco/annotations/'
_ANNO_FILE = 'instances_train2014.json'

_TRAIN_CAT_SELECT = 60
_TEST_CAT_SELECT = 20

_WRITE_FOLDER = './data/dataset/'

_TRAIN_IMAGE_WRITE = './data/images/train/'
_TEST_IMAGE_WRITE = './data/images/test/'

_TRAIN_FILENAME = 'train.txt'
_TEST_FILENAME = 'test.txt'

_COCO_NAMES = "./data/classes/coco.names"


def create_voc_file(voc_write_path, file_name, img_write_path, 
                    img_select_per_cat):
    with open(voc_write_path+file_name, 'w') as f:
        coco = COCO(_ANNO_PATH+_ANNO_FILE)
        cats = coco.loadCats(coco.getCatIds())
        cat_names = [cat['name'] for cat in cats]

        name_book = {}
        for ID, name in enumerate(cat_names):
            name_book[name] = ID

        print('\nTrain categories length: {}\n'.format(len(cat_names)))

        for name in cat_names:
            
            catIds = coco.getCatIds(catNms=[name])
            imgIds = coco.getImgIds(catIds=catIds)
            imgIds_selects = random.sample(imgIds, img_select_per_cat)
            
            img_selects = coco.loadImgs(imgIds_selects)
            
            for img in img_selects:
                img_path = img_write_path+img['file_name']
                wget.download(img['coco_url'], img_path)
                ann_ids = coco.getAnnIds(imgIds=img['id'])
                anns = coco.loadAnns(ann_ids)

                annotation_str = []
                annotation_str.append(img_path)

                for ann in anns:
                    temp_ar = []
                    ann_cat_id = ann['category_id']
                    cat_name = coco.loadCats(ids=ann_cat_id)[0]['name']
                    x_topleft   = int(ann['bbox'][0])
                    y_topleft   = int(ann['bbox'][1])
                    bbox_width  = int(ann['bbox'][2])
                    bbox_height = int(ann['bbox'][3])
                    x_bottomright = x_topleft + bbox_width
                    y_bottomright = y_topleft + bbox_height
                    temp_ar.append(x_topleft)
                    temp_ar.append(y_topleft)
                    temp_ar.append(x_bottomright)
                    temp_ar.append(y_bottomright)
                    temp_ar.append(name_book[cat_name])
                    temp_ar = [str(s) for s in temp_ar]
                    annotation_str.append(','.join(temp_ar))

                f.write(' '.join(annotation_str) + '\n')

        f.close()

'''
    Main
'''
if __name__=='__main__':
    
    create_voc_file(_WRITE_FOLDER, _TRAIN_FILENAME, _TRAIN_IMAGE_WRITE, _TRAIN_CAT_SELECT)
    create_voc_file(_WRITE_FOLDER, _TEST_FILENAME, _TEST_IMAGE_WRITE, _TEST_CAT_SELECT)
