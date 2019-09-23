






# Create image datasets.
# Processes images and saves them in train, val, test splits.
import os

from datetime import datetime as dt, datetime
from datetime import timedelta
import numpy as np
import hickle as hkl
from cv2 import cv2

from kitti_settings import *



# def process_data():
#     splits = {s: [] for s in ['train', 'test', 'val']}
#     splits['val'] = val_recordings
#     splits['test'] = test_recordings
#     not_train = splits['val'] + splits['test']
#     for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
#         c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
#         folders= list(os.walk(c_dir, topdown=False))[-1][-2]
#         splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]
#
#     for split in splits:
#         im_list = []
#         source_list = []  # corresponds to recording that image came from
#         for category, folder in splits[split]:
#             im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, folder[:10], folder, 'image_03/data/')
#             print(list(os.walk(im_dir, topdown=False)))
#
#             try:
#                 files = list(os.walk(im_dir, topdown=False))[-1][-1]
#                 im_list += [im_dir + f for f in sorted(files)]
#                 source_list += [category + '-' + folder] * len(files)
#             except:
#                 print("!!!exception!!!")
#                 continue
#
#         print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
#         X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
#         for i, im_file in enumerate(im_list):
#             im = imread(im_file)
#             X[i] = process_im(im, desired_im_sz)
#
#         hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
#         hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))
#
#
# # resize and crop image
# def process_im(im, desired_sz):
#     target_ds = float(desired_sz[0])/im.shape[0]
#     im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
#     d = int((im.shape[1] - desired_sz[1]) / 2)
#     im = im[:, d:d+desired_sz[1]]
#     return im
#


height=420
width=340
img_comp_num = 2

# desired_im_sz = (int(height/img_comp_num), int(width/img_comp_num))
desired_im_sz = (128, 256)
training_data_path="/media/diyosko7/HDD2.0T/WeatherChallengeData/train"


def get_pixel_values(dt_month):
    file = training_data_path + "/sat/{year}-{month:02}-{day:02}/{year}-{month:02}-{day:02}-{hour:02}-00.fv.png". \
        format(year=dt_month.year, month=dt_month.month, day=dt_month.day, hour=dt_month.hour)
    if os.path.exists(file):
        img = cv2.imread(file, 0)
    else:
        print("img file" + str(file) + " not found.")
        return None
    return cv2.resize(img[40:40+420, 130:130+340],desired_im_sz)



def get_pixel_values_recursively(target_date, hours_diff):
    img_after_hour_diff = get_pixel_values(target_date + timedelta(hours=hours_diff))
    if img_after_hour_diff is None:
        img_after_hour_diff, _ = get_pixel_values_recursively(target_date, hours_diff=hours_diff - 1)
    source_dir = get_source_list(target_date + timedelta(hours=hours_diff))
    return img_after_hour_diff, source_dir


def create_training_data(year):
    date_start = dt(year, 1, 1, 0, 0, 0)
    date_end = dt(year, 12, 31, 23, 0, 0)
    img_list = []
    sourcedir_list = []
    for date_diff in range((date_end - date_start).days):
        for h in range(24):
            target_date = date_start + timedelta(days=date_diff) + timedelta(hours=h)
            img, source_dir = get_pixel_values_recursively(target_date, hours_diff=0)
            img_list.append(img)
            sourcedir_list.append(source_dir)
    return np.asarray(img_list), np.asarray(sourcedir_list)


def get_source_list(dt_obj):
    file = "/sat/{year}-{month:02}-{day:02}". \
        format(year=dt_obj.year, month=dt_obj.month, day=dt_obj.day, hour=dt_obj.hour)
    return file

def split_data_with_shuffle(img_list, source_list, val_ratio):
    num_all  = len(list(img_list))
    # num_train = num_all*(1 - val_ratio)
    num_val   = int(num_all*val_ratio)

    id_all   = np.random.choice(num_all, num_all, replace=False)
    id_val  = id_all[0:num_val]
    id_train = id_all[num_val:num_all]
    print(id_val)
    img_list_val  = img_list[id_val]
    img_list_train = img_list[id_train]
    source_list_val  = source_list[id_val]
    source_list_train = source_list[id_train]
    return img_list_train,source_list_train, img_list_val, source_list_val

def reshape_img_list(img_list):
    X = np.zeros((len(img_list),) + (desired_im_sz[1], desired_im_sz[0]) + (1,), np.uint8)
    print(X[0].shape)
    for i in range(len(img_list)):
        print(img_list[i])
        print(str(img_list[i][:,:,np.newaxis].shape))
        X[i] = img_list[i][:,:,np.newaxis]
    return X




def main():
    is_train = True
    if is_train:
        year = 2017
        img_list, source_list = create_training_data(year)

        # {img, source}_list を train用とvalidation用にランダムに分割する
        img_list_train,source_list_train, img_list_val, source_list_val = split_data_with_shuffle(img_list, source_list, val_ratio=0.2)

        img_list_train = reshape_img_list(img_list_train)
        img_list_val = reshape_img_list(img_list_val)

        hkl.dump(img_list_train, os.path.join(DATA_DIR, 'X_' + "train" + '.hkl'))
        hkl.dump(list(source_list_train), os.path.join(DATA_DIR, 'sources_' + "train" + '.hkl'))

        hkl.dump(img_list_val, os.path.join(DATA_DIR, 'X_' + "val" + '.hkl'))
        hkl.dump(list(source_list_val), os.path.join(DATA_DIR, 'sources_' + "val" + '.hkl'))
    else:
        pass


if __name__ == '__main__':
    main()

