# Script that splits the data in 4 folds random folds
import glob, os, shutil
import numpy as np

# scan image files
data_dir = 'data'
image_extension = '.png'
images_list = glob.glob(data_dir + '/*' + image_extension)

# fold X
fold_num = 'fold5'
#train, val, test, train2 = np.split(images_list, [int(len(images_list)*0.2), int(len(images_list)*0.3), int(len(images_list)*0.4)])
#train = list(train) + list(train2)
val, test, train = np.split(images_list, [int(len(images_list)*0.1), int(len(images_list)*0.2)])
print("TRAIN split # images = %d" % len(train))
print("VAL split # images = %d" % len(val))
print("TEST split # images = %d" % len(test))


def split_sort_dataset(data_split, split_name, fold_num):
    os.makedirs(os.path.join('data_folds\\' + fold_num, split_name), exist_ok=True)
    for image_filepath in data_split:
        shutil.copy2(image_filepath, os.path.join('data_folds\\' + fold_num, split_name))

split_sort_dataset(train, 'train', fold_num)
split_sort_dataset(val, 'val', fold_num)
split_sort_dataset(test, 'test', fold_num)





