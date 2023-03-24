import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
from torchvision.transforms import Resize, FiveCrop
from torchvision.transforms.functional import rotate
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir', '-i', required=True)
    parser.add_argument('--maskDir', '-m', required=True)
    parser.add_argument('--ResizeSize', default=512)
    parser.add_argument('--CropSize', default=256)
    parser.add_argument('--rotations', default=[90, 180, 270])
    parser.add_argument('--randomseed', default=1234)
    parser.add_argument('--train_val_split', default=0.9)
    parser.add_argument('--saveloc', default='./processed_data')
    args = parser.parse_args()
    
    list_of_images = sorted(glob.glob('{}/*.jpg'.format(args.imgDir)))
    list_of_masks = sorted(glob.glob('{}/*.png'.format(args.maskDir)))

    np.random.seed(args.randomseed)
    inds = np.arange(len(list_of_images)).astype(int)
    np.random.shuffle(inds)
    
    
    list_of_images = np.array(list_of_images)[inds]
    list_of_masks = np.array(list_of_masks)[inds]
    
    if not os.path.exists(args.saveloc):
        os.mkdir(args.saveloc)
        print('Created Missing Directory {}'.format(args.saveloc))
    
    print(args.saveloc)
    train_imgs, train_masks = prepare_dataset(list_of_images, list_of_masks, 'training', args)
    np.save('{}/resize_five_crop_images_train.npy'.format(args.saveloc), train_imgs)
    np.save('{}/resize_five_crop_masks_train.npy'.format(args.saveloc), train_masks)
    print('---> Training Image and Masks Generated with shape {}'.format(train_imgs.shape))
    
    test_imgs, test_masks = prepare_dataset(list_of_images, list_of_masks, 'testing', args)
    np.save('{}/resize_five_crop_images_test.npy'.format(args.saveloc), test_imgs)
    np.save('{}/resize_five_crop_masks_test.npy'.format(args.saveloc), test_masks)
    print('---> Testing Image and Masks Generated with shape {}'.format(test_imgs.shape))
    

    
def prepare_dataset(images, masks, mode, args):
    split_ratio, rots, resize_size, cropsize = args.train_val_split, args.rotations, args.ResizeSize, args.CropSize
    
    resize_function = Resize(resize_size)
    five_crop_function = FiveCrop(size=(cropsize,cropsize))
    IMAGE_ARRAY = []
    MASK_ARRAY = []
    train_ind_lim = int(len(images) * split_ratio)
    if mode == 'training':
        consider_images =  images[:train_ind_lim]
        consider_masks = masks[:train_ind_lim]
    elif mode == 'testing':
        consider_images =  images[train_ind_lim:]
        consider_masks = masks[train_ind_lim:]

    for IMAGE, MASK in tqdm.tqdm(zip(consider_images,consider_masks)):
        IM_data = plt.imread(IMAGE)
        MASK_data = plt.imread(MASK)
        MASK_data[np.where(MASK_data>0)] = 1

        resize_image = resize_function(torch.tensor(np.moveaxis(IM_data, 2, 0)))
        resize_mask = resize_function(torch.tensor(np.expand_dims(MASK_data,0)))
        im_crops = five_crop_function(resize_image)
        mask_crops = five_crop_function(resize_mask)
        for each_im_crop, each_mask_crop in zip(im_crops,mask_crops):
            IMAGE_ARRAY.append(each_im_crop.numpy())
            MASK_ARRAY.append(each_mask_crop.numpy())
        for AUGS, angle in zip(range(len(rots)), rots):
            for icrop, mcrop in zip(im_crops,mask_crops):
                auged_image = rotate(icrop, angle)
                auged_mask = rotate(mcrop, angle)
                IMAGE_ARRAY.append(auged_image.numpy())
                MASK_ARRAY.append(auged_mask.numpy())
    
    
    return np.asarray(IMAGE_ARRAY), np.asarray(MASK_ARRAY)

if __name__ == '__main__':
    main()