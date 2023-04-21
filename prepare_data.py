import argparse
import tqdm
import numpy as np
from numpy.lib.format import open_memmap
import glob
from torchvision.transforms import Resize, FiveCrop
from torchvision.transforms.functional import rotate
from torchvision.io import read_image
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
        os.makedirs(args.saveloc)
        print('Created Missing Directory {}'.format(args.saveloc))

    print(args.saveloc)
    train_imgs, train_masks = prepare_dataset(list_of_images, list_of_masks, 'training', args)
    print('---> Training Image and Masks Generated with shape {}'.format(train_imgs.shape))

    test_imgs, test_masks = prepare_dataset(list_of_images, list_of_masks, 'testing', args)
    print('---> Testing Image and Masks Generated with shape {}'.format(test_imgs.shape))


def prepare_dataset(images, masks, mode, args):
    split_ratio, rots, resize_size, cropsize, saveloc = args.train_val_split, args.rotations, args.ResizeSize, args.CropSize, args.saveloc

    resize_function = Resize(resize_size)
    five_crop_function = FiveCrop(size=(cropsize, cropsize))
    train_ind_lim = int(len(images) * split_ratio)

    img_file = open_memmap(os.path.join(saveloc, f'resize_five_crop_images_{mode}.npy'), mode='w',
                           dtype=float, shape=(len(images) * 5, 3, resize_size, resize_size))
    mask_file = open_memmap(os.path.join(saveloc, f'resize_five_crop_masks_{mode}.npy'), mode='w',
                            dtype=float, shape=(len(images) * 5, 1, resize_size, resize_size))

    if mode == 'training':
        consider_images = images[:train_ind_lim]
        consider_masks = masks[:train_ind_lim]
    elif mode == 'testing':
        consider_images = images[train_ind_lim:]
        consider_masks = masks[train_ind_lim:]

    for i, (IMAGE, MASK) in enumerate(tqdm.tqdm(zip(consider_images, consider_masks))):
        IMAGE_ARRAY = []
        MASK_ARRAY = []
        IM_data = read_image(IMAGE) / 255.
        MASK_data = read_image(MASK) / 255.
        MASK_data[np.where(MASK_data > 0)] = 1

        resize_image = resize_function(IM_data)
        resize_mask = resize_function(MASK_data.unsqueeze(0))
        im_crops = five_crop_function(resize_image)
        mask_crops = five_crop_function(resize_mask)
        for each_im_crop, each_mask_crop in zip(im_crops, mask_crops):
            IMAGE_ARRAY.append(each_im_crop.numpy())
            MASK_ARRAY.append(each_mask_crop.numpy())
        for AUGS, angle in zip(range(len(rots)), rots):
            for icrop, mcrop in zip(im_crops, mask_crops):
                auged_image = rotate(icrop, angle)
                auged_mask = rotate(mcrop, angle)
                IMAGE_ARRAY.append(auged_image.numpy())
                MASK_ARRAY.append(auged_mask.numpy())

        img_file[i * 5: (i + 1) * 5, :] = IMAGE_ARRAY
        mask_file[i * 5: (i + 1) * 5, :] = MASK_ARRAY


if __name__ == '__main__':
    main()
