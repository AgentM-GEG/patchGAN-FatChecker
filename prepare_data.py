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

    assert len(list_of_images) == len(list_of_masks),\
        f"Number of images ({len(list_of_masks)}) does not match number of masks ({len(list_of_masks)})!"

    np.random.seed(args.randomseed)
    inds = np.arange(len(list_of_images)).astype(int)
    np.random.shuffle(inds)
    train_ind_lim = int(len(list_of_images) * args.train_val_split)

    print(f"Found {len(list_of_images)} images. Splitting to {train_ind_lim} for training and {len(list_of_images) - train_ind_lim} for testing.")

    list_of_images = np.array(list_of_images)[inds]
    list_of_masks = np.array(list_of_masks)[inds]

    if not os.path.exists(args.saveloc):
        os.makedirs(args.saveloc)
        print('Created Missing Directory {}'.format(args.saveloc))

    prepare_dataset(list_of_images[:train_ind_lim], list_of_masks[:train_ind_lim], 'train', args)
    prepare_dataset(list_of_images[train_ind_lim:], list_of_masks[train_ind_lim:], 'test', args)


def prepare_dataset(images, masks, mode, args):
    rots, resize_size, crop_size, saveloc = args.rotations, args.ResizeSize, args.CropSize, args.saveloc

    resize_function = Resize(resize_size)
    five_crop_function = FiveCrop(size=(crop_size, crop_size))

    nbatch = 5 * (len(rots) + 1)

    img_file = open_memmap(os.path.join(saveloc, f'resize_five_crop_images_{mode}.npy'), mode='w+',
                           dtype=np.uint8, shape=(len(images) * nbatch, 3, crop_size, crop_size))
    mask_file = open_memmap(os.path.join(saveloc, f'resize_five_crop_masks_{mode}.npy'), mode='w+',
                            dtype=np.uint8, shape=(len(images) * nbatch, 1, crop_size, crop_size))

    for i, (IMAGE, MASK) in enumerate(tqdm.tqdm(zip(images, masks),
                                                ascii=True,
                                                desc=f'Saving {mode} data',
                                                total=len(images))):
        IMAGE_ARRAY = []
        MASK_ARRAY = []
        IM_data = read_image(IMAGE)  # save the image as uint8 to save space
        MASK_data = read_image(MASK) / 255.  # normalize the mask to 0-1
        MASK_data[np.where(MASK_data > 0)] = 1

        resize_image = resize_function(IM_data)
        resize_mask = resize_function(MASK_data)
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

        img_file[i * nbatch: (i + 1) * nbatch, :] = np.asarray(IMAGE_ARRAY)
        mask_file[i * nbatch: (i + 1) * nbatch, :] = np.asarray(MASK_ARRAY)


if __name__ == '__main__':
    main()
