import numpy as np
from patchgan.unet import UNet
from patchgan.utils import get_norm_layer
from patchgan.disc import Discriminator
from patchgan.io import MmapDataGenerator
from patchgan.trainer import Trainer, device
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', '-i', required=True)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gen_filts', type=int, default=32)
    parser.add_argument('--disc_filts', type=int, default=16)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--transfer_weights', default=None)
    args = parser.parse_args()

    mmap_imgs = f'{args.inputDir}/resize_five_crop_images_train.npy'
    mmap_mask = f'{args.inputDir}/resize_five_crop_masks_train.npy'
    batch_size = args.batchsize
    traindata = MmapDataGenerator(mmap_imgs, mmap_mask, batch_size)

    mmap_imgs_val = f'{args.inputDir}/resize_five_crop_images_test.npy'
    mmap_mask_val = f'{args.inputDir}/resize_five_crop_images_test.npy'
    batch_size = args.batchsize
    val_dl = MmapDataGenerator(mmap_imgs_val, mmap_mask_val, batch_size)

    GEN_FILTS = args.gen_filts
    DISC_FILTS = args.disc_filts
    ACTIV = 'relu'

    IN_NC = args.in_channels
    OUT_NC = args.out_channels

    norm_layer = get_norm_layer()

    # create the generator
    generator = UNet(IN_NC, OUT_NC, GEN_FILTS, norm_layer=norm_layer,
                     use_dropout=False, activation=ACTIV).to(device)

    # create the discriminator
    discriminator = Discriminator(IN_NC + OUT_NC, DISC_FILTS, n_layers=3, norm_layer=norm_layer).to(device)

    # create the training object and start training
    trainer = Trainer(generator, discriminator,
                      f'checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/')

    if args.transfer_weights is not None:
        generator.load_transfer_data(args.transfer_weights)

    G_loss, D_loss = trainer.train(traindata, val_dl, 200, gen_learning_rate=5.e-4,
                                   dsc_learning_rate=1.e-4, lr_decay=0.95)

    # save the loss history
    np.savez(f'checkpoints-{GEN_FILTS}-{DISC_FILTS}-{ACTIV}/loss_history.npz', D_loss=D_loss, G_loss=G_loss)


if __name__ == '__main__':
    main()
