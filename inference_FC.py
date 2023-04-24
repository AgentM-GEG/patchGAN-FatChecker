import numpy as np
import torch
from patchgan.unet import UNet, get_norm_layer
from torchvision.io import read_image
import argparse
import matplotlib.pyplot as plt
import tqdm
import json
import os
from torchvision.transforms import Resize
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir', '-i', required=True)
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--generate_json_annotations', type=bool, default=False)
    parser.add_argument('--gen_filts', type=int, default=32)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--saveloc', default='./predicted_masks')
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()

    if torch.cuda.is_available() & args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    list_of_images = glob.glob(f'{args.imgDir}/*.png')
    list_of_images.extend(glob.glob(f'{args.imgDir}/*.jpg'))
    print(f'Found {len(list_of_images)} images in the folder {args.imgDir}')

    GEN_FILTS = args.gen_filts
    ACTIV = 'relu'

    IN_NC = args.in_channels
    OUT_NC = args.out_channels

    norm_layer = get_norm_layer()

    # create the generator
    generator = UNet(IN_NC, OUT_NC, GEN_FILTS, norm_layer=norm_layer,
                     use_dropout=False, activation=ACTIV).to(device)
    generator.load_state_dict(torch.load(f'{args.checkpoint}', map_location=device))
    print('Loaded the generator checkpoint successfully')

    generator.eval()

    resize_function = Resize(args.resize_size)
    if args.generate_json_annotations:
        json_data = []
    with torch.no_grad():
        for each_image in tqdm.tqdm(list_of_images):
            image_data = read_image(each_image)
            resized_image = torch.unsqueeze(resize_function(image_data), 0).to(device)
            predicted_mask = generator(resized_image)
            predicted_mask_array = torch.squeeze(torch.squeeze(predicted_mask, 0), 0).numpy()
            save_filename, _ = os.path.splitext(os.path.basename(each_image))
            np.save(f'{args.saveloc}/{save_filename}.npy', predicted_mask_array)

            if args.generate_json_annotations:
                anno = generate_annotations(predicted_mask_array)
                dati = {}
                dati['subject'] = int(save_filename)  # must be an integer..
                dati['annotations'] = [line.T.tolist() for line in anno]
                json_data.append(dati)

    if args.generate_json_annotations:
        with open('annotation_line_data.json', 'w') as outfile:
            json.dump(json_data, outfile)


def generate_annotations(output):
    figt, axt = plt.subplots(1, 1)
    cont = axt.contour(output, [0.99])
    paths = cont.collections[0].get_paths()
    figt.clf()
    plt.close(figt)

    lines = []
    for path in paths:
        v = path.vertices
        lines.append(v.T)
    return lines


if __name__ == '__main__':
    main()
