import numpy as np
from torchinfo import summary
from patchgan.unet import UNet, Discriminator, get_norm_layer
from patchgan.io import MmapDataGenerator
from patchgan.trainer import Trainer, device
import argparse
import tqdm
from torchvision.transforms import Resize
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgDir', '-i', required=True)
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--resize_size', default=256)
    parser.add_argument('--generate_json_annotations', default=False)
    parser.add_argument('--gen_filts', default=32)
    parser.add_argument('--disc_filts', default=16)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--in_channels', default=3)
    parser.add_argument('--out_channels', defaults=1)
    parser.add_argument('--saveloc', default='./predicted_masks')
    args = parser.parse_args()
    
    list_of_images = glob.glob(f'{args.imgDir}/*.png')
    print(f'Found {len(list_of_images)} images in the folder {args.imgDir}')
    
    GEN_FILTS = args.gen_filts
    DISC_FILTS = args.disc_filts
    ACTIV = 'relu'

    IN_NC = args.in_channels
    OUT_NC = args.out_channels

    norm_layer = get_norm_layer()

    # create the generator
    generator = UNet(IN_NC, OUT_NC, GEN_FILTS, norm_layer=norm_layer,
                     use_dropout=False, activation=ACTIV).to(device)
    generator.load_state_dict(torch.load(f'{args.checkpoint}', map_location='cpu'))
    print('Loaded the generator checkpoint successfully')
    
    generator.eval()
    
    resize_function = Resize(args.resize_size)
    if args.generate_json_annotations:
        json_data = []
    with torch.no_grad():
        for each_image in tqdm.tqdm(list_of_images):
            image_data = plt.imread(each_image)
            resized_image = torch.unsqueeze(resize_function(torch.tensor(np.moveaxis(IM_data, 2, 0))), 0)
            predicted_mask = generator(resized_image)
            predicted_mask_array = torch.squeeze(torch.squeeze(gen_mask,0),0).numpy()
            save_filename = os.path.basename(each_image).strip('.png')
            np.save(f'{args.saveloc}/{save_filename}.npy', predicted_mask_array)
            
            if args.generate_json_annotations:
                anno = generate_annotations(predicted_mask_array)
                dati = {}
                dati['subject'] = int(save_filename) #must be an integer..
                dati['annotations'] = [line.T.tolist() for line in anno]
                json_data.append(dati)
    
    if args.generate_json_annotations:
        with open('annotation_line_data.json', 'w') as outfile:
            json.dump(json_data, outfile)
                
            

def generate_annotations(array):
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
            

if name == '__main__':
    main()
            
    