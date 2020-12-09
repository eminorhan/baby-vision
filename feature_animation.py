'''Animating features on short clips'''
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

# TODO: combine the map extraction functions into a single function 
# TODO: combine model loading functions into a single function 

def extract_map_layer_7x7_res(res_model):
    layer_list = list(res_model.module.children())[:-2]
    new_model = torch.nn.Sequential(*layer_list)
    return new_model

def extract_map_layer_7x7(mobilenetV2_model):
    layer_list = list(mobilenetV2_model.module.features.children())
    new_model = torch.nn.Sequential(*layer_list)
    return new_model

def extract_map_layer_14x14(mobilenetV2_model):
    layer_list = list(mobilenetV2_model.module.features.children())
    new_layer_list = layer_list[:-5]
    new_layer_list.append(layer_list[-5].conv[0])
    new_model = torch.nn.Sequential(*new_layer_list)
    return new_model

def load_model_res(args):
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)
    model = torch.nn.DataParallel(model).cuda()

    if args.model_path:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    return model

def load_model(args):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Linear(in_features=1280, out_features=args.n_out, bias=True)
    model = torch.nn.DataParallel(model).cuda()

    if args.model_path:
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    return model

def load_data(data_dir, args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    return train_loader

def predict(data_loader, model, batch_size, feature_idx):

    # switch to evaluate mode
    model.eval()

    preds_list = []
    imgs_list = []

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda()

            # compute predictions
            preds = model(images)
            preds = preds[:, feature_idx, :, :]
            preds_list.append(preds)
            imgs_list.append(images)

    preds = torch.cat(preds_list, 0)
    images = torch.cat(imgs_list, 0)

    print('Images shape:', images.size())
    print('Preds shape:', preds.size())

    # Copy activation map to all channels and upsample to image size
    x = torch.zeros(preds.size()[0], 3, 7, 7)
    x[:, 0, :, :] = preds
    x[:, 1, :, :] = preds
    x[:, 2, :, :] = preds

    m = torch.nn.Upsample(scale_factor=32, mode='bicubic')

    upsampled_maps = m(x).cuda()
    # upsampled_maps = torch.sigmoid(10. * upsampled_maps / torch.std(upsampled_maps))

    upsampled_maps = upsampled_maps.cpu().numpy()
    images = images.cpu().numpy()

    return  upsampled_maps, images

def show_img(ax, img, save_name):
    '''Save maps'''
    npimg = img.cpu().numpy()

    print(npimg.shape)

    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    mp.rcParams['axes.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 1.15
    mp.rcParams['font.sans-serif'] = ['FreeSans']
    mp.rcParams['mathtext.fontset'] = 'cm'

    plt.savefig(save_name, bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot spatial attention maps')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--workers', default=32, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=900, type=int, help='mini-batch size, this is the total '
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--model-path', default='', type=str, help='path to model checkpoint (default: '
                                                                   'ImageNet-pretrained)')
    parser.add_argument('--n_out', default=2765, type=int, help='output dim of pre-trained model')
    parser.add_argument('--feature-idx', default=1, type=int, help='feature index for which the maps will be computed')

    args = parser.parse_args()

    model = load_model(args)
    map_layer = extract_map_layer_7x7(model)

    data_loader = load_data(args.data, args)
    preds, images = predict(data_loader, map_layer, args.batch_size, args.feature_idx)

    preds = preds - preds.min()
    preds = preds / preds.max()
    preds = np.uint8(255 * preds)

    images = images - images.min()
    images = images / images.max()
    # images = np.uint8(255 * images)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title('Feature: ' + str(args.feature_idx))

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    preds = jet_colors[preds[:, 0, :, :]]

    masked_imgs = 1.0 * preds + np.transpose(images, (0, 2, 3, 1))
    masked_imgs = np.uint8(255 * masked_imgs / masked_imgs.max())

    imgs = []
    for i in range(900):
        im = ax.imshow(masked_imgs[i])

        if i == 0:
            im = ax.imshow(masked_imgs[i])

        imgs.append([im])

    ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True, repeat_delay=1000)

    # To save the animation, use e.g.
    ani.save('intphys_feature_animation_' + str(args.feature_idx) + '.mp4')