'''Plots spatial attention maps'''
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
        transforms.Compose([transforms.ToTensor(), normalize])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    return train_loader

def predict(data_loader, model, weights, batch_size):

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda()

            print(images.size())

            # compute predictions
            pred = model(images)

            if i == 0:
                break

    linear_combination_map = torch.einsum('ijkl,j->ikl', pred, weights)

    x = torch.zeros(batch_size, 3, 7, 7)
    x[:, 0, :, :] = linear_combination_map
    x[:, 1, :, :] = linear_combination_map
    x[:, 2, :, :] = linear_combination_map

    m = torch.nn.Upsample(scale_factor=32, mode='bicubic')
    mm = m(x).cuda()
    mm = torch.sigmoid(10. * mm / torch.std(mm))

    return mm * images

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
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=36, type=int, help='mini-batch size, this is the total '
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--model-path', default='', type=str, help='path to model checkpoint (default: '
                                                                   'ImageNet-pretrained)')
    parser.add_argument('--n_out', default=1000, type=int, help='output dim of pre-trained model')
    parser.add_argument('--class-idx', default=6, type=int, help='class index for which the maps will be computed')

    args = parser.parse_args()

    model = load_model(args)
    map_layer = extract_map_layer_7x7(model)

    weights = model.module.classifier.weight.data[args.class_idx, :].cuda()

    data_loader = load_data(args.data, args)
    preds = predict(data_loader, map_layer, weights, args.batch_size)

    print('Preds shape:',preds.shape)
    print('Imgs shape:', imgs.shape)

    fig_pred = plt.figure(figsize=(16, 16), dpi=300)
    ax_pred = fig_pred.add_subplot('111')
    grid_pred = make_grid(preds, nrow=12, padding=1, normalize=True, scale_each=False)
    show_img(ax_pred, grid_pred, 'linear_combination_maps_class_' + str(args. class_idx) + '.pdf')