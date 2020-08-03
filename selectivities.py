'''Measure single feature class selectivities'''
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid


def extract_map_layer_7x7(mobilenetV2_model):
    layer_list = list(mobilenetV2_model.module.features.children())
    new_model = torch.nn.Sequential(*layer_list)
    return new_model

def extract_map_layer_14x14(mobilenetV2_model, layer):
    layer_list = list(mobilenetV2_model.module.features.children())
    new_layer_list = layer_list[:-layer]
    new_layer_list.append(layer_list[-layer].conv[0])
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
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    return train_loader

def predict(data_loader, model):

    targets = []
    preds = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda()

            # compute predictions
            pred = model(images)
            pred = torch.mean(pred, dim=(2, 3))

            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())

            print('Iter:', i)

    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)

    print('Targets size:', targets.shape)
    print('Preds size:', preds.shape)

    return targets, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Measure single feature class selectivities')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=580, type=int, help='mini-batch size, this is the total '
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--model-path', default='', type=str, help='path to model checkpoint '
                                                                   '(default: ImageNet-pretrained)')
    parser.add_argument('--n_out', default=1000, type=int, help='output dim of pre-trained model')
    parser.add_argument('--layer', default=1, type=int, choices=[1, 2, 6, 10, 14, 18], help='which layer?')

    args = parser.parse_args()

    model = load_model(args)
    if args.layer == 1:
        map_layer = extract_map_layer_7x7(model)
    else:
        map_layer = extract_map_layer_14x14(model, args.layer)

    data_loader = load_data(args.data, args)
    targets, preds = predict(data_loader, map_layer)

    n_classes = 26
    n_neurons = preds.shape[1]
    class_matrix_mean = np.zeros((n_neurons, n_classes))
    class_matrix_std = np.zeros((n_neurons, n_classes))

    for i in range(n_neurons):
        for j in range(n_classes):
            aux_vec = preds[targets==j, i]
            class_matrix_mean[i, j] = np.mean(aux_vec)
            class_matrix_std[i, j] = np.std(aux_vec)

    sorted_mean = np.sort(class_matrix_mean, axis=1)
    selectivity = (sorted_mean[:, -1] - np.mean(sorted_mean[:, :-1], axis=1)) / \
                  (sorted_mean[:, -1] + np.mean(sorted_mean[:, :-1], axis=1))

    print('Most selective 10 features:', np.argsort(selectivity)[-10:])
    print('Highest 10 selectivities:', np.sort(selectivity)[-10:])
    print('Selectivity shape:', selectivity.shape)

    np.save('selectivity_' + str(args.layer) + '.npy', selectivity)