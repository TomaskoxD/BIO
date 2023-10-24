import os
import argparse
import time
import numpy as np
import pandas as pd
import sys
import warnings
import platform

warnings.filterwarnings("ignore", category=UserWarning) 
sys.path.append('common')

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from common.progress.bar import Bar

from libs.EyeQ_loader import DatasetGenerator
from libs.trainer import train_step, validation_step, save_output
from libs.metric import compute_metric
from networks.densenet_mcf import dense121_mcs

np.random.seed(0)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Arguments 
parser = argparse.ArgumentParser(description='Project to BIO class in academic year 2023/2024 by Tomáš Ondrušek and Peter Ďurica\n This project is based on EyeQ_dense121 from Kaggle competition that uses different types of nn models to classify images of eyes into 3 categories: Good, Usable, Reject using GPU acceleration.')

parser.add_argument('--model', type=str, default='dense121_mcs', help='model (default: dense121_mcs)')
parser.add_argument('--model_dir', type=str, default='./result/', help='save model directory (default: ./result/)')
parser.add_argument('--pre_model', type=str, default='model' , help='pretrained model (default: model)')
parser.add_argument('--save_model', type=str, default='model' , help='save model name (default: model)')
parser.add_argument('--crop_size', type=int, default=224 , help='crop size (default: 224)')
parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'] , help='label index (default: [\'Good\', \'Usable\', \'Reject\'])')

parser.add_argument('--n_classes', type=int, default=3 , help='number of classes (default: 3)')

parser.add_argument('--epochs', default=20, type=int , help='number of total epochs to run (default: 20)')
parser.add_argument('--batch-size', default=4, type=int , help='mini-batch size (default: 4)')
parser.add_argument('--lr', default=0.01, type=float , help='initial learning rate (default: 0.01)')
parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list , help='loss weight (default: [0.1, 0.1, 0.1, 0.1, 0.6])')

parser.add_argument(
    "--mode",
    type=str,
    nargs='+',
    choices=["train", "test", "evaluate"],
    help="Select the program mode: train, test, evaluate, or a combination like 'train test'",
    required=True,
)

args = parser.parse_args()

cudnn.benchmark = True

if args.model == 'dense121_mcs':
    model = dense121_mcs(n_class=args.n_classes)

if args.pre_model is not None:
    if not os.path.isfile(args.pre_model):
        print('Pretrained model not found in \'' + args.pre_model + '\'... \nExiting...')
        sys.exit(1)
    loaded_model = torch.load(args.pre_model)
    model.load_state_dict(loaded_model['state_dict'])

model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

print('==========================================')
print('device: ')
if torch.cuda.is_available():
    print('GPU - ' + torch.cuda.get_device_name(0))
else:
    print('CPU - ' + platform.processor())


print('==========================================')
print('model: ')
print(model.__class__.__name__)
print('==========================================')
print('criterion: ')
print(criterion)
print('==========================================')
print('optimizer: ')
print(optimizer)
print('==========================================')



transform_list1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, +180)),
    ])

transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])


if "train" in args.mode:
    print("Training mode selected")
    # ...

if "test" in args.mode:
    print("Testing mode selected")
    # ...

if "evaluate" in args.mode:
    print("Evaluation mode selected")
    # ...