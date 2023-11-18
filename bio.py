##############################################
#                                            #
#    BIO                                     #
#                                            #
#    Main file of the project.               #
#                                            #
#    Author: Tomáš Ondrušek (xondru18)       #
#            Peter Ďurica   (xduric05)       #
#                                            #
##############################################

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

from libs.loader import DatasetGenerator
from libs.trainer import Trainer, Validator
from libs.metric import MetricCalculator
from libs.saver import Saver
from networks.densenet_mcf import dense121_mcs
from networks.resnet_mcf import resnet50_mcs, resnet18_mcs

np.random.seed(0)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cprint(text, color):
    if color == "blue":
        color = 94
    elif color == "red":
        color = 91
    elif color == "green":
        color = 92
    print("\033[{}m{}\033[0m".format(color, text))


# Arguments 
parser = argparse.ArgumentParser(description='Project to BIO class in academic year 2023/2024 by Tomáš Ondrušek and Peter Ďurica\n This project is based on EyeQ_dense121 from Kaggle competition that uses different types of nn models to classify images of eyes into 3 categories: Good, Usable, Reject using GPU acceleration.')

parser.add_argument('--model', type=str, default='dense121_mcs', help='model (default: dense121_mcs)')
parser.add_argument('--model_dir', type=str, default='./result/', help='save model directory (default: ./result/)')
parser.add_argument('--pre_model', type=str, default='model' , help='pretrained model (default: model)')
parser.add_argument('--save_model', type=str, default='model' , help='save model name (default: model)')
parser.add_argument('--crop_size', type=int, default=224 , help='crop size (default: 224)')
parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'] , help='label index (default: [\'Good\', \'Usable\', \'Reject\'])')
parser.add_argument('--test_images_dir', type=str, default='images/test' , help='test image directory (default: preprocess/train_preprocessed)') # TODO: change to test_preprocessed
parser.add_argument('--train_images_dir', type=str, default='images/train' , help='train image directory (default: preprocess/train_preprocessed)')
parser.add_argument('--label_train_file', type=str, default='data/train_labels.csv' , help='label train file (default: data/Label_EyeQ_train.csv)')
parser.add_argument('--label_test_file', type=str, default='data/test_labels.csv' , help='label test file (default: data/Label_EyeQ_test.csv)') # TODO: change to test

parser.add_argument('--clasified_images_dir', type=str, default='data/DenseNet121_v3_v1_mine.csv' , help='clasified images directory (default: data/DenseNet121_v3_v1_mine.csv)')


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
elif args.model == 'resnet50_mcs':
    model = resnet50_mcs(n_class=args.n_classes)
elif args.model == 'resnet18_mcs':
    model = resnet18_mcs(n_class=args.n_classes)

else:
    print('Model not found... \nExiting...')
    sys.exit(1)
if "train" not in args.mode:
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

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

# check if files exist
if not os.path.isfile(args.label_train_file):
    cprint('Label train file not found in \'{}\' ... \nExiting...'.format(args.label_train_file), "red")
    sys.exit(1)

if not os.path.isfile(args.label_test_file):
    cprint('Label test file not found in \'{}\' ... \nExiting...'.format(args.label_test_file), "red")
    sys.exit(1)

if not os.path.isdir(args.train_images_dir):
    cprint('Train images directory not found in \'{}\' ... \nExiting...'.format(args.train_images_dir), "red")
    sys.exit(1)

if not os.path.isdir(args.test_images_dir):
    cprint('Test images directory not found in \'{}\' ... \nExiting...'.format(args.test_images_dir), "red")
    sys.exit(1)

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)






if "train" in args.mode:
    cprint("\n\nTraining mode selected", "blue")

    data_train = DatasetGenerator(data_dir=args.train_images_dir, list_file=args.label_train_file, transform1=transform_list1, transform2=transformList2, n_class=args.n_classes, set_name='train')

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    data_val = DatasetGenerator(data_dir=args.train_images_dir, list_file=args.label_train_file, transform1=transform_list_val1, transform2=transformList2, n_class=args.n_classes, set_name='val')

    val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    print('Length of train_loader: ', len(train_loader))
    print('Length of val_loader: ', len(val_loader))
    best_metric = np.inf
    best_iter = 0   
    best_model = model
    trainer = Trainer(model, optimizer, criterion, args.loss_w, args.epochs)
    validator = Validator(model, criterion)
    for epoch in range(0, args.epochs):
        _ = trainer.train_step(train_loader, epoch)
        validation_loss = validator.validate(val_loader)
        print('Current Loss: {}| Best Loss: {} at epoch: {}'.format(validation_loss, best_metric, best_iter))

        # save model
        if best_metric > validation_loss:
            best_iter = epoch
            best_metric = validation_loss
            best_model = model
    
    model = best_model
    cprint('Saving model with best loss: {} at epoch: {}'.format(best_metric, best_iter), "red")
    model_save_file = os.path.join(args.model_dir, args.save_model + '.tar')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save({'state_dict': model.state_dict(), 'best_loss': best_metric}, model_save_file)
    cprint('Model saved to {}'.format(model_save_file), "green")

if "test" in args.mode: # DONE
    cprint("\n\nTesting mode selected", "blue")

    data_test = DatasetGenerator(data_dir=args.test_images_dir, list_file=args.label_test_file, transform1=transform_list_val1, transform2=transformList2, n_class=args.n_classes, set_name='test')

    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print('Length of test_loader: ', len(test_loader))

    outPRED_mcs = torch.FloatTensor().cuda()
    model.eval()
    iters_per_epoch = len(test_loader)
    bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
    bar.check_tty = False
    for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        begin_time = time.time()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
        outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.1f} min.'.format(epochID + 1, len(test_loader),
                                                            batch_time=batch_time * (iters_per_epoch - epochID) / 60)
        bar.next()
    bar.finish()

    saver = Saver(args.label_idx)
    cprint('Saving results to: {}'.format(args.clasified_images_dir), "green")
    saver.save(args.label_test_file, outPRED_mcs, args.clasified_images_dir)


if "evaluate" in args.mode: # DONE
    cprint("\n\nEvaluating mode selected", "blue")
    cprint('Reading test labels from: {}'.format(args.label_test_file), "red")
    df_gt = pd.read_csv(args.label_test_file)
    img_list = df_gt["image"].tolist()
    GT_QA_list = np.array(df_gt["quality"].tolist())
    img_num = len(img_list)

    cprint('Reading results from: {}'.format(args.clasified_images_dir), "red")
    df_tmp = pd.read_csv(args.clasified_images_dir)
    predict_tmp = np.zeros([img_num, 3])

    for idx in range(3):
        predict_tmp[:, idx] = np.array(df_tmp[args.label_idx[idx]].tolist())

    calculator = MetricCalculator(GT_QA_list, predict_tmp, args.label_idx)
    calculator.get_metrics()
    calculator.print_metrics(args.model, args.clasified_images_dir)


