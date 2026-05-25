from __future__ import print_function

import os
import sys
from pathlib import Path  # if you haven't already done so

file = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import time
import argparse
import torch.backends.cudnn as cudnn

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import csgp.models.dnn.resnet_large as resnet
from svdkl_cifar import SVDKLCIFAR
from nn_cifar import NNCIFAR
from dak_cifar import DAKCIFAR


model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='CIFAR')
parser.add_argument('--mode',
                    default='test',
                    type=str,
                    choices=['train', 'test', 'val'],
                    help='train | test')
parser.add_argument('--model',
                    type=str,
                    default='dak',
                    choices=['nn', 'nnsvgp', 'svdkl', 'dak'],
                    help='Choose the DKL models to use.')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet34',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--num_classes',
                    type=int,
                    default=100,
                    choices=[10, 100],
                    help='Number of classes in the dataset (default: 10 for CIFAR-10)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--seed',
                    type=int,
                    default=2024,
                    metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('-use-sparse',
                    default=False,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use sparse inference; (default: True)')
parser.add_argument('--dyadic-level',
                    default=5,
                    type=int,
                    metavar='N',
                    help='number of inducing points for SVGP (default: 64)')
parser.add_argument('-pretrained',
                    default=True,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use pre-trained NN; (default: False)')
parser.add_argument('-fix-features',
                    default=True,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use pre-trained NN; (default: False)')
parser.add_argument('--epochs',
                    default=50,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=1000,
                    metavar='N',
                    help='input batch size for testing (default: 10000)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gamma',
                    type=float,
                    default=0.2,
                    metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--num_mc_test',
                    type=int,
                    default=100,
                    metavar='N',
                    help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument('--num_mc_train',
                    type=int,
                    default=17,
                    metavar='N',
                    help='number of Monte Carlo runs during training')
parser.add_argument('--num-features',
                    default=128,
                    type=int,
                    metavar='N',
                    help='number of projections (base GP layers) (default: 16)')
parser.add_argument('--noise-var',
                    default=0.1,
                    type=float,
                    metavar='NV',
                    help='noise variance (default: 0.1)')
parser.add_argument('--log-interval',
                    type=int,
                    default=195,
                    metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint-dir',
                    type=str,
                    default='./checkpoint')
parser.add_argument('--log-dir',
                    type=str,
                    default='./logs')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ###################################
    # Model Setup
    ###################################
    assert args.model in ['dak', 'nnsvgp', 'svdkl', 'nn']
    if args.model == 'dak':
        cifar = DAKCIFAR(args)
    # elif args.model == 'nnsvgp':
    #     cifar = NNSVGPCIFAR(args)
    elif args.model == 'svdkl':
        cifar = SVDKLCIFAR(args)
    else:
        cifar = NNCIFAR(args)

    ###################################
    # optionally resume from a checkpoint
    ###################################
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            cifar.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ###################################
    # Data Loading
    ###################################
    assert args.num_classes in [10, 100]
    if args.num_classes == 10:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_set = datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
        test_set = datasets.CIFAR10(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
        test_set = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    ###################################
    # Log directory
    ###################################
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ###################################
    # Training and Testing
    ###################################
    print(args.mode)
    start = time.time()
    if args.mode == 'train':
        log_dir = args.log_dir
        current_num_files = next(os.walk(log_dir))[2]  # get all files in the directory
        run_num = len(current_num_files)
        log_f_name = log_dir + '/CIFAR' + str(args.num_classes) + "_" + args.model + "_batch_" + str(args.batch_size) + "_log_" + str(run_num) + ".csv"
        print("logging at : " + log_f_name)

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('epoch,loss,acc,top1,nll\n')
        
        epoch_time = []
        for epoch in range(args.epochs):
            if epoch > 5:
                cifar.scheduler.step()

            if args.resume:
                if epoch < args.start_epoch:
                    continue
            
            start_time = time.time()
            loss, top1 = cifar.train(train_loader, epoch)
            end_time = time.time()
            epoch_time.append(end_time - start_time)
            print(f"Runtime: {end_time - start_time:.2f} seconds")
            print(f"Average Runtime: {sum(epoch_time) / len(epoch_time):.2f} seconds")
            
            prec1, nll = cifar.validate(test_loader)

            log_f.write('{},{},{},{},{}\n'.format(epoch, loss, top1, prec1, nll))
            log_f.flush()

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': cifar.model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    os.path.join(
                        args.checkpoint_dir,
                        '{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                           args.num_classes,
                                                           args.batch_size)
                    )
                )

        log_f.close()

    elif args.mode == 'test':
        checkpoint_file = args.checkpoint_dir + '/{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                                                   args.num_classes,
                                                                                   args.batch_size)
        checkpoint = torch.load(checkpoint_file)
        cifar.model.load_state_dict(checkpoint['state_dict'])
        
        start_time = time.time()
        acc, nll, ece = cifar.test(test_loader)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        print("Accuracy: ", acc)
        print("NLL: ", nll)
        print("ECE: ", ece)
    else:
        checkpoint_file = args.checkpoint_dir + '/{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                                                   args.num_classes,
                                                                                   args.batch_size)
        checkpoint = torch.load(checkpoint_file)
        cifar.model.load_state_dict(checkpoint['state_dict'])
        top1, _ = cifar.validate(test_loader)
        print(' * Prec@1 {top1:.3f}'.format(top1=top1))


if __name__ == '__main__':
    main()
