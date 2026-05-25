from __future__ import print_function

import os
import sys
from pathlib import Path  # if you haven't already done so

file = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import time
import argparse

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from dak_mnist import DAKMNIST
from svdkl_mnist import SVDKLMNIST
from nn_mnist import NNMNIST

len_trainset = 60000
len_testset = 10000

parser = argparse.ArgumentParser(description='MNIST')
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
parser.add_argument('--seed',
                    type=int,
                    default=0,
                    metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('-use-sparse',
                    default=False,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use sparse inference; (default: True)')
parser.add_argument('--epochs',
                    default=5,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_mc_train',
                    type=int,
                    default=20,
                    metavar='N',
                    help='number of Monte Carlo runs during training')
parser.add_argument('--dyadic-level',
                    default=6,
                    type=int,
                    metavar='N',
                    help='number of inducing points for DKL (default: 64)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
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
parser.add_argument('--subset-size',
                    type=int,
                    default=60000,
                    metavar='N',
                    help='the size of the training subset')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1.0,
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
                    default=0.7,
                    metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--num_mc_test',
                    type=int,
                    default=100,
                    metavar='N',
                    help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument('--num-features',
                    default=64,
                    type=int,
                    metavar='N',
                    help='number of features (GP layers) (default: 16)')
parser.add_argument('--noise-var',
                    default=0.01,
                    type=float,
                    metavar='NV',
                    help='noise variance (default: 0.1)')
parser.add_argument('--log-interval',
                    type=int,
                    default=200,
                    metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    default='./checkpoint')
parser.add_argument('--log_dir',
                    type=str,
                    default='./logs')
parser.add_argument('-vb',
                    '--verbose',
                    default=True,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if train the hyperparameters; (default: True)')
parser.add_argument('-val',
                    '--validate',
                    default=False,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if validate model on validation set; (default: False)')


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ###################################
    # Prepare DataLoader
    ###################################
    # Prepare MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Create a subset of the MNIST dataset
    if args.subset_size < len_trainset:
        subset_size = args.subset_size
        subset_indices = torch.randperm(len(mnist_dataset))[:subset_size]
        subset_mnist = Subset(mnist_dataset, subset_indices)

        # Create the subset DataLoader
        train_loader = DataLoader(subset_mnist, batch_size=args.batch_size, shuffle=True)

    else:
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size,
            shuffle=True
        )

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        './data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size,
        shuffle=False
    )

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ###################################
    # Model Setup
    ###################################
    assert args.model in ['dak', 'nnsvgp', 'svdkl', 'nn']
    if args.model == 'dak':
        mnist = DAKMNIST(args)
    # elif args.model == 'nnsvgp':
    #     mnist = NNSVGPMNIST(args)
    elif args.model == 'svdkl':
        mnist = SVDKLMNIST(args)
    else:
        mnist = NNMNIST(args)

    ###################################
    # Training and Testing
    ###################################
    print(args.mode)
    if args.mode == 'train':
        log_dir = args.log_dir
        current_num_files = next(os.walk(log_dir))[2]  # get all files in the directory
        run_num = len(current_num_files)
        log_f_name = log_dir + '/MNIST_' + args.model + "_batch_" + str(
            args.batch_size) + "_log_" + str(run_num) + ".csv"
        print("logging at : " + log_f_name)

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('epoch,loss,acc\n')

        losses = []
        epoch_time = []
        for epoch in range(args.epochs):
            print("epoch " + str(epoch), end=', ')
            start_time = time.time()
            loss = mnist.train(train_loader, epoch)
            end_time = time.time()
            epoch_time.append(end_time - start_time)
            print(f"Runtime: {end_time - start_time:.2f} seconds")
            print(f"Average Runtime: {sum(epoch_time) / len(epoch_time):.2f} seconds")
            
            val_loss, val_acc = mnist.validate(test_loader)
            # mnist.scheduler.step()

            log_f.write('{},{},{}\n'.format(epoch, val_loss, val_acc))
            log_f.flush()
            losses += loss
            torch.save(
                mnist.model.state_dict(),
                os.path.join(args.checkpoint_dir, '{}_mnist_{}_batch.pth'.format(args.model, args.batch_size))
            )
        log_f.close()
    elif args.mode == 'test':
        checkpoint = os.path.join(args.checkpoint_dir, '{}_mnist_{}_batch.pth'.format(args.model, args.batch_size))
        mnist.model.load_state_dict(torch.load(checkpoint))
        
        start_time = time.time()
        acc, nll, ece = mnist.test(test_loader)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        print("Accuracy: ", acc)
        print("NLL: ", nll)
        print("ECE: ", ece)
    else:
        checkpoint = os.path.join(args.checkpoint_dir, '{}_mnist_{}_batch.pth'.format(args.model, args.batch_size))
        mnist.model.load_state_dict(torch.load(checkpoint))
        mnist.validate(test_loader)
    


if __name__ == '__main__':
    main()
