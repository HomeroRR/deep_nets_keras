# Import necessary packages
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Main entrypoint for training ResNet50, vgg19 or alexnet')
    parser.add_argument(
        '--workers',
        dest='workers',
        help='number of workers for data loader',
        default=4,
        type=int)
    parser.add_argument(
        '--model',
        dest='model',
        help='model type to be trained',
        default='resnet50',
        type=str)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='learning rate',
        default=0.01,
        type=float)
    parser.add_argument(
        '--epochs',
        dest='epochs',
        help='number of epochs to train for',
        default=10,
        type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='size of each batch',
        default=128,
        type=int)
    parser.add_argument(
        '--box_sz',
        dest='box_sz',
        help='width or height of image',
        default=32,
        type=int)
    parser.add_argument(
        '--box_chls',
        dest='box_chls',
        help='number of color channels, usually 3 , Red, Green, Blue',
        default=3,
        type=int)
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='number of classes for the model',
        default=1000,
        type=int)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset to use',
        default='ImageNet',
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='directory that contains the desired dataset',
        default='./datasets/ImageNet',
        type=str)
    parser.add_argument(
        '--results_dir',
        dest='results_dir',
        help='directory to place model results',
        default='./results',
        type=str)
    parser.add_argument(
        '--train',
        dest='train',
        help='whether to train the model',
        default=True,
        type=bool)
    parser.add_argument(
        '--eval',
        dest='eval',
        help='whether to eval the model',
        default=False,
        type=bool)
    parser.add_argument(
        '--test',
        dest='test',
        help='whether to run model on test set',
        default=False,
        type=bool)
    args = parser.parse_args()
    return args

def get_logs_dir(args):
    return args.results_dir+"/"+args.dataset+"/"+args.model+"/learningrate"+str(
        round(args.lr,2))+"_batchsize"+str(args.batch_size)+"/models"

