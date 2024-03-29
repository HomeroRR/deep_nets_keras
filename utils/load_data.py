import numpy as np

from datasets.imagenet import load_imagenet
from datasets.cifar10 import load_cifar10
from datasets.mnist import load_mnist

def load_data(args, data_partition="train"):
    X,Y = None, None
    if args.dataset=='ImageNet':
        shape, classes = (args.box_sz, args.box_sz, args.box_chls), args.num_classes   
        X,Y = load_imagenet(args.batch_size, classes, shape, args.data_dir +"/"+data_partition)
    elif args.dataset=='cifar10':
        X,Y = load_cifar10(args.num_classes)
    else:
        X,Y = load_mnist(args.num_classes)
        X = np.expand_dims(X,axis=3)
    return X,Y