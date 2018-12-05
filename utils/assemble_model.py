import keras

#import models
from models.alexnet import alexnet_model
from models.resnet import resnet_model
from models.vgg import vgg19_model

def assemble_model(args):
    model = None
    shape, classes = (args.box_sz, args.box_sz, args.box_chls), args.num_classes 
    model_name = args.model
    if model_name=='resnet50':
        x = keras.layers.Input(shape)
        model = resnet_model(x, classes)
    elif model_name=='vgg':
        model = vgg19_model(shape, classes)
    else:
        model = alexnet_model(shape, classes)
    return model