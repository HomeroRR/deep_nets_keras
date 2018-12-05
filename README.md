# deep_nets_keras

Implementation of AlexNet, VGG, and Resnet using Keras!
Available datasets are CIFAR10, MNIST, and ImageNet

To use run the run.sh on terminal.

Example run:
```python
 python main.py --train True --eval True --test True --dataset 'ImageNet' --model 'resnet50' --epochs 90 --workers 4 --batch_size 128 -lr 0.01
```


To run plotting of train loss and accuracy:
```python
python tools/compare_models.py [optional dir default is '../results/ImageNet']
```

To run plotting or loss and accuracies for train,eval,test:
```python
python tools/plot_loss.py [log.txt file for a model result default is '../results/ImageNet/resnet50/learningrate0.01_batchsize128/models/log.txt']
 ```
Tensorflow implementations are certainly available online but they tend to be messy and hard to explain.
I opted to use Keras for simplicity. 
Keras is already pretty robust and can have a tensorflow with GPU backend, if GPUs are available in your machine.
Main drawbacks are the lack of flexibility to customize models.

Future plans include to re-write using Tensorflow to add on to my other custom models. Specially the GAN ones.
