DATASET='ImageNet'
MODEL='resnet50'
EPOCHS=90
N_WORKERS=4 # Number of workers for data loader
BATCH_SIZE=128
LR=0.01
python main.py --workers $N_WORKERS --model $MODEL --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE \
    --dataset $DATASET

