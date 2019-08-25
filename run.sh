#!/bin/sh

LR=0.001
NEPOCH=1000

python3 3_layer_ReLU_net_jupyter.py --dataset 'synthetic'\
  --params '100'\
  --n_epoch $NEPOCH\
  --lr $LR &
python3 3_layer_ReLU_net_jupyter.py --dataset 'synthetic'\
  --params '50'\
  --n_epoch $NEPOCH\
  --lr $LR &
python3 3_layer_ReLU_net_jupyter.py --dataset 'MNIST'\
  --params '01'\
  --n_epoch $NEPOCH\
  --lr $LR &
python3 3_layer_ReLU_net_jupyter.py --dataset 'MNIST'\
  --params '17'\
  --n_epoch $NEPOCH\
  --lr $LR &
python3 3_layer_ReLU_net_jupyter.py --dataset 'Fashion'\
  --params '01'\
  --n_epoch $NEPOCH\
  --lr $LR &
python3 3_layer_ReLU_net_jupyter.py --dataset 'Fashion'\
  --params '79'\
  --n_epoch $NEPOCH\
  --lr $LR &
