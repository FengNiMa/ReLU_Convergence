#!/bin/sh

python3 3_layer_ReLU_net_jupyter.py --dataset 'synthetic'\
  --params '100'\
  --n_epoch 1000\
  --lr 0.0002 &
python3 3_layer_ReLU_net_jupyter.py --dataset 'synthetic'\
  --params '200'\
  --n_epoch 1000\
  --lr 0.0002 &
python3 3_layer_ReLU_net_jupyter.py --dataset 'MNIST'\
  --params '01'\
  --n_epoch 1000\
  --lr 0.0002 &
python3 3_layer_ReLU_net_jupyter.py --dataset 'MNIST'\
  --params '17'\
  --n_epoch 1000\
  --lr 0.0002 &
python3 3_layer_ReLU_net_jupyter.py --dataset 'Fashion'\
  --params '01'\
  --n_epoch 1000\
  --lr 0.0002 &
python3 3_layer_ReLU_net_jupyter.py --dataset 'Fashion'\
  --params '79'\
  --n_epoch 1000\
  --lr 0.0002

