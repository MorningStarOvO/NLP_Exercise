#! /bin/sh
seed=13
epoch=64
batch_sz=256
model="NNLM"
language="en"

CUDA_VISIBLE_DEVICES=1  python  tools_qxy/exercise2/main.py  --model ${model} \
--epochs ${epoch}  --random ${seed} --batch_size ${batch_sz} --txt_type ${language} \
--learning-rate  0.01