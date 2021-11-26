#! /bin/sh
seed=1321
epoch=64
batch_sz=256
model="RNNLM"
language="en"

CUDA_VISIBLE_DEVICES=0  python  tools_qxy/exercise2/main.py  --model ${model} \
--epochs ${epoch}  --random ${seed} --batch_size ${batch_sz} --txt_type ${language}  \
--learning-rate  1e-2