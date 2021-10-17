seed=100
epochs=25

for model in CNN
do 

CUDA_VISIBLE_DEVICES=0 python tools_qxy/exercise1/main.py --model ${model} \
--epochs ${epochs}  --random ${seed}

done