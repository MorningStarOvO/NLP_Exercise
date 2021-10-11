seed=100
epochs=25

for model in DNN
do 

python tools_qxy/exercise1/main.py --model ${model} \
--epochs ${epochs}  --random ${seed}

done