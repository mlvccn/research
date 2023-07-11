export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate real-time
PYTHON=python

now=$(date +"%Y%m%d_%H%M%S")

model_name=$1
layers=$2
bs=$3
lr=$4
cw=$5
epoch=$6
gpu=$7
pretrained=$8
aw=$9
fusion=$10
wg=$11
ws=$12
es=$13


exp_dir=PolyWeightPath/${model_name}/${layers}/${fusion}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
tensorboard_dir=${exp_dir}/${now}

mkdir -p ${model_dir} ${result_dir} ${tensorboard_dir}
cp train_gate1_weight.py model/${model_name}.py ${tensorboard_dir}

$PYTHON -u ${tensorboard_dir}/train_gate1_weight.py \
    --model_name ${model_name} \
    --layers ${layers} \
    --batch_size ${bs} \
    --lr_start ${lr} \
    --epoch_max ${epoch} \
    --gpu ${gpu} \
    --class_weight ${cw} \
    --aux_weight ${aw} \
    --save_path ${model_dir} \
    --save_model ${now} \
    --tensorboard_dir ${tensorboard_dir} \
    --pretrained ${pretrained} \
    --with_gate ${wg} \
    --with_skip ${ws} \
    --early_skip ${es} \
    2>&1 | tee ${tensorboard_dir}/train.log

