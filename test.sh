export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate real-time
PYTHON=python

now=$(date +"%Y%m%d_%H%M%S")

model_name=$1
layer=$2
weight_name=$3
file_name=$4
gpu=$5
model_dir=$6

$PYTHON -u run_demo.py \
    --model_name ${model_name} \
    --layers ${layer} \
    --weight_name ${weight_name} \
    --file_name ${file_name} \
    --gpu ${gpu} \
    --model_dir ${model_dir} \


