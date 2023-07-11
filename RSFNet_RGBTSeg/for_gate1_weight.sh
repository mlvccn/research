model_name=$1
layers=$2
bs=$3
lr=$4
cw=$5
epoch=$6
gpu=$7
pretrain=$8
aw=$9

sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${lr} ${cw} ${epoch} ${gpu} ${pretrain} ${aw} poly/${pretrain}/2aux/early_fusion_weight_c${cw}/wd05_b${bs}_lr${lr}_e${epoch}_aw${aw}/w_gate_w_skip_wo_early true true false 

