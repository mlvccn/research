model_name=$1
layers=$2
bs=$3
lr=$4
cw=$5
epoch=$6
gpu=$7
pretrain=$8
aw=$9


# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.0 wo_gate_wo_skip false false false

# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.0 wo_gate_w_skip false true true

# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.1 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.2 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.3 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.4 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.5 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.6 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.7 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.8 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.9 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_wo_skip true false false

# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.1 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.1 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.2 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.3 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.4 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.5 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.6 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.7 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.8 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.9 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_w_early true true true

sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${lr} ${cw} ${epoch} ${gpu} ${pretrain} ${aw} poly/${pretrain}/2aux/early_fusion_weight_c${cw}/wd05_b${bs}_lr${lr}_e${epoch}_aw${aw}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.1 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.2 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.3 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.4 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.5 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.6 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.7 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.8 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false 
# sh train_gate1_weight.sh ${model_name} ${layers} ${bs} ${cw} ${epoch} ${gpu} ${pretrain} 0.9 poly/${pretrain}/early_fusion_weight_c${cw}/wd05_b${bs}_lr001_e${epoch}/w_gate_w_skip_wo_early true true false
