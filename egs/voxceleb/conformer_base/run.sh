#!/bin/bash
# Apache 2.0.

# data path
SRE_ROOT=../../..
voxceleb2_test_root=path/to/voxceleb2_test
voxceleb2_train_root=path/to/voxceleb2_train/dev/aac
musan_path=path/to/musan_split
rirs_path=path/to/RIRS_NOISES/simulated_rirs

# Conformer
num_blocks=12
input_layer=conv2d
pos_enc_layer_type=rel_pos # no_pos| rel_pos

# train args
encoder_name=conformer_cat # conformer | conformer_cat
pooling_type=ASP #ASP
embedding_dim=256
loss_name=aamsoftmax # softmax | amsoftmax | aamsoftmax
scale=32.0
margin=0.2
num_classes=5994 # 1211 | 5994 | 7205
second=2

# other
save_dir=exp/${encoder_name}_${pooling_type}_${embedding_dim}_${loss_name}_${scale}_${margin}_${num_blocks}
batch_size=256
max_epochs=40
cuda_device=0
learning_rate=0.001
step_size=5
gamma=0.5
stage=-1
stop_stage=-1

checkpoint=

# model parameter averaging
num_avg=5
avg_model=$save_dir/avg_model_${num_avg}.ckpt

#large margin fine-tuning
lm_save_dir=exp/${encoder_name}_${pooling_type}_${embedding_dim}_${loss_name}_${scale}_${margin}_${num_blocks}_lm
lm_margin=0.5
lm_second=6
lm_batch_size=128
lm_max_epochs=5
lm_learning_rate=0.00001
lm_avg_model=$lm_save_dir/avg_model_${num_avg}.ckpt
lm_num_avg=1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # prepare data for model training
  rm -rf data
  mkdir -p data
  echo Build $voxceleb2_train_root list
  python3 local/build_datalist.py \
          --extension wav \
          --speaker_level 1 \
          --dataset_dir $voxceleb2_train_root \
          --data_list_path data/train_lst.csv

  echo Build $musan_path list
  python3 local/build_datalist.py \
          --dataset_dir $musan_path \
          --extension wav \
          --data_list_path data/musan_lst.csv

  echo Build $rirs_path list
  python3 local/build_datalist.py \
          --dataset_dir $rirs_path \
          --extension wav \
          --data_list_path data/rirs_lst.csv
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # prepare test trials for evaluation

  mkdir -p data/trials
  # VoxCeleb1-Clean 
  wget -P data/trials https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
  # VoxCeleb1-H-Clean
  wget -P data/trials https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt
  # VoxCeleb1-E-Clean
  wget -P data/trials https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt
  # voxsrc2021_val.txt
  wget -P data/trials https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/voxsrc2021_val.txt

  python3 local/format_trials.py \
          --voxceleb1_root $voxceleb2_test_root \
          --src_trials_path data/trials/veri_test2.txt \
          --dst_trials_path data/trials/VoxCeleb1-Clean.txt

  python3 local/format_trials.py \
          --voxceleb1_root $voxceleb2_test_root \
          --src_trials_path data/trials/list_test_hard2.txt \
          --dst_trials_path data/trials/VoxCeleb1-H-Clean.txt

  python3 local/format_trials.py \
          --voxceleb1_root $voxceleb2_test_root \
          --src_trials_path data/trials/list_test_all2.txt \
          --dst_trials_path data/trials/VoxCeleb1-E-Clean.txt

  python3 local/format_trials.py \
          --voxceleb1_root $voxceleb2_test_root \
          --src_trials_path data/trials/voxsrc2021_val.txt \
          --dst_trials_path data/trials/VoxSRC2021_val.txt
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # model training
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --train_csv_path data/train_lst.csv \
        --trial_path data/trials/VoxCeleb1-Clean.txt \
        --save_dir $save_dir \
        --batch_size $batch_size \
        --num_workers 40 \
        --max_epochs $max_epochs \
        --learning_rate $learning_rate \
        --step_size $step_size \
        --gamma $gamma \
        --weight_decay 0.0000001 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $margin \
        --num_classes $num_classes \
        --second $second \
        ${checkpoint:+--checkpoint $checkpoint} \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --add_reverb_noise \
        --noise_csv_path data/musan_lst.csv \
        --rir_csv_path data/rirs_lst.csv \
        --spec_aug_flag
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  python local/average_model.py \
    --dst_model $avg_model \
    --src_path $save_dir \
    --num ${num_avg}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Large margin fine-tuning
  echo "Large margin fine-tuning"
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --train_csv_path data/train_lst.csv \
        --trial_path data/trials/VoxCeleb1-Clean.txt \
        --save_dir $lm_save_dir \
        --batch_size $lm_batch_size \
        --num_workers 40 \
        --max_epochs $lm_max_epochs \
        --learning_rate $lm_learning_rate \
        --step_size $step_size \
        --gamma $gamma \
        --weight_decay 0.0000001 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $lm_margin \
        --num_classes $num_classes \
        --second $lm_second \
        ${checkpoint:+--checkpoint $checkpoint} \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --add_reverb_noise \
        --noise_csv_path data/musan_lst.csv \
        --rir_csv_path data/rirs_lst.csv \
        --spec_aug_flag \
        --do_lm_path $avg_model
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Do large margin fine-tuning model average ..."
  python local/average_model.py \
    --dst_model $lm_avg_model \
    --src_path $lm_save_dir \
    --num ${lm_num_avg}
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Make cohort set"
  python $SRE_ROOT/scripts/make_cohort_set.py \
        --data_list_path data/train_lst.csv \
        --cohort_save_path data/cohort.txt \
        --num_cohort 3000
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Evaluate VoxCeleb1-Clean"
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --eval \
        --save_dir $lm_save_dir \
        --trial_path data/trials/VoxCeleb1-Clean.txt \
        --num_workers 40 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $lm_margin \
        --num_classes $num_classes \
        --second $lm_second \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $lm_avg_model \
        --cohort_path data/cohort.txt \
        --asnorm
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Evaluate VoxCeleb1-H-Clean"
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --eval \
        --save_dir $lm_save_dir \
        --trial_path data/trials/VoxCeleb1-H-Clean.txt \
        --num_workers 40 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $lm_margin \
        --num_classes $num_classes \
        --second $lm_second \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $lm_avg_model \
        --cohort_path data/cohort.txt \
        --asnorm
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Evaluate VoxCeleb1-E-Clean"
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --eval \
        --save_dir $lm_save_dir \
        --trial_path data/trials/VoxCeleb1-E-Clean.txt \
        --num_workers 40 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $lm_margin \
        --num_classes $num_classes \
        --second $lm_second \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $lm_avg_model \
        --cohort_path data/cohort.txt \
        --asnorm
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "Evaluate voxsrc2021_val"
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SRE_ROOT/main.py \
        --eval \
        --save_dir $lm_save_dir \
        --trial_path data/trials/VoxSRC2021_val.txt \
        --num_workers 40 \
        --encoder_name $encoder_name \
        --pooling_type $pooling_type \
        --embedding_dim $embedding_dim \
        --loss_name $loss_name \
        --scale $scale \
        --margin $lm_margin \
        --num_classes $num_classes \
        --second $lm_second \
        --num_blocks $num_blocks \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $lm_avg_model \
        --cohort_path data/cohort.txt \
        --asnorm
fi