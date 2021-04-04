#!/bin/bash
step='UODA'
date='0404'
method='UODA'
model='resnet34'
LR=0.01
num=3
device_id=0
main_file='main.py'
dataset='multi'

source=real
target=clipart
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
--method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

source=real
target=painting
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
 --method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

source=painting
target=clipart
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
 --method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

source=clipart
target=sketch
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
 --method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

source=sketch
target=painting
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
 --method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

source=real
target=sketch
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
--method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

ssource=painting
target=real
CUDA_VISIBLE_DEVICES=$device_id python $main_file \
 --method $method --dataset $dataset \
 --source $source --target $target --net $model --lr $LR --num $num \
 > './semi_baselines_logs/'$source'_to_'$target'_'$method'_'$num'_'$model'_'$date'_'$step'_'$LR'.file'

