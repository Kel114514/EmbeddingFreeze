#!/bin/bash

# model=$1
# model_name=$2
# model_config=$3
# model_config_name=$4
# dataset=$5
# dataset_config=$6
# dataset_name=$7
# param=$8
model=GRU4Rec_Ours
model_name="GRU"
# model_config=$3
model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--reranker_CAN_NUM=100+--reranker_merging_mode=replace"
model_config_name="softmax_CPR:100_Mi"
dataset=gowalla
dataset_config=gowalla
dataset_name=gowalla
param=GRU_mem4_fast

python run_recbole.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml ${model_config//+/ } --pretr_emb_epoch=14 > log_${model_name}_${dataset_name}_default.txt 2>&1
# python run_hyper.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml --params_file=hyper_config/hyper.${param} ${model_config//+/ } --efficient_mode='None' --hyper_results="hyper_results/hyper_${model_name}_${dataset_name}_${model_config_name}"
echo "python run_hyper.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml --params_file=hyper_config/hyper.${param} ${model_config//+/ } --efficient_mode='None' --hyper_results='hyper_results/hyper_${model_name}_${dataset_name}_${model_config_name}'"
