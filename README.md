# LexMAE Pre-training and Fine-tuning

- This is a python implementation with PyTorch for our paper [**LexMAE: Lexicon-Bottlenecked Pretraining for Large-Scale Retrieval**](https://openreview.net/forum?id=SHD0Dc1M5r)
- This is an early release version and would have some bugs caused by code reorganizations. 

## TODO

- [] Fine-tuned Models
- [] Pre-trained Models


## Env

### Python
Please use conda env.
```
conda create -n lexmae python=3.7
conda activate lexmae
conda install pytorch=1.9.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install transformers==4.11.3
pip install gpustat ipython jupyter datasets accelerate sklearn tensorboard nltk pytrec_eval
conda install -c conda-forge faiss-gpu
pip install prettytable gradio
pip install setuptools==59.5.0
```

To use mixed precision, please `export MIXED_PRECISION=fp16`

### Anserini 

Install Anserini for Sparse Lexcon-based Retrieval.
```
# I personally recommend to install java and maven in Anaconda for Anserini
conda install -c conda-forge openjdk=11 maven
# Please install anserini-0.14.0 As we only tested on ths version. Just follow
https://github.com/castorini/anserini/tree/anserini-0.14.0#getting-started
```
We will involke aserini from python script for lexicon-based retrieval. Suppose we have env var $ANSERINI_PATH to `.../anserini`

## Data Preparation

### Overview of Data Dir
```
$DATA_DIR/
--msmarco/
----collection.tsv
----collection.tsv.title.tsv (titles, copied from https://github.com/texttron/tevatron)
----passage_ranking/
------train.query.txt [502939 lines]
------qrels.train.tsv [532761 lines] 
------train.negatives.tsv [400782 lines] (BM25 negatives, copied from tevatron)
------dev.query.txt [6980 lines]
------qrels.dev.tsv [7437 lines] 
------top1000.dev [6668967 lines] 
------test2019.query.txt [200 lines]  
------qrels.test2019.tsv [9260 lines] 
------top1000.test2019 [189877 lines] 
```
If not specified, please download the file from official, https://microsoft.github.io/msmarco/Datasets, and then rename it accordingly.



## Pre-training



```
LEXMAE_DIR="/TO-DIR-PATH-SAVING-LEXMAE"
DATA_DIR="/TO-DIR-PATH-WITH-DATA-FILES"
DATA_NAME="msmarco/fullpsgs.tsv"

export MIXED_PRECISION=fp16

INIT_MODEL_DIR="bert-base-uncased --encoder bert"; NUM_EPOCH="20"; BATCH="2048"; LR="3e-4"; SEED="42"; 
DATA_P="0.30"; DEC_P="0.50"; PARAMS=""; NAME="";
python3 -m torch.distributed.run --nproc_per_node=8 \
  -m script_lexmae \
  --do_train \
  --model_name_or_path ${INIT_MODEL_DIR} \
  --warmup_proportion 0.05 --weight_decay 0.01 --max_grad_norm 0. \
  --data_dir $DATA_DIR \
  --data_rel_paths msmarco/fullpsgs.tsv --max_length 144 \
  --output_dir ${LEXMAE_DIR} \
  --logging_steps 100 --eval_batch_size 48 \
  --eval_steps -1 --dev_key_metric none \
  --data_load_type disk --num_proc 4 --seed ${SEED} \
  --gradient_accumulation_steps 1 --learning_rate ${LR} --num_train_epochs ${NUM_EPOCH} --train_batch_size ${BATCH} \
  --data_mlm_prob ${DATA_P} --enc_mlm_prob 0.00 \
  --dec_mlm_prob ${DEC_P} ${PARAMS} --tensorboard_steps 100
```

Note that, an extra step is required to get `msmarco/fullpsgs.tsv`. That is combining `msmarco/collection.tsv.title.tsv` and `msmarco/collection.tsv` with columns of PID, TITLE and PASSAGE.

## Fine-tuning

First, set env variables:

```
export MIXED_PRECISION=fp16
export CUDA_VISIBLE_DEVICES=0
```

### Stage 1

#### Fine-tuning

```
STG1_MODEL_DIR="/SET-A-DIR-NAME"

python3 \
  -m script_lexmae \
  --do_train \
  --encoder ${ENCODER_TYPE} --model_name_or_path ${LEXMAE_DIR} \
  --max_length 144 ${MODEL_PARAMS} \
  --warmup_proportion 0.05 --weight_decay 0.01 --max_grad_norm 1. \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir $STG1_MODEL_DIR \
  --logging_steps 100 --num_dev 6980 --eval_batch_size 48  \
  --data_load_type memory --num_proc 3 --gradient_accumulation_steps 1 \
  --eval_steps 10000 --seed ${SEED} --lambda_d 0.${LMD} --lambda_ratio ${LMD_R} \
  --learning_rate ${LR} --num_train_epochs ${NUM_EPOCH} --train_batch_size ${BATCH} \
  --negs_sources official --num_negs_per_system ${NUM_SYS} --num_negatives ${NUM_NEG} \
  --tensorboard_steps 100 --do_xentropy

ENCODER_TYPE="bert"; MODEL_PARAMS=""; 
LMD="0020"; LMD_R="0.75"; NUM_SYS="1000"; NUM_NEG="15"; NUM_EPOCH="3"; BATCH="24"; LR="2e-5"; SEED="42"; 
```

#### Eval

```
python3 \
  -m script_lexmae \
  --do_prediction \
  --model_name_or_path $STG1_MODEL_DIR \
  --seed 42 --anserini_path $ANSERINI_PATH \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG1_MODEL_DIR}-EVAL \
  --data_load_type disk --num_proc 3 --max_length 144 --eval_batch_size 160 \
  --hits_num 1000 --encoder bert
```

#### Hard Negative Mining

```
python3 \
  -m script_lexmae \
  --do_hn_gen \
  --model_name_or_path $STG1_MODEL_DIR \
  --seed 42 --anserini_path $ANSERINI_PATH \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG1_MODEL_DIR}-EVAL \
  --data_load_type disk --num_proc 3 --max_length 144 --eval_batch_size 160 \
  --hn_gen_num 512 --encoder bert 
```

### Stage 2

#### Fine-tuning

```

STG2_MODEL_DIR="/SET-A-DIR-NAME"

STAGE1_HN="${STG1_MODEL_DIR}-EVAL/sparse_retrieval/qid2negatives.pkl"; 
MODEL_PARAMS=""; LMD="0080"; LMD_R="0.75"; NUM_SYS="200"; NUM_NEG="15"; NUM_EPOCH="3"; BATCH="24"; LR="2e-5"; SEED="42";

python3 \
  -m script_lexmae \
  --do_train \
  --encoder ${ENCODER_TYPE} --model_name_or_path ${LEXMAE_DIR} \
  --max_length 144 ${MODEL_PARAMS} \
  --warmup_proportion 0.05 --weight_decay 0.01 --max_grad_norm 1. \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG2_MODEL_DIR} \
  --logging_steps 100 --num_dev 6980 --eval_batch_size 48  \
  --data_load_type memory --num_proc 3 --gradient_accumulation_steps 1 \
  --eval_steps 10000 --seed ${SEED} --lambda_d 0.${LMD} --lambda_ratio ${LMD_R} \
  --learning_rate ${LR} --num_train_epochs ${NUM_EPOCH} --train_batch_size ${BATCH} \
  --negs_sources custom --negs_source_paths ${STAGE1_HN} --num_negs_per_system ${NUM_SYS} --num_negatives ${NUM_NEG} \
  --tensorboard_steps 100 --do_xentropy

```

#### Eval
```
python3 \
  -m script_lexmae \
  --do_prediction \
  --model_name_or_path $STG2_MODEL_DIR \
  --seed 42 --anserini_path $ANSERINI_PATH \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG2_MODEL_DIR}-EVAL \
  --data_load_type disk --num_proc 3 --max_length 144 --eval_batch_size 160 \
  --hits_num 1000 --encoder bert
```

#### Hard Negative Mining

```
python3 \
  -m script_lexmae \
  --do_hn_gen \
  --model_name_or_path $STG2_MODEL_DIR \
  --seed 42 --anserini_path $ANSERINI_PATH \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG2_MODEL_DIR}-EVAL \
  --data_load_type disk --num_proc 3 --max_length 144 --eval_batch_size 160 \
  --hn_gen_num 512 --encoder bert 
```

### Stage 3

#### Fine-tuning

```
STG3_MODEL_DIR="/SET-A-DIR-NAME"

RANKER_DIR="/PATH-TO-DIR"
STAGE2_HN="${STG2_MODEL_DIR}-EVAL/sparse_retrieval/qid2negatives.pkl"; 
MODEL_PARAMS="--distill_reranker ${RANKER_DIR} --xentropy_sparse_loss_weight 0.2"; 
LMD="0080"; LMD_R="0.75"; NUM_SYS="1000"; NUM_NEG="23"; NUM_EPOCH="3"; BATCH="16"; LR="2e-5"; SEED="42";

python3 \
  -m script_lexmae \
  --do_train \
  --encoder ${ENCODER_TYPE} --model_name_or_path ${INIT_MODEL_DIR} \
  --max_length 144 ${MODEL_PARAMS} \
  --warmup_proportion 0.05 --weight_decay 0.01 --max_grad_norm 1. \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG3_MODEL_DIR} \
  --logging_steps 100 --num_dev 6980 --eval_batch_size 48  \
  --data_load_type memory --num_proc 3 --gradient_accumulation_steps 1 \
  --eval_steps 10000 --seed ${SEED} --lambda_d 0.${LMD} --lambda_ratio ${LMD_R} \
  --learning_rate ${LR} --num_train_epochs ${NUM_EPOCH} --train_batch_size ${BATCH} \
  --negs_sources custom --negs_source_paths ${STAGE2_HN} --num_negs_per_system ${NUM_SYS} --num_negatives ${NUM_NEG} \
  --tensorboard_steps 100 --do_xentropy
```

#### Eval
```
python3 \
  -m script_lexmae \
  --do_prediction \
  --model_name_or_path $STG3_MODEL_DIR \
  --seed 42 --anserini_path $ANSERINI_PATH \
  --data_dir $DATA_DIR --overwrite_output_dir \
  --output_dir ${STG3_MODEL_DIR}-EVAL \
  --data_load_type disk --num_proc 3 --max_length 144 --eval_batch_size 160 \
  --hits_num 1000 --encoder bert
```


