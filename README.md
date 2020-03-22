# argument-mining

## finetune with IMHO
```bash
export TASK_NAME=IMHO

python run_argmining.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_lower_case \
    --data_dir ./$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```
## finetune with ChangeMyView
```bash
export TASK_NAME=CMV

python run_argmining.py \
    --model_type bert \
    --model_name_or_path ./pytorch_model.bin \
    --task_name $TASK_NAME \
    --do_train \
    --do_lower_case \
    --data_dir ./$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/$TASK_NAME/
```
