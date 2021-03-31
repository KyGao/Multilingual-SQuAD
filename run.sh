DATA_DIR=$1
RESULT_DIR=$2
NUM_EPOCHS=$3
LR=$4
SEED=$5

TASK_DATA_DIR=DATA_DIR
TRAIN_FILE=${TASK_DATA_DIR}/train-v1.1.json
PREDICT_FILE=${TASK_DATA_DIR}/dev-v1.1.json

MAXL=384
# LR=3e-5
# NUM_EPOCHS=3.0

RESULT_DIR=${RESULT_DIR}/mlm_LR${LR}_EPOCH${NUM_EPOCHS}_SEED${SEED}

python run_squad.py \
    --model_type "xlm" \
    --model_name_or_path "xlm-mlm-xnli15-1024" \
    --do_lower_case \
    --do_train \
    --do_eval \
    --data_dir ${TASK_DATA_DIR} \
    --train_file ${TRAIN_FILE} \
    --predict_file ${PREDICT_FILE} \
    --per_gpu_train_batch_size 16 \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length $MAXL \
    --doc_stride 128 \
    --save_steps -1 \
    --overwrite_output_dir \
    --fp16 \
    --seed ${SEED} \
    --gradient_accumulation_steps 2 \
    --warmup_steps 500 \
    --output_dir ${RESULT_DIR} \
    --weight_decay 0.005 \
    --threads 8 \
    --lang_id 4 \
