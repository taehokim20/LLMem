export WANDB_MODE=offline
# train_llama.py train_colo.py train_colo_wiki.py // train_colo_dp.py
# --model_name_or_path huggyllama/llama-7b facebook/opt-6.7b huggyllama/llama-13b
# --master_port=8888 
# facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b
# gpt2 gpt2-medium gpt2-large gpt2-xl distilgpt2 / openai-gpt (not support gradient checkpointing)
# bigcode/gpt_bigcode-santacoder (1.1B params)
# bigscience/bloom-560m bigscience/bloom-1b1 bigscience/bloom-1b7 bigscience/bloom-3b
# Salesforce/codegen-350M-nl Salesforce/codegen-2B-nl
# microsoft/biogpt microsoft/BioGPT-Large
# bigcode/gpt_bigcode-santacoder
# EleutherAI/gpt-neo-1.3B
# bert-base-uncased bert-large-uncased
# prev: --learning_Rate 1e-5, --per_device_train_batch_size 8 --gradient_accumulation_steps 16
torchrun --nproc_per_node 1 dp_real.py \
    --model_name_or_path facebook/opt-125m \
    --data_path ./alpaca_data.json \
    --output_dir ./trained/no.pt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03
    # \
    # | tee ./logs/colo_opt-6.7b.log
