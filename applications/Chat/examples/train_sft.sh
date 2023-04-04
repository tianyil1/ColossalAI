RANDOM_SEED=`date '+%Y-%m-%d-%H-%M-%S'`
export OMP_NUM_THREADS=64

torchrun --standalone --nproc_per_node=2 train_sft.py \
    --pretrain "/home/vmagent/app/dataset/LLaMA-7B/llama-7b-hf" \
    --model 'llama' \
    --strategy naive \
    --log_interval 10 \
    --save_path  /home/vmagent/app/dataset/Coati-7B \
    --dataset /home/vmagent/app/dataset/InstructionWild/data/instinwild_en.json \
    --batch_size 4 \
    --accimulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 2>&1 | tee SFT_${RANDOM_SEED}.log\
