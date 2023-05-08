RANDOM_SEED=`date '+%Y-%m-%d-%H-%M-%S'`
export OMP_NUM_THREADS=64

### Supervised Instruction Tuning ###
torchrun --standalone --nproc_per_node=1 train_sft.py \
    --pretrain "/home/vmagent/app/dataset/gpt2" \
    --model 'gpt2' \
    --strategy naive \
    --log_interval 10 \
    --save_path  /home/vmagent/app/dataset/Coati-7B/sf-instinwild \
    --dataset /home/vmagent/app/dataset/InstructionWild/data/instinwild_en.json \
    --batch_size 4 \
    --accimulation_steps 8 \
    --lr 2e-5 \
    --max_epochs 1 2>&1 | tee SFT_${RANDOM_SEED}.log \
    #--dataset /home/vmagent/app/dataset/InstructionWild/data/instinwild_en.json \

### Training the Reward Model ###
#python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=1 --nnodes=1 train_reward_model.py \
#torchrun --standalone --nproc_per_node=1 train_reward_model.py \
#    --pretrain "/home/vmagent/app/dataset/gpt2" \
#    --model 'gpt2' \
#    --strategy naive \
#    --loss_fn 'log_exp' \
#    --test True \
#    --save_path '/home/vmagent/app/dataset/Coati-7B/rw' #2>&1 | tee Train_RW_${RANDOM_SEED}.log \

### Training Model with Reinforcement Learning by Human Feedback ###
#torchrun --standalone --nproc_per_node=1 train_prompts.py \
#         --pretrain "/home/vmagent/app/dataset/gpt2" \
#         --model 'gpt2' \
#         --strategy naive \
#         --prompt_path /home/vmagent/app/dataset/InstructionWild/data/instinwild_en.json \
#         --pretrain_dataset /home/vmagent/app/dataset/InstructionWild/data/instinwild_en.json \
#         --rm_pretrain /home/vmagent/app/dataset/gpt2 \
#         --save_path /home/vmagent/app/dataset/Coati-7B/actor_checkpoint_prompts \
#         --rm_path /home/vmagent/app/dataset/Coati-7B/rw/model.pt 2>&1 | tee Train_RL_${RANDOM_SEED}.log