CUDA_VISIBLE_DEVICES=1 python NFM-DRA/DRA/train.py \
                --dataset fundus \
                --experiment_dir NFM-DRA/DRA/experiment \
                --epochs 300 \
                --test_type valid
