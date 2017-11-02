## run multi-gpu train:
CUDA_VISIBLE_DEVICES='0,1' python cifar100_multi_gpu_train.py --data_dir ../data --num_gpus 2 --max_steps=1000 --train_dir /train --log_dir log
