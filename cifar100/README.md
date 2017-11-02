## one machine, multi-gpu training on CIFAR100 dataset 

####run multi-gpu train:

CUDA_VISIBLE_DEVICES='0,1' python cifar100_multi_gpu_train.py --data_dir <cifar100-data> --num_gpus 2 --max_steps=1000 --train_dir <train-dir> --log_dir <log-dir>

#### run multi-gpu training on MLCloud:

mlcloud fs put <cifar100-data> /data/

mlcloud fs put examples/cifar100 /

mlcloud fs mkdir /cifar100/log

mlcloud fs mkdir /cifar100/train

mlcloud job submit --name=multi-gpu --type=tensorflow --num-worker=1 --num-gpu 2  --tensorboard --log-dir /mlcloud/cifar100/train --command="python /mlcloud/cifar100/cifar100_multi_gpu_train.py --data_dir /mlcloud/data/cifar100 --num_gpus 2 --max_steps 1000 --train_dir /mlcloud/cifar100/train --log_dir /mlcloud/cifar100/log"