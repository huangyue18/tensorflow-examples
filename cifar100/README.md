Cifar100 examples - running on CPU , one GPU and multiple GPUs
--------------------------------------------------------------
## running on local CPU
```console
python cifar100_train.py

```

## running on local GPUs
```console
CUDA_VISIBLE_DEVICES='0,1' python cifar100_multi_gpu_train.py \
    --data_dir <cifar100-data>
    --num_gpus 2 --max_steps=1000 \
    --train_dir <train-dir> --log_dir <log-dir>
```

## running on CPU on MLCloud
```console
# upload the dataSet
mlcloud fs put <cifar100-data> /data/

# upload your code
mlcloud fs put examples/cifar100 /

# create the event log dir
mlcloud fs mkdir /cifar100/train

# submit job
mlcloud job submit --name=cifar100-cpu --type=tensorflow --num-worker=1 \
  --tensorboard --log-dir /mlcloud/cifar100/train \
   --command="python /mlcloud/cifar100/cifar100_train.py \
              --data_dir /mlcloud/data/cifar100 \
              --train_dir /mlcloud/cifar100/train"
```


## running on single GPU on MLCloud
```console
mlcloud fs put <cifar100-data> /data/

mlcloud fs put examples/cifar100 /

mlcloud fs mkdir /cifar100/log

mlcloud fs mkdir /cifar100/train_single_gpu

mlcloud job submit --name=cifar100-single-gpu --type=tensorflow --num-worker=1 --num-gpu 1 \
  --tensorboard --log-dir /mlcloud/cifar100/train_single_gpu \
   --command="python /mlcloud/cifar100/cifar100_multi_gpu_train.py \
              --data_dir /mlcloud/data/cifar100 \
              --train_dir /mlcloud/cifar100/train_single_gpu"
```

## running on multiple GPUs on MLCloud
```console
mlcloud fs put <cifar100-data> /data/

mlcloud fs put examples/cifar100 /

mlcloud fs mkdir /cifar100/log

mlcloud fs mkdir /cifar100/train_multiple_gpu

mlcloud job submit --name=cifar100-multiple-gpu --type=tensorflow --num-worker=1 --num-gpu 2 \
  --tensorboard --log-dir /mlcloud/cifar100/train_multiple_gpu \
   --command="python /mlcloud/cifar100/cifar100_multi_gpu_train.py \
              --data_dir /mlcloud/data/cifar100 \
              --num_gpus 2 \
              --train_dir /mlcloud/cifar100/train_multiple_gpu"
```
