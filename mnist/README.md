# mnist example

This example is modified from official tensorflow notebook "mnist_from_scratch" example. Use only CPU on  one machine

#### command to run:

python mint_train.py --data <mnist data> --log-dir <log dir>

#### command to run in MLCloud:

mlcloud fs put <mnist-data> /data/

mlcloud fs put mnist_train.py /mnist/

mlcloud fs mkdir /mnist/logs

mlcloud job submit --name=mnist --type=tensorflow --num-workers=1 --tensorboard --log-dir="/mlcloud/mnist/logs" --command="python /mlcloud/mnist/mnist_train.py --data /mlcloud/data/mnist --log-dir /mlcloud/mnist/logs"

