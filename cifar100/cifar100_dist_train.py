import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os.path
import cifar100

import logging

FORMAT = '%(asctime)-15s %(message)s'
logfile = './log/cifar100-dist-gpu-bench' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.log'
logging.basicConfig(format=FORMAT,filename=logfile,level='INFO')
logger = logging.getLogger('log')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar100_dist_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1,
                            """Number of training step to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """worker's task index.""")
tf.app.flags.DEFINE_string('ps_hosts', '127.0.0.1:2222',
                            """ps server hosts url.""")
tf.app.flags.DEFINE_string('worker_hosts', '127.0.0.1:2223',
                            """worker server hosts url.""")
tf.app.flags.DEFINE_string('job_name', 'woker',
                            """job name.""")


def train():
  print('FLAGS.data_dir: %s' % FLAGS.data_dir)
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  if FLAGS.job_name == 'ps':
    server.join()
  is_chief = (FLAGS.task_index == 0)
  with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % FLAGS.task_index,
      ps_device="/job:ps/task:0",
      cluster=cluster)):
    global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)

    # Get images and labels for CIFAR-100.
    images, labels = cifar100.distorted_inputs()
    num_workers = len(worker_hosts)
    num_replicas_to_aggregate = num_workers
    logits = cifar100.inference(images)
    # Calculate loss.
    loss = cifar100.loss(logits, labels)
    # Retain the summaries from the chief.
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar100.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar100.NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar100.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar100.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    if is_chief:
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
      # Add a summary to track the learning rate.
      summaries.append(tf.summary.scalar('learning_rate', lr))

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.SyncReplicasOptimizer(
        opt,
        replicas_to_aggregate=num_replicas_to_aggregate,
        total_num_replicas=num_workers,
        #use_locking=True)
        use_locking=False)
    # Calculate the gradients for the batch
    grads = opt.compute_gradients(loss)
    # Add histograms for gradients at the chief worker.
    if is_chief:
      for grad, var in grads:
          if grad is not None:
              summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # apply gradients to variable
    train_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    if is_chief:
      for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    #variable_averages = tf.train.ExponentialMovingAverage(
    #      cifar100.MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #train_op = tf.group(train_op, variables_averages_op)

    if is_chief:
      #Build the summary operation at the chief worker
      summary_op = tf.summary.merge(summaries)

  chief_queue_runner = opt.get_chief_queue_runner()
  init_token_op = opt.get_init_tokens_op()
  # Build an initialization operation to run below.
  init_op = tf.global_variables_initializer()
  # Create a saver.
  saver = tf.train.Saver(tf.global_variables())

  sv = tf.train.Supervisor(is_chief=is_chief,
                       global_step=global_step,
                       init_op=init_op)
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)


  with sv.prepare_or_wait_for_session(server.target,config=sess_config) as sess:
  # Start running operations on the Graph. allow_soft_placement must be set to
  # True to build towers on GPU, as some of the ops do not have GPU
  # implementations.
    # start sync queue runner and run the init token op at the chief worker
    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    sv.start_queue_runners(sess, queue_runners)

    if is_chief:
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_token_op)
    #open the summary writer
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    t1 = time.time()
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * num_workers
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / num_workers
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
      if step % 100 == 0:
        if is_chief:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        if is_chief:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    t2 = time.time()
    print('spent %f seconds to train %d step' % (t2 - t1, FLAGS.max_steps))
    logger.info('spent %f seconds to train %d step' % (t2 - t1, FLAGS.max_steps))
    logger.info('last loss value: %.2f ' % loss_value)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
