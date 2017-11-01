import argparse
import sys

import tensorflow as tf

FLAGS = None


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device("/job:ps/task:0"):
            W = tf.Variable([1], dtype=tf.float32)
            b = tf.Variable([0], dtype=tf.float32)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)
            loss = tf.reduce_sum(tf.square(W * x + b - y))

            global_step = tf.contrib.framework.get_or_create_global_step()

        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
        # train_op = tf.train.GradientDescentOptimizer(0.001).minimize(


        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=10000)]

        x_train = [1, 2, 3, 4]
        y_train = [0, -1, -2, -3]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/tmp/train_logs/"+FLAGS.job_name+"_"+str(FLAGS.task_index),
                                               # checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:
            loops = 0
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                # print("run loops %d" % (loops))
                mon_sess.run(train_op, {x: x_train, y: y_train})
                loops += 1
                if 0 == loops % 1000:
                    curr_W, curr_b, curr_loss = mon_sess.run([W, b, loss], {x: x_train, y: y_train})
                    print("loop %d: W: %s b: %s loss: %s" % (loops, curr_W, curr_b, curr_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
