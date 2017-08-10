from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
from six.moves import xrange

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from nn_alexnet import alexnet
from nn_alexnet import alexnet_input

FLAGS = tf.app.flags.FLAGS

def evaluate(fake=False, test=False):
    with tf.Graph().as_default() as g:
        if fake:
            # initialize fake data
            fake_data = np.zeros([1, 256, 256, 1], dtype=np.float32)
            #fake_data = np.random.rand(1, 256, 256, 1).astype(np.float32)
            fake_tensor = tf.convert_to_tensor(fake_data)
            logits = alexnet.inference(fake_tensor)
            probs = alexnet.probability(logits)
            init_op = tf.initialize_all_variables()
        else:
            eval_data = False
            images = []
            labels = []
            if test:
                ids = [ ['45f18c5c_varied_sm', '{:08x}'.format(i)]
                        for i in range(200)]
                images, labels = alexnet_input.synthetic_input_by_ids(
                                    '../xray_data/experiment',
                                    ids)
            else:
                images, labels = alexnet.inputs(eval_data=eval_data, num_threads=1)
            #global_step = (0, name='global_step', trainable=False)

            logits = alexnet.inference(images)
            probs = alexnet.probability(logits)

            batch_size = FLAGS.batch_size
            num_examples = 200
            # variable_averages = tf.train.ExponentialMovingAverage(
            #     alexnet.MOVING_AVERAGE_DECAY
            # )
            # variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            if fake:
                # inference with fake data
                sess.run(init_op)
                lresult, result = sess.run([logits, probs])
                print(lresult)
                print(result)
            else:
                ckpt = tf.train.get_checkpoint_state('../xray_data/alexnet_train')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return

                if test:
                    [pred, gt] = sess.run([probs, labels])
                    predt = (pred > 0.5).astype(np.int)
                    gt = gt[:, 0:alexnet.NUM_CLASSES]
                    np.save('pred_f.npy', pred)
                    np.save('gt_f.npy', gt)
                else:
                    coord = tf.train.Coordinator()
                    try:
                        threads = []
                        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                             start=True))
                        num_iter = int(num_examples / batch_size)
                        true_count = 0
                        step = 0
                        predicted_array = None
                        true_array = None
                        while step < num_iter and not coord.should_stop():
                            #predicted_array_new = sess.run(probs)
                            predicted_array_new, true_array_new = sess.run([probs, labels])
                            if predicted_array is None:
                                predicted_array = predicted_array_new
                            else:
                                predicted_array = np.concatenate(
                                    (predicted_array, predicted_array_new),
                                    axis=0
                                )
                            #true_array_new = sess.run(labels)
                            if true_array is None:
                                true_array = true_array_new
                            else:
                                true_array = np.concatenate(
                                    (true_array, true_array_new),
                                    axis=0
                                )
                            print('Evaluated: %d / %d' % (step + 1, num_iter))
                            step += 1
                    except Exception as e:  # pylint: disable=broad-except
                        coord.request_stop(e)

                    coord.request_stop()
                    coord.join(threads, stop_grace_period_secs=10)

                    # mask = np.array([0, 1, 2, 3, 4, 10, 11, 13, 14])
                    # true_array = true_array[:,mask]
                    # predicted_array = predicted_array[:,mask]
                    ap = average_precision_score(true_array, predicted_array, average=None)
                    np.save('true.npy', true_array)
                    np.save('output.npy', predicted_array)
                    print('AP = ')
                    print(ap)
                    print('mAP = %f' % np.mean(ap))


if __name__ == '__main__':
    evaluate(fake='--fake' in sys.argv[1:], test='--test' in sys.argv[1:])
