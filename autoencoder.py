import tensorflow as tf
import numpy as np
from data import dataprep
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.estimator.inputs import numpy_io

tf.logging.set_verbosity(tf.logging.INFO)

train_arr, test_arr, train_eval_arr = dataprep.get_train_test()

train_rating = {'ratings': train_arr.astype(np.float32)}
test_eval_rating = {'ratings': train_eval_arr.astype(np.float32), 'targets': test_arr.astype(np.float32)}
train_eval_rating = {'ratings': train_eval_arr.astype(np.float32), 'targets': train_eval_arr.astype(np.float32)}


def auto_encoder(features, targets, mode, params):
    ratings = features['ratings']

    with tf.name_scope("dense_to_sparse"):
        idx = tf.where(tf.not_equal(ratings, 0.0))
        # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
        sparse_ratings = tf.SparseTensor(idx, tf.gather_nd(ratings, idx), ratings.get_shape())

    with tf.variable_scope("encoder"):
        encoder_w = tf.get_variable("e_w", shape=[params['n_users'], params['n_dims']])
        encoder_b = tf.get_variable("e_b", shape=[params['n_dims']], initializer=tf.zeros_initializer)

        encoding_op = tf.sigmoid(tf.sparse_tensor_dense_matmul(sparse_ratings, encoder_w)) + encoder_b

    with tf.variable_scope("decoder"):
        decoder_w = tf.get_variable("d_w", shape=[params['n_dims'], params['n_users']])
        decoder_b = tf.get_variable("d_b", shape=[params['n_users']], initializer=tf.zeros_initializer)

        decoding_op = tf.identity(tf.matmul(encoding_op, decoder_w)) + decoder_b

    loss = None
    if mode == model_fn_lib.ModeKeys.TRAIN:
        with tf.name_scope("loss"):
            # backpropagate only partial observed ratings
            neg_sparse_decoding = tf.SparseTensor(idx, tf.negative(tf.gather_nd(decoding_op, idx)),
                                                  decoding_op.get_shape())
            reg_loss = layers.apply_regularization(layers.l2_regularizer(scale=params['l2reg']),
                                                   weights_list=[encoder_w, decoder_w])
            diff = tf.sparse_add(sparse_ratings, neg_sparse_decoding).values
            loss = tf.reduce_mean(tf.square(diff)) + reg_loss
    if mode == model_fn_lib.ModeKeys.EVAL:
        with tf.name_scope("eval_loss"):
            targets = features['targets']
            target_idx = tf.where(tf.not_equal(targets, 0.0))
            # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
            sparse_targets = tf.SparseTensor(target_idx, tf.gather_nd(targets, target_idx), targets.get_shape())
            neg_sparse_decoding = tf.SparseTensor(target_idx, tf.negative(tf.gather_nd(decoding_op, target_idx)),
                                                  decoding_op.get_shape())
            diff = tf.sparse_add(sparse_targets, neg_sparse_decoding).values
            loss = tf.sqrt(tf.reduce_mean(tf.square(diff)))

    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(var.name, var)

    predictions = loss
    eval_metric_ops = {'loss': loss}
    train_op = layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer=tf.train.AdamOptimizer,
        summaries=[
            "learning_rate",
            "loss",
            "gradients",
            "gradient_norm",
        ])

    return ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops)


with tf.Session() as sess:
    model_params = dict(
        n_items=train_arr.shape[0],
        n_users=train_arr.shape[1],
        n_dims=300,
        l2reg=0.0001,
        learning_rate=0.0001
    )

    # input queue for training
    train_input_fn = numpy_io.numpy_input_fn(
        x=train_rating, y=np.zeros(shape=[train_arr.shape[0], 1]), batch_size=256, shuffle=True, num_epochs=None)
    # input queue for evaluation on test data
    test_eval_input_fn = numpy_io.numpy_input_fn(
        x=test_eval_rating, y=np.zeros(shape=[test_arr.shape[0], 1]), batch_size=test_arr.shape[0], shuffle=False,
        num_epochs=None)
    # input queue for evaluation on training data
    train_eval_input_fn = numpy_io.numpy_input_fn(
        x=train_eval_rating, y=np.zeros(shape=[test_arr.shape[0], 1]), batch_size=test_arr.shape[0], shuffle=False,
        num_epochs=None)

    validation_monitor = monitors.ValidationMonitor(input_fn=test_eval_input_fn, eval_steps=1, every_n_steps=100,
                                                    name='test')
    train_monitor = monitors.ValidationMonitor(input_fn=train_eval_input_fn, eval_steps=1, every_n_steps=100,
                                               name='train')

    autoencoder_cf = Estimator(
        model_fn=auto_encoder,
        params=model_params,
        model_dir='model/_summary/auto_rec_test',
        config=RunConfig(save_checkpoints_secs=60))

    autoencoder_cf.fit(input_fn=train_input_fn, steps=15000, monitors=[validation_monitor, train_monitor])
