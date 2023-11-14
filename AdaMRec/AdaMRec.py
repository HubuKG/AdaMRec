import functools
import numpy as np
import toolz
from evaluator import Evaluator
from sampler import WarpSampler
import load_data as Data
import scipy as sp
import os, sys
import argparse
from time import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess = tf.Session()

class GroupNormalization(tf.keras.layers.Layer):

    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupNormalization, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = self.add_weight(name='gamma', shape=(1, 1, c_num), initializer='random_normal', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(1, 1, c_num), initializer='zeros', trainable=True)
        self.eps = eps

    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3],
        x = tf.reshape(x, (N, self.group_num, -1))
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        x = (x - mean) / (std + self.eps)
        x = tf.reshape(x, (N, H, W, C))
        return x * self.gamma + self.beta


class SRU(tf.keras.layers.Layer):

    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupNormalization(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        gn_x = self.gn(x)

        w_gamma = self.gn.gamma / tf.reduce_sum(self.gn.gamma)
        reweights = self.sigmoid(gn_x * w_gamma)
        # Gate
        info_mask = reweights >= self.gate_treshold
        info_mask_float = tf.cast(info_mask, dtype=tf.float32)

        noninfo_mask = reweights < self.gate_treshold
        noninfo_mask_float = tf.cast(noninfo_mask, dtype=tf.float32)

        x_1 = info_mask_float * x
        x_2 = noninfo_mask_float * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = tf.split(x_1, num_or_size_splits=2, axis=3)
        x_21, x_22 = tf.split(x_2, num_or_size_splits=2, axis=3)
        return tf.concat([x_11 + x_22, x_12 + x_21], axis=3)


class CRU(tf.keras.layers.Layer):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.high_channel = high_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - high_channel
        self.squeeze1 = tf.keras.layers.Conv2D(high_channel // squeeze_ratio, kernel_size=1, use_bias=False)
        self.squeeze2 = tf.keras.layers.Conv2D(low_channel // squeeze_ratio, kernel_size=1, use_bias=False)
        self.GWC = tf.keras.layers.Conv2D(op_channel, kernel_size=group_kernel_size, strides=1,
                                          padding='same')

        self.PWC1 = tf.keras.layers.Conv2D(op_channel, kernel_size=1, use_bias=False)
        self.PWC2 = tf.keras.layers.Conv2D(op_channel - low_channel // squeeze_ratio, kernel_size=1, use_bias=False)
        self.advavg = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):

        high, low = tf.split(x, num_or_size_splits=[self.high_channel, self.low_channel], axis=3)
        high, low = self.squeeze1(high), self.squeeze2(low)
        Y1 = self.GWC(high) + self.PWC1(high)
        Y2 = tf.concat([self.PWC2(low), low], axis=3)


        out = tf.concat([Y1, Y2], axis=3)

        out_weight = self.advavg(out)
        out_weight_a = tf.expand_dims(out_weight, 1)
        out_weight_b = tf.expand_dims(out_weight_a, 1)
        out_w = tf.nn.softmax(out_weight_b, axis=3) * out

        out1, out2 = tf.split(out_w, num_or_size_splits=2, axis=3)
        return out1 + out2


class ScConv(tf.keras.layers.Layer):

    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_threshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_threshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_ratio=squeeze_ratio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def call(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


def wrap(function):
    """
    Decorators, which allow the use of decorators without parentheses
    if no arguments are provided. All parameters must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@wrap
def define_range(function, scope=None, *args, **kwargs):
    """
    A decorator designed for functions responsible for defining TensorFlow operations.
    Execution of the wrapped function is limited to a single occurrence. Any subsequent
    invocations will yield the stored result directly, ensuring that operations are
    incorporated into the graph only once.  The operations introduced by the function
    are encapsulated within a tf.variable_scope().  In case this decorator is applied
    with arguments, they will be passed on to the variable scope.
    The default scope name corresponds to the name of the encapsulated function.
    """
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class AdaMRec(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 batch_size=10,
                 visualfeatures=None,
                 textualfeatures=None,
                 op_channel: int = 32,
                 decay_r=1e-4,
                 decay_d=1e-3,
                 learning_rate=0.1,
                 hidden_dim_a=256,
                 hidden_dim_b=256,
                 dropout_a=0.2,
                 dropout_b=0.2,
                 sikp_connect=0.2,
                 ):

        self.ScConv = ScConv(op_channel)
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        if visualfeatures is not None:
            self.visualfeatures = tf.constant(visualfeatures, dtype=tf.float32)
        else:
            self.visualfeatures = None
        if textualfeatures is not None:
            self.textualfeatures = tf.constant(textualfeatures, dtype=tf.float32)
        else:
            self.textualfeatures = None
        self.learning_rate = learning_rate
        self.hidden_dim_a = hidden_dim_a
        self.hidden_dim_b = hidden_dim_b
        self.dropout_a = dropout_a
        self.dropout_b = dropout_b
        self.sikp_connect = sikp_connect
        self.n_semantics = args.n_semantics
        self.decay_r = decay_r
        self.decay_d = decay_d
        self.num_neg = args.num_neg
        self.user_positive_items_pairs = tf.placeholder(tf.int32, [self.batch_size, 2])
        self.negative_samples = tf.placeholder(tf.int32, [self.batch_size, self.num_neg])
        self.user_ids = tf.placeholder(tf.int32, [None])
        self.max_train_count = tf.placeholder(tf.int32, None)
        self.initializer = tf.initializers.glorot_uniform()
        self.user_emb
        self.item_emb
        self.visual_projection
        self.textual_projection
        self.emb_loss
        self.loss
        self.optimize

    @define_range
    def user_emb(self):
        return tf.Variable(self.initializer([self.n_users, self.embed_dim]),
                           name='user_embedding')

    @define_range
    def item_emb(self):
        return tf.Variable(self.initializer([self.n_items, self.embed_dim]),
                           name='item_embedding')

    @define_range
    def emb_loss(self):
        """
        return: the distance metric loss
        """
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair
        # user embedding (N, K)

        # Find user embeddings and positive and negative sample item embeddings
        users = tf.nn.embedding_lookup(self.user_emb,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")
        pos_items = tf.nn.embedding_lookup(self.item_emb, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        pos_i_f = tf.nn.embedding_lookup(self.textual_projection, self.user_positive_items_pairs[:, 1])
        pos_i_v = tf.nn.embedding_lookup(self.visual_projection, self.user_positive_items_pairs[:, 1])

        # negative item embedding (N, K)
        neg_items = tf.reshape(tf.nn.embedding_lookup(self.item_emb, self.negative_samples, name="neg_items"),
                               [-1, self.embed_dim])
        neg_i_f = tf.reshape(tf.nn.embedding_lookup(self.textual_projection, self.negative_samples),
                             [-1, self.embed_dim])
        neg_i_v = tf.reshape(tf.nn.embedding_lookup(self.visual_projection, self.negative_samples),
                             [-1, self.embed_dim])

        items = tf.concat([pos_items, neg_items], 0)
        textual_f = tf.concat([pos_i_f, neg_i_f], 0)
        visual_f = tf.concat([pos_i_v, neg_i_v], 0)

        user_a = tf.tile(users, [self.num_neg + 1, 1])

        # Split the feature vector by blocks
        user_semantics_embedding = tf.split(users, self.n_semantics, 1)
        item_semantics_embedding = tf.split(items, self.n_semantics, 1)
        item_semantics_embedding_p = tf.split(pos_items, self.n_semantics, 1)

        textual_semantics_embedding = tf.split(textual_f, self.n_semantics, 1)
        textual_semantics_embedding_p = tf.split(pos_i_f, self.n_semantics, 1)
        visual_semantics_embedding = tf.split(visual_f, self.n_semantics, 1)
        visual_semantics_embedding_p = tf.split(pos_i_v, self.n_semantics, 1)

        dis_loss = tf.constant(0, dtype=tf.float32)

        for i in range(0, self.n_semantics - 1):
            x = visual_semantics_embedding_p[i]
            y = visual_semantics_embedding_p[i + 1]
            dis_loss += self.distance_correlation(x, y)
            x = textual_semantics_embedding_p[i]
            y = textual_semantics_embedding_p[i + 1]
            dis_loss += self.distance_correlation(x, y)
            x = user_semantics_embedding[i]
            y = user_semantics_embedding[i + 1]
            dis_loss += self.distance_correlation(x, y)
            x = item_semantics_embedding_p[i]
            y = item_semantics_embedding_p[i + 1]
            dis_loss += self.distance_correlation(x, y)

        dis_loss /= ((self.n_semantics + 1.0) * self.n_semantics / 2)

        p_item, n_item = tf.split(items, [self.batch_size, self.num_neg * self.batch_size], 0)
        user_ap, user_an = tf.split(user_a, [self.batch_size, self.num_neg * self.batch_size], 0)

        user_semantics_embedding_a = tf.split(user_a, self.n_semantics, 1)
        user_semantics_embedding_ap = tf.split(user_ap, self.n_semantics, 1)
        user_semantics_embedding_an = tf.split(user_an, self.n_semantics, 1)

        p_item_semantics_embedding = tf.split(p_item, self.n_semantics, 1)
        n_item_semantics_embedding = tf.split(n_item, self.n_semantics, 1)

        regularizer = tf.constant(0, dtype=tf.float32)

        pos_scores, neg_scores = [], []

        for i in range(0, self.n_semantics):
            weights = self.create_weight(user_semantics_embedding_a[i], item_semantics_embedding[i],
                                          textual_semantics_embedding[i], visual_semantics_embedding[i])
            p_weights, n_weights = tf.split(weights, [self.batch_size, self.num_neg * self.batch_size], 0)
            textual_trans = textual_semantics_embedding[i]
            p_textual_trans, n_textual_trans = tf.split(textual_trans,
                                                        [self.batch_size, self.num_neg * self.batch_size], 0)
            visual_trans = visual_semantics_embedding[i]
            p_visual_trans, n_visual_trans = tf.split(visual_trans, [self.batch_size, self.num_neg * self.batch_size],
                                                      0)

            p_score = p_weights[:, 1] * tf.nn.softplus(tf.reduce_sum(tf.multiply(user_semantics_embedding_ap[i],
                                                                                 p_textual_trans), 1)) + p_weights[:,
                                                                                                         2] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(user_semantics_embedding_ap[i], p_visual_trans), 1)) + p_weights[:,
                                                                                              0] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(
                    user_semantics_embedding_ap[i], p_item_semantics_embedding[i]), 1))

            pos_scores.append(tf.expand_dims(p_score, 1))

            n_score = n_weights[:, 1] * tf.nn.softplus(tf.reduce_sum(tf.multiply(user_semantics_embedding_an[i],
                                                                                 n_textual_trans), 1)) + n_weights[:,
                                                                                                         2] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(user_semantics_embedding_an[i], n_visual_trans), 1)) + n_weights[:,
                                                                                              0] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(
                    user_semantics_embedding_an[i], n_item_semantics_embedding[i]), 1))

            neg_scores.append(tf.expand_dims(n_score, 1))

        pos_s = tf.concat(pos_scores, 1)
        neg_s = tf.concat(neg_scores, 1)
        regularizer += tf.norm(users, ord=1) + tf.norm(pos_items, ord=1) + tf.norm(neg_items, ord=1) + tf.norm(pos_i_v,
                                                                                                               ord=1) + tf.norm(
            neg_i_v, ord=1) + tf.norm(pos_i_f, ord=1) + tf.norm(neg_i_f, ord=1)

        regularizer = regularizer / self.batch_size

        pos_score = tf.reduce_sum(pos_s, 1, name="pos")

        negtive_score = tf.reduce_max(tf.reshape(tf.reduce_sum(neg_s, 1), [self.batch_size, self.num_neg]), 1)

        loss_per_pair = tf.nn.softplus(-(pos_score - negtive_score))

        loss = tf.reduce_sum(loss_per_pair, name="loss")


        return loss + self.decay_r * regularizer + self.decay_d * dis_loss

    def create_weight(self, user, item, textual, visual):
        skip = self.sikp_connect
        input = tf.nn.l2_normalize(tf.concat([user, item, textual, visual], 1), 1)
        input_o = input
        input = tf.expand_dims(input, 1)
        input = tf.expand_dims(input, -1)
        input_conv = tf.layers.conv2d(inputs=input, filters=32, kernel_size=(1, 1), activation=tf.nn.sigmoid,
                                      name="weight_conv", reuse=tf.AUTO_REUSE)
        input_sc = self.ScConv(input_conv)
        input_la = tf.reduce_mean(input_sc, axis=[1, 3])
        input_lb = skip*input_o+(1-skip)*input_la
        output_h = tf.layers.dense(inputs=input_lb, units=3, activation=tf.nn.tanh, name="weight_h", reuse=tf.AUTO_REUSE)
        output = tf.layers.dense(inputs=output_h, units=3, activation=None, use_bias=None, name="weight_o",
                                 reuse=tf.AUTO_REUSE)
        weight = tf.nn.softmax(output, 1)
        return weight


    @define_range
    def visual_projection(self):
        """
        return: Projection of visual feature vectors to user-item embedding
        """
        mlp_layer_1 = tf.layers.dense(
            inputs=tf.nn.l2_normalize(self.visualfeatures, 1),
            units=2 * self.hidden_dim_a,
            activation=tf.nn.leaky_relu, name="mlp_layer_v1")
        dropout = tf.layers.dropout(mlp_layer_1, self.dropout_a)
        output = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout, 1), activation=None, units=self.embed_dim,
                                 reuse=tf.AUTO_REUSE, name="mlp_layer_v4")
        return output

    @define_range
    def textual_projection(self):
        """
        return: Projection of text feature vectors to user-item embedding
        """
        mlp_layer_1 = tf.layers.dense(
            inputs=tf.nn.l2_normalize(self.textualfeatures, 1),
            units=2 * self.hidden_dim_b,
            activation=tf.nn.leaky_relu, name="mlp_layer_t1")
        dropout = tf.layers.dropout(mlp_layer_1, self.dropout_a)
        output = tf.layers.dense(inputs=tf.nn.l2_normalize(dropout, 1), activation=None, units=self.embed_dim,
                                 reuse=tf.AUTO_REUSE, name="mlp_layer_t4")
        return output

    def distance_correlation(self, X1, X2):
        '''
        Calculate the distance between blocks of the vector.
        '''

        def centered_distance(X):

            # calculate the pairwise distance of X
            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        def distance_covariance(E1, E2):

            # calculate distance covariance between E1 and E2
            n_samples = tf.cast(tf.shape(E1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(E1 * E2) / (n_samples * n_samples), 0.0) + 1e-8)
            return dcov

        E1 = centered_distance(X1)
        E2 = centered_distance(X2)

        dcov_12 = distance_covariance(E1, E2)
        dcov_11 = distance_covariance(E1, E1)
        dcov_22 = distance_covariance(E2, E2)

        # calculate the distance correlation
        dccor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)

        return dccor

    @define_range
    def loss(self):
        loss = self.emb_loss
        return loss

    @define_range
    def clip_by_norm_op(self):
        return [tf.assign(self.user_emb, tf.clip_by_norm(self.user_emb, 1.0, axes=[1])),
                tf.assign(self.item_emb, tf.clip_by_norm(self.item_emb, 1.0, axes=[1]))]

    @define_range
    def optimize(self):

        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.user_emb,
                                                                                               self.item_emb])

    @define_range
    def item_scores(self):
        # (n_users, 1, K)
        users = tf.expand_dims(tf.nn.embedding_lookup(self.user_emb, self.user_ids), 1)

        # (1, n_items, K)
        item = tf.expand_dims(self.item_emb, 0)
        textual = tf.expand_dims(self.textual_projection, 0)
        visual = tf.expand_dims(self.visual_projection, 0)

        item_expand = tf.reshape(tf.tile(item, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])
        textual_expand = tf.reshape(tf.tile(textual, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])
        visual_expand = tf.reshape(tf.tile(visual, [tf.shape(users)[0], 1, 1]), [-1, self.embed_dim])

        users_expand = tf.reshape(tf.tile(users, [1, tf.shape(item)[1], 1]), [-1, self.embed_dim])

        user_expand_semantics_embedding = tf.split(users_expand, self.n_semantics, 1)
        item_expand_semantics_embedding = tf.split(item_expand, self.n_semantics, 1)

        textual_expand_semantics_embedding = tf.split(textual_expand, self.n_semantics, 1)
        visual_expand_semantics_embedding = tf.split(visual_expand, self.n_semantics, 1)

        semantics_scores = []
        semantics_sc = []
        semantics_ws = []
        for i in range(0, self.n_semantics):
            weights = self.create_weight(user_expand_semantics_embedding[i], item_expand_semantics_embedding[i],
                                          textual_expand_semantics_embedding[i], visual_expand_semantics_embedding[i])
            textual_trans = textual_expand_semantics_embedding[i]
            visual_trans = visual_expand_semantics_embedding[i]
            f_score = weights[:, 1] * tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_semantics_embedding[i],
                                                                               textual_trans), 1)) + weights[:,
                                                                                                     2] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(user_expand_semantics_embedding[i], visual_trans), 1)) + weights[:,
                                                                                                0] * tf.nn.softplus(
                tf.reduce_sum(tf.multiply(
                    user_expand_semantics_embedding[i], item_expand_semantics_embedding[i]), 1))
            semantics_scores.append(tf.expand_dims(f_score, 1))
            semantics_sc.append([weights[:, 0] * tf.nn.softplus(tf.reduce_sum(tf.multiply(
                user_expand_semantics_embedding[i], item_expand_semantics_embedding[i]), 1)),
                              weights[:, 1] * tf.nn.softplus(tf.reduce_sum(tf.multiply(user_expand_semantics_embedding[i],
                                                                                       textual_trans), 1)),
                              weights[:, 2] * tf.nn.softplus(
                                  tf.reduce_sum(tf.multiply(user_expand_semantics_embedding[i], visual_trans), 1))])
            semantics_ws.append(weights)

        semantics_s = tf.concat(semantics_scores, 1)
        scores = tf.reshape(tf.reduce_sum(semantics_s, axis=1), [tf.shape(users)[0], -1])
        return scores, semantics_sc, semantics_ws


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def optimize(model, sampler, train, train_num, test):
    """
    Optimize the model.
    param:
        model: model to optimize
        sampler: mini-batch sampler
        train: train user-item matrix
        test: test user-item matrix
    return: None
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    test_users = np.asarray(list(set(test.nonzero()[0])), dtype=np.int32)
    evaluate_every_n_batches = train_num // args.batch_size + 1
    cur_best_pre_0 = 0.
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    stopping_step = 0

    for epoch in range(args.epochs):
        t1 = time()

        # train model
        losses = 0
        for _ in range(10 * evaluate_every_n_batches):
            user_pos, neg = sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg})
            losses += loss
        t2 = time()

        testresult = Evaluator(model, train, test)
        test_recalls = []
        test_ndcg = []
        test_hr = []
        test_pr = []

        for user_chunk in toolz.partition_all(20, test_users):
            recalls, ndcgs, hit_ratios, precisions = testresult.eval(sess, user_chunk)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)

        recalls = sum(test_recalls) / float(len(test_recalls))
        precisions = sum(test_pr) / float(len(test_pr))
        hit_ratios = sum(test_hr) / float(len(test_hr))
        ndcgs = sum(test_ndcg) / float(len(test_ndcg))

        rec_loger.append(recalls)
        pre_loger.append(precisions)
        ndcg_loger.append(ndcgs)
        hit_loger.append(hit_ratios)

        t3 = time()
        print("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (
        epoch, t2 - t1, t3 - t2, losses / (10 * evaluate_every_n_batches), recalls, precisions, hit_ratios, ndcgs))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(recalls, cur_best_pre_0, stopping_step,
                                                                    expected_order='acc', flag_step=10)

        if should_stop == True:
            sampler.close()
            break
        if epoch == args.epochs - 1:
            sampler.close()

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs)
    idx = list(recs).index(best_rec_0)
    final_perf = "Best Iter = recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (recs[idx], pres[idx], hit[idx], ndcgs[idx])
    print(final_perf)


def parse_args():
    parser = argparse.ArgumentParser(description='Run AdaMRec.')
    parser.add_argument('--dataset', nargs='?', default='Office', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='total_epochs')
    parser.add_argument('--gpu', nargs='?', default='1', help='gpu_id')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate.')
    parser.add_argument('--decay_r', type=float, default=1e-2, help='decay_r.')
    parser.add_argument('--decay_d', type=float, default=1e-0, help='decay_d.')
    parser.add_argument('--decay_p', type=float, default=0, help='decay_p.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--n_semantics', type=int, default=4, help='Number of semantics.')
    parser.add_argument('--num_neg', type=int, default=4, help='negative items')
    parser.add_argument('--hidden_dim_a', type=int, default=256, help='Hidden layer dim a.')
    parser.add_argument('--hidden_dim_b', type=int, default=128, help='Hidden layer dim b.')
    parser.add_argument('--dropout_a', type=float, default=0.2, help='dropout_a.')
    parser.add_argument('--dropout_b', type=float, default=0.2, help='dropout_b.')
    parser.add_argument('--sikp_connect', type=float, default=0.2, help='sikp_connect.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    Filename = args.dataset
    Filepath = 'Data/' + Filename
    # get train/valid/test user-item matrices
    data = Data.Data(Filepath)
    train, test = data.trainMatrix, data.testRatings
    # make feature as dense matrix
    textualfeatures, visualfeatures = data.textualfeatures, data.visualfeatures
    n_users, n_items = max(train.shape[0], test.shape[0]), max(train.shape[1], test.shape[1])
    train_num = data.train_num
    # create sampler
    sampler = WarpSampler(train, batch_size=args.batch_size, n_negative=args.num_neg, check_negative=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = AdaMRec(n_users,
                 n_items,
                 visualfeatures=visualfeatures,
                 textualfeatures=textualfeatures,
                 embed_dim=128,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 hidden_dim_a=args.hidden_dim_a,
                 hidden_dim_b=args.hidden_dim_b,
                 decay_r=args.decay_r,
                 decay_d=args.decay_d,
                 dropout_a=args.dropout_a,
                 dropout_b=args.dropout_b,
                 sikp_connect=args.sikp_connect,
                 )

    optimize(model, sampler, train, train_num, test)
