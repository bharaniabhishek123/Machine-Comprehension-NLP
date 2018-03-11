# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        print "hidden_size", hidden_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):

            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            print "inputs lenght", input_lens

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            print "inputs", inputs.shape
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            print "fw_out:", fw_out
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class BiRNNChar(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks,scope="BiRNNChar"):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(scope,reuse=False):

            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            print "inputs lenght", input_lens

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size). ? 600 8 / ? 30 8
            print "inputs", inputs.shape
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            print "fw_out:", fw_out
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out



class BiRNN(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiRNN"):

            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            print "inputs lenght", input_lens

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            print "inputs", inputs.shape
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            print "fw_out:", fw_out
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class Rnetchar(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int

        """
        # self.hidden_size = hidden_size

        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        Context attend to Query.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("Rnetchar"):
            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1])  # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t)  # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask,2)   # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values)  # shape (batch_size, num_keys(600), value_vec_size(400))

            # Apply dropout ? 600 600 v
            output = tf.nn.dropout(output, self.keep_prob)

            print "Output Shape", output.shape

            T = keys.get_shape().as_list()[1]   #600 for our data #output = v

            W1 = tf.get_variable("W1", shape=[self.value_vec_size, 1], initializer=tf.contrib.layers.xavier_initializer())

            W2 = tf.get_variable("W2", shape=[self.value_vec_size, 1], initializer=tf.contrib.layers.xavier_initializer())

            v_mat = tf.get_variable("v_mat", shape=[T, T], initializer=tf.contrib.layers.xavier_initializer())

            output1 = tf.reshape(output, [-1, self.value_vec_size ])   #value vec size = 2*hidden=400

            part1 = tf.matmul(output1, W1)

            # self.part1_after_matmul = part1

            print "Part1",part1.shape # ?, 1

            part1 = tf.reshape(part1,[-1,T]) #batchsize X 600

            # self.part1_after_reshape = part1

            print "After reshaping Part1", part1.shape  # ?, 600
            part1_ex = tf.expand_dims(part1, 2)  # ? , 600, 1
            print "part1 expansion", part1_ex.shape

            # self.part1_after_ex = part1_ex

            part1_tile = tf.layers.dense(part1_ex, T, activation=None) # ? , 600 , 600  : square matrix
            # part1_tile1 = tf.layers.dense(part1_tile, T, activation=None)
            print "After tiling Part1", part1_tile.shape  # ?, 600

            # self.part1_after_tile = part1_tile


            part2 = tf.matmul(output1, W2)
            print "Part2", part2.shape # ?, 1

            part2 = tf.reshape(part2,[-1,T])
            print "After reshaping Part2", part2.shape  # ?, 600

            part2_ex = tf.expand_dims(part2, 2)
            print "part2 expansion", part2_ex.shape
            part2_tile = tf.layers.dense(part2_ex, T, activation=None)
            # part1_tile1 = tf.layers.dense(part1_tile, T, activation=None)
            print "After tiling Part2", part2_tile.shape  # ?, 600

            part2_tile_t = tf.transpose(part2_tile,[0,2,1]) # ?, 600 , 600 ( ? , context ,  context)
            print "part2 transpose", part2_tile_t.shape

            # part = part1_tile + part2_tile_t  #()
            # print "part shape", part.shape

            part_tanh = tf.tanh(tf.add(part1_tile, part2_tile_t))
            print "part tanh shape", part_tanh.shape
            e = self.mat_weight_mul(part_tanh, v_mat)
            # P_ones = tf.ones(shape=[T, T])

            # BSize = keys.get_shape().as_list()[0]

            attn_logits_mask_keys = tf.expand_dims(keys_mask, 1)  # shape (batch_size, key_values,1)
            print " shape of key mask", keys_mask.shape
            print "shape of attn_logits_mask_keys", attn_logits_mask_keys.shape

            self.e = e
            self.attn_logits_mask_keys = attn_logits_mask_keys

            _, alpha = masked_softmax(e, attn_logits_mask_keys, 2)

            # part = tf.tanh(part1+part2)
            print "alpha shape", alpha.shape
            # alpha = tf.reshape(alpha,[-1,T,])
            #
            # W1 = tf.get_variable("W1", shape=[self.value_vec_size, self.value_vec_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())

            self.alpha = alpha
            self.output = output
            a_i = tf.matmul(alpha,output)
            print "shape of a_i", a_i.shape
            print "shape of output", output.shape

            return a_i, output


    def mat_weight_mul(self, mat, weight):
            # [batch_size, n, m] * [m, p] = [batch_size, n, p]
            mat_shape = mat.get_shape().as_list()
            weight_shape = weight.get_shape().as_list()
            assert(mat_shape[-1] == weight_shape[0])
            mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
            mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
            return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])





class Rnet(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int

        """
        # self.hidden_size = hidden_size

        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        Context attend to Query.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("Rnet"):
            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1])  # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t)  # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask,
                                          2)  # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values)  # shape (batch_size, num_keys(600), value_vec_size(400))

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            print "Output Shape",output.shape
            T = keys.get_shape().as_list()[1]   #600 for our data
            #output = v

            W1 = tf.get_variable("W1", shape=[self.value_vec_size, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())

            W2 = tf.get_variable("W2", shape=[self.value_vec_size, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            v_mat = tf.get_variable("v_mat",shape=[T, T],initializer=tf.contrib.layers.xavier_initializer())

            ##### Start of RNet logic - 2#################
            # v = tf.reshape(output,[batch_size,T,self.value_vec_size])
            # v_matrix = tf.get_variable("v_m", shape=[T, self.value_vec_size],
            #                         initializer=tf.contrib.layers.xavier_initializer())
            # W_g_SM = tf.get_variable("w_g_SM",shape=[self.value_vec_size * 2, self.value_vec_size * 2],initializer=tf.contrib.layers.xavier_initializer())
            # SM_star = []
            # for t in range(T):
            #     # Calculate s_t
            #     print "t", t
            #     W_p1_v_P = self.mat_weight_mul(v, W1)  # [batch_size, p_length, state_size]
            #     # print "W_p1_v_P", W_p1_v_P.shape
            #     # print "output shape", output.shape
            #     tiled_v_tP = tf.concat([tf.reshape(W_p1_v_P[:, t, :], [batch_size, 1, -1])] * T, 1)
            #     # print "tiled_v_tP", tiled_v_tP.shape
            #     W_p2_v_tP = self.mat_weight_mul(tiled_v_tP, W2)
            #     # print "W_p2_v_tP",W_p2_v_tP.shape
            #     tanh = tf.tanh(W_p1_v_P + W_p2_v_tP)
            #     # print "tanh shape", tanh.shape
            #     s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(v_matrix, [-1, 1])))
            #     # print "s_t shape", s_t.shape
            #     alpha = tf.nn.softmax(s_t, 1)
            #     # print "alpha shape", alpha.shape
            #     a_t = tf.concat([tf.reshape(alpha, [batch_size, -1, 1])] * self.key_vec_size, 2)
            #     # print "a_t shape",a_t.shape
            #     # print "v shape", v.shape
            #     c_t = tf.reduce_sum(tf.multiply(a_t, v),1)  # [batch_size, 2 * state_size]
            #     # print "c_t shape", c_t.shape
            #     # gate
            #     v_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(v[:, t, :]), c_t], 1), 1)
            #     # print "shape of v_tP_c_t", v_tP_c_t.shape
            #     # print "shape W_g_SM", W_g_SM.shape
            #     g_t = tf.sigmoid(self.mat_weight_mul(v_tP_c_t, W_g_SM))
            #     v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))
            #     SM_star.append(v_tP_c_t_star)
            # SM_star = tf.stack(SM_star, 1)
            # # unstacked_SM_star = tf.unstack(SM_star, T, 1)
            # # with tf.variable_scope('Self_match') as scope:
            # #     # SM_fw_cell = self.DropoutWrapper(self.key_vec_size, self.keep_prob)
            # #     # SM_bw_cell = self.DropoutWrapper(self.key_vec_size, self.keep_prob)
            # #     sm_outputs, sm_final_fw, sm_final_bw = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell,
            # #                                                                                    unstacked_SM_star,
            # #                                                                                    dtype=tf.float32)
            # #     h_P = tf.stack(sm_outputs, 1)
            # # h_P = tf.nn.dropout(h_P, self.keep_prob)
            # h_P = SM_star
            # print "h_P", h_P, tf.shape(h_P)
            # print "SM_Star shape", SM_star
            # return h_P
            # ##########Rnet Logic 2 - end###############


            output1 = tf.reshape(output, [-1, self.value_vec_size ])   #value vec size = 2*hidden=400
            part1 = tf.matmul(output1, W1)
            print "Part1",part1.shape # ?, 1

            part1 = tf.reshape(part1,[-1,T]) #batchsize X 600
            print "After reshaping Part1", part1.shape  # ?, 600
            part1_ex = tf.expand_dims(part1, 2)  # ? , 600, 1
            print "part1 expansion", part1_ex.shape
            part1_tile = tf.layers.dense(part1_ex, T, activation=None) # ? , 600 , 600  : square matrix
            # part1_tile1 = tf.layers.dense(part1_tile, T, activation=None)
            print "After tiling Part1", part1_tile.shape  # ?, 600


            part2 = tf.matmul(output1, W2)
            print "Part2", part2.shape # ?, 1

            part2 = tf.reshape(part2,[-1,T])
            print "After reshaping Part2", part2.shape  # ?, 600

            part2_ex = tf.expand_dims(part2, 2)
            print "part2 expansion", part2_ex.shape
            part2_tile = tf.layers.dense(part2_ex, T, activation=None)
            # part1_tile1 = tf.layers.dense(part1_tile, T, activation=None)
            print "After tiling Part2", part2_tile.shape  # ?, 600

            part2_tile_t = tf.transpose(part2_tile,[0,2,1]) # ?, 600 , 600 ( ? , context ,  context)
            print "part2 transpose", part2_tile_t.shape

            # part = part1_tile + part2_tile_t  #()
            # print "part shape", part.shape

            part_tanh = tf.tanh(tf.add(part1_tile, part2_tile_t))
            print "part tanh shape", part_tanh.shape
            e = self.mat_weight_mul(part_tanh, v_mat)
            # P_ones = tf.ones(shape=[T, T])

            # BSize = keys.get_shape().as_list()[0]

            # for i in range(BSize):
            # for i, (output, W1) in enumerate(zip(output, W1)):
            # # for i in range(batch_size):
            #     print "in for zip"
            #     print "part1 i shape", part1[i].shape
            #     # print "part2 i shape", part2[i].shape
            #     # part1_e = tf.multiply(P_ones, part1_i)
            #     # part2_e = tf.multiply(P_ones, part2_i)
            #
            #     part1_e = tf.multiply(P_ones, part1[i])
            #     part2_e = tf.multiply(P_ones, part2[i])
            #
            #     # part2_tile = tf.transpose(part2[i],perm=[1,0])*P_ones
            #     # print "part1_e after expand shape", part1_e.shape
            #     # print "part2_e after expand shape", part2_e.shape
            #     # print "part1 i shape after tile", part1_tile.shape
            #
            #     part = tf.tanh(tf.add(part1_e,tf.transpose(part2_e)))
            #
            #     # print "part after add shape", part.shape
            #
            #     e_temp = tf.matmul(part, v)   #
            #
            #     # print "e_temp shape", e_temp.shape
            #
            #     e_temp_t = tf.transpose(e_temp,perm=[1,0])
            #
            #     # print "shape after transpose e_temp_t", e_temp_t.shape
            #     # e_temp_expand = tf.expand_dims(e_temp, 0)  # shape (batch_size, key_values, value_vec_size)
            #     if (i == 0):
            #         e = e_temp_t
            #         print "first e", e
            #     else:
            #         e=tf.concat([e, e_temp_t],axis=0)
            #
            #     # e = e + [e_temp]
            #
            #
            #     # print "in loop e", e.shape  #  600, 600
            # print "after for loop e shape", e.shape
            # e = e[1:]
            # print "after removing 1st row in e shape", e.shape
            # e = tf.reshape(e,[-1,T,T])
            # print "after reshape    `    e shape", e.shape
            attn_logits_mask_keys = tf.expand_dims(keys_mask, 1)  # shape (batch_size, key_values,1)
            print " shape of key mask", keys_mask.shape
            print "shape of attn_logits_mask_keys", attn_logits_mask_keys.shape
            _, alpha = masked_softmax(e, attn_logits_mask_keys, 2)

            # part = tf.tanh(part1+part2)
            print "alpha shape", alpha.shape
            # alpha = tf.reshape(alpha,[-1,T,])
            #
            # W1 = tf.get_variable("W1", shape=[self.value_vec_size, self.value_vec_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            a_i = tf.matmul(alpha,output)
            print "shape of a_i", a_i.shape
            print "shape of output", output.shape

            return a_i, output


            # output = tf.reshape(output,[-1,self.value_vec_size])
            #
            # W1 = tf.get_variable("W1",shape=[self.value_vec_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            # part1 = tf.matmul(output,W1)
            # print "part1",part1.shape
            #
            #
            # J = values.get_shape().as_list()[1]
            #
            # W2 = tf.get_variable("W2", shape=[self.value_vec_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            #
            # part2 = tf.matmul(output,W2)
            # print "part2",part2.shape
            #
            # p = part1 + part2

    def mat_weight_mul(self, mat, weight):
            # [batch_size, n, m] * [m, p] = [batch_size, n, p]
            mat_shape = mat.get_shape().as_list()
            weight_shape = weight.get_shape().as_list()
            assert(mat_shape[-1] == weight_shape[0])
            mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
            mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
            return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    # if (logits.get_shape()[2] != mask.get_shape()[2]): # If part is for sentinel handling in CoAttn
    #     sequence_length = tf.shape(mask)[2]
    #     score_mask = tf.sequence_mask([sequence_length], maxlen=tf.shape(logits)[1])
    #     score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(logits)[2]))
    #     affinity_mask_value = float('-inf')
    #     affinity_mask_values = affinity_mask_value * tf.ones_like(logits)
    #     masked_logits = tf.where(score_mask, logits, affinity_mask_values)
    #     prob_dist = tf.nn.softmax(masked_logits, dim)
    #
    # else :
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


class BiDAF(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAF"):
            #values are my question hidden state : so will have dimension of (batchsize , 30 (question len) , 400 (2*hidden))
            # Calculate attention distribution

            # num_values = question_len = 30
            # value_vec_size = 400
            # values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)

            values_ex = tf.expand_dims(values, 1)  # shape (batch_size, 1 , J, 2* hidden_size)
            T = keys.get_shape().as_list()[1]
            # values_ex = tf.tile(values_ex,[1,T,1,1]) # removed due to performance issues

            values_ex = tf.layers.dense(values_ex, self.value_vec_size, activation=None)

            #keys are my context hidden state : so will have dimension of (batchsize , 600 (context len) , 400 (2*hidden))
            # num_keys = context_len = 600
            # value_vec_size = 400

            keys_ex = tf.expand_dims(keys, 2)  # shape (batch_size, T , 1, 2* hidden_size)
            J = values.get_shape().as_list()[1]

            # keys_ex = tf.tile(keys_ex,[1,1,J,1]) # removed due to performance issues


            keys_ex = tf.layers.dense(keys_ex, self.key_vec_size, activation=None)

            temp = tf.multiply(values_ex, keys_ex)  # Performance Changes

            # temp = tf.einsum('bjh,bth->btj', values,keys) # Changed on 03062018 for performance issues on multiply
            # print "Shape of ElementWise temp",temp.shape

            W_sim = tf.get_variable("W_sim",shape=[self.value_vec_size,1], initializer=tf.contrib.layers.xavier_initializer())

            # print "Shape of keys_ex", keys_ex.shape

            keys_ex = tf.reshape(keys_ex, [-1, self.key_vec_size])

            pk = tf.matmul(keys_ex,W_sim)

            pk = tf.reshape(pk,[-1,T,1])

            # print "Shape of pk",pk.shape

            values_ex = tf.reshape(values_ex, [-1, self.value_vec_size])

            pv = tf.matmul(values_ex, W_sim)

            pv = tf.reshape(pv, [-1, 1, J])

            # print "Shape of pv",pv.shape

            temp = tf.reshape(temp, [-1, self.key_vec_size])

            pd = tf.matmul(temp, W_sim)

            pd = tf.reshape(pd, [-1, T, J])

            # print "Shape of pd", pd.shape

            # pd = tf.matmul(W_sim,temp)

            attn_logits_C2Q = pk + pv + pd
            # print "Shape of attn_logits_C2Q", attn_logits_C2Q.shape
            # concat_C2Q = tf.concat([keys_ex,values_ex,temp],3) # (batch_Size, T 600 , J 30 , 1200)
            # print "Shape of concat concat_C2Q",concat_C2Q.shape

            # concat_C2Q = tf.reshape(concat_C2Q, [-1, self.key_vec_size * 3])


            # Similarity_Temp = tf.matmul(concat_C2Q,W_sim)

            # print "Shape of Similarity_Matrix" , Similarity_Temp.shape

            # attn_logits_C2Q = tf.reshape(Similarity_Temp,[-1,T,J])

            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist_C2Q = masked_softmax(attn_logits_C2Q, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output_C2Q = tf.matmul(attn_dist_C2Q, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output_C2Q = tf.nn.dropout(output_C2Q, self.keep_prob)

            attn_logits_Q2C_ = tf.reduce_max(attn_logits_C2Q,axis=2)

            attn_logits_Q2C = tf.expand_dims(attn_logits_Q2C_,1)
            attn_logits_mask_key = tf.expand_dims(keys_mask, 1) # shape (batch_size, 1, 600)
            _, attn_dist_Q2C = masked_softmax(attn_logits_Q2C, attn_logits_mask_key, 2)
            # print "attn_dist_Q2C",attn_dist_Q2C.shape
            # Use attention distribution to take weighted sum of values
            output_Q2C = tf.matmul(attn_dist_Q2C, keys)  # shape (batch_size, num_keys, value_vec_size)
            # print "output_Q2C",output_Q2C.shape
            # Apply dropout
            output_Q2C = tf.nn.dropout(output_Q2C, self.keep_prob)
            # print "output_C2Q",output_C2Q.shape
            return output_C2Q, output_Q2C



class BiLSTM(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code1 uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("BiLSTM"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class CoAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("CoAttn"):

            # First, apply a linear layer with tanh nonlinearity to the question hidden states to obtain projected question hidden states
            # q01 ;:::; q0M:

            # W = tf.get_variable("W", shape=[self.key_vec_size,self.key_vec_size] , initializer=tf.contrib.layers.xavier_initializer())
            #
            # J = values.get_shape().as_list()[1]
            # b = tf.get_variable("b", shape=[self.key_vec_size,J+1], initializer=tf.constant_initializer(0.0),dtype=tf.float32 )
            # q_dash = tf.tanh(tf.matmul(W,values)+b)

            q_dash = tf.layers.dense(values,values.get_shape()[2],activation=tf.tanh)

            # Adding Sentinel Vector to context hidden
            c_phi = tf.get_variable("c_phi", shape=keys.get_shape()[2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            c_phi = tf.reshape(c_phi, [1, 1, -1])
            c_phi = tf.tile(c_phi, (tf.shape(keys)[0], 1, 1))
            keys = tf.concat([c_phi, keys], 1)

            # Adding Sentinel Vector to question hidden
            q_dash_phi = tf.get_variable("q_dash_phi",shape=values.get_shape()[2],initializer=tf.contrib.layers.xavier_initializer())
            q_dash_phi = tf.reshape(q_dash_phi, [1, 1, -1])
            q_dash_phi = tf.tile(q_dash_phi, (tf.shape(values)[0], 1, 1))
            values = tf.concat([q_dash_phi, values], 1)

            #Affinity Matrix L
            Affin_mat = tf.einsum('bth,bjh->btj',keys,values)


            attn_logits_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, num_values(ques))

            unit_value = tf.get_variable("sentinelOffset", shape=[1], initializer=tf.constant_initializer(1), dtype=tf.int32)
            unit_value = tf.reshape(unit_value,[1,1,-1])
            unit_value = tf.tile(unit_value,(tf.shape(values_mask)[0],1,1))

            attn_logits_mask = tf.concat([unit_value,attn_logits_mask],2)

            _, attn_dist_c2q = masked_softmax(Affin_mat, attn_logits_mask, 2) # shape (batch_size, num_keys+1, num_values+1)
                                                                              # . take softmax over dim=2 values(ques)

            output_c2q = tf.matmul(attn_dist_c2q, values) # a = alpha * q

            Affin_mat_t = tf.transpose(Affin_mat, [0, 2, 1])  # [N, Q, D] or [N, 1+Q, 1+D] if sentinel

            attn_logits_mask_key = tf.expand_dims(keys_mask, 1)

            attn_logits_mask_key = tf.concat([unit_value,attn_logits_mask_key],2)

            _, attn_dist_q2c = masked_softmax(Affin_mat_t, attn_logits_mask_key,2)

            output_q2c = tf.matmul(attn_dist_q2c, keys) # b = beta * c

            # Second level attention
            sec_lvl_attn = tf.matmul(attn_dist_c2q, output_q2c) # s = alpha * b

            feed_to_biLSTM = tf.concat([output_c2q, sec_lvl_attn],1)

            cell_fw = tf.nn.rnn_cell.LSTMCell(self.key_vec_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.key_vec_size)
            # compute coattention encoding
            u, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, sec_lvl_attn,
                dtype=tf.float32)

            # values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            # attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            # attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            # _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            # output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            # output = tf.nn.dropout(output, self.keep_prob)

            return u