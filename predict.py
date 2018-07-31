import tensorflow as tf
import os
import codecs
import time
import numpy as np
import tensorflow as tf
import data_util
import pickle
import random
import similarity
from biLSTM import BiLSTM

# Parameters
# ==================================================
# data loading params
tf.flags.DEFINE_string("train_file", "data/train_factoid_1.json", "Training data.")
tf.flags.DEFINE_string("val_file", "data/valid_factoid.json", "Validation data.")
tf.flags.DEFINE_string("stop_words_file", "data/stop_words.txt", "Stop words.")

# result & model save params
tf.flags.DEFINE_string("result_file", "res/predictRst.score", "Predict result.")
tf.flags.DEFINE_string("save_file", "res/savedModel", "Save model.")

# pre-trained word embedding vectors
tf.flags.DEFINE_string("embedding_file", "data/zhwiki_2017_03.sg_50d.word2vec", "Embedding vectors.")

# hyperparameters
tf.flags.DEFINE_integer("k", 5, "K most similarity knowledge (default: 5).")
tf.flags.DEFINE_integer("k_neg", 5, "K negative samples (default: 5).")
tf.flags.DEFINE_integer("rnn_size", 100, "Neurons number of hidden layer in LSTM cell (default: 100).")
tf.flags.DEFINE_float("margin", 0.2, "Constant of max-margin loss (default: 0.1).")
tf.flags.DEFINE_integer("max_grad_norm", 5, "Control gradient expansion (default: 5).")
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50).")
tf.flags.DEFINE_integer("max_sentence_len", 200, "Maximum number of words in a sentence (default: 100).")
tf.flags.DEFINE_float("dropout_keep_prob", 0.45, "Dropout keep probability (default: 0.5).")
tf.flags.DEFINE_float("learning_rate", 0.4, "Learning rate (default: 0.4).")
tf.flags.DEFINE_float("lr_down_rate", 0.5, "Learning rate down rate(default: 0.5).")
tf.flags.DEFINE_integer("lr_down_times", 4, "Learning rate down times (default: 4)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 20, "Number of checkpoints to store (default: 5)")

# gpu parameters
tf.flags.DEFINE_float("gpu_mem_usage", 0.75, "GPU memory max usage rate (default: 0.75).")
tf.flags.DEFINE_string("gpu_device", "/gpu:1", "GPU device name.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

print("loading embedding...")
embedding, word2idx = data_util.load_embedding(FLAGS.embedding_file)

val_data_path = "tmp/val_data"
if os.path.exists(val_data_path):
    with open(val_data_path, "rb") as f:
        val_data_dict = pickle.load(f)
else:
    val_questions, val_answers, val_knowledges = \
        data_util.load_data_index(FLAGS.val_file, FLAGS.stop_words_file, word2idx, FLAGS.max_sentence_len,\
                            FLAGS.max_sentence_len, FLAGS.max_sentence_len)
    val_data_dict = {'val_questions': val_questions,
                  'val_answers': val_answers,
                  'val_knowledges': val_knowledges}

    with open(val_data_path, "wb") as f:
        pickle.dump(val_data_dict, f)

val_questions, val_answers, val_knowledges = val_data_dict['val_questions'],\
                                             val_data_dict['val_answers'],\
                                             val_data_dict['val_knowledges']


def predictEveryAnswer(knowledge, word2idx, question_num, max_len):
    all_knowledges, all_answers = [], []
    for i in range(question_num):
        for word in knowledge:
            answer = [word2idx.get('UNKNOWN', 0)] * max_len
            answer[0] = word
            all_answers.append(answer)
            all_knowledges.append(knowledge[i])
    return all_knowledges, all_answers

with tf.Graph().as_default(), tf.device('/cpu:0'):

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_mem_usage
    )

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        gpu_options=gpu_options
    )

    with tf.Session(config=session_conf) as sess:
        globalStep = tf.Variable(0, name="globle_step", trainable=False)
        lstm = BiLSTM(
            FLAGS.batch_size,
            FLAGS.max_sentence_len,
            embedding,
            FLAGS.embedding_dim,
            FLAGS.rnn_size,
            FLAGS.margin
        )
        saver = tf.train.Saver()
        saver.restore(sess, "res/.meta")
        print("Model restored.")
        question_num = len(val_questions)
        epoch_num = 20
        mask_percent = 0.1
        for epoch in range(epoch_num):
            # 500 samples per time
            scores = []
            mask = [random.random() < mask_percent for i in range(question_num)]
            val_Q_batch = [val_questions[i] for i in range(question_num) if mask[i]]
            val_K_batch = [val_knowledges[i] for i in range(question_num) if mask[i]]
            val_A_batch = [val_answers[i] for i in range(question_num) if mask[i]]
            val_q, val_a = predictEveryAnswer(val_K_batch, word2idx, len(val_K_batch), \
                                              FLAGS.max_sentence_len)
            val_feed_dict = {
                lstm.inputTestQuestions: val_q,
                lstm.inputTestAnswers: val_a,
                lstm.dropout_keep_prob: 1.0
            }
            _, score = sess.run([globalStep, lstm.result], val_feed_dict)
            scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for val_id in range(len(val_K_batch)):
                offset = val_id * FLAGS.max_sentence_len
                predict_true_ix = np.argmax(scores[offset:offset + FLAGS.max_sentence_len])
                if val_a[offset + predict_true_ix] == val_answers[val_id]:  # every k_neg answers, the first answer is true
                    cnt += 1
            print("evaluation acc: ", cnt / len(val_K_batch))
