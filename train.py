import os
import codecs
import time
import numpy as np
import tensorflow as tf
import data_util
import pickle
import similarity
from biLSTM import BiLSTM

# time stamp
start = time.time()
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
tf.flags.DEFINE_integer("max_sentence_len", 20, "Maximum number of words in a sentence (default: 100).")
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
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# load pre-trained embedding vector
print("loading embedding...")
embedding, word2idx = data_util.load_embedding(FLAGS.embedding_file)
print("time used %d"%(time.time() - start))

# load stop words
stop_words = codecs.open(FLAGS.stop_words_file, 'r', encoding='utf8').readlines()
stop_words = [w.strip() for w in stop_words]

# top k most related knowledge
#print("computing similarity...")
#train_sim_ixs = similarity.topk_sim_ix(FLAGS.train_file, stop_words, FLAGS.k)
#val_sim_ixs = similarity.topk_sim_ix(FLAGS.val_file, stop_words, FLAGS.k)

# load dictionary and corpus
train_data_dict, val_data_dict = {}, {}
# save the train data, load all the data if needs later
train_data_path = "tmp/train_data"
if os.path.exists(train_data_path):
    with open(train_data_path, "rb") as f:
        train_data_dict = pickle.load(f)
else:
    train_questions, train_answers, train_knowledges = \
        data_util.load_data_index(FLAGS.train_file, FLAGS.stop_words_file, word2idx, FLAGS.max_sentence_len,\
                            FLAGS.max_sentence_len, FLAGS.max_sentence_len)
    train_data_dict = {'train_questions': train_questions,
                  'train_answers': train_answers,
                  'train_knowledges': train_knowledges}

    with open(train_data_path, "wb") as f:
        pickle.dump(train_data_dict, f)

# save the train data, load all the data if needs later
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
'''
train_questions, train_answers, train_knowledges = train_data_dict['train_questions'],\
                                                                   train_data_dict['train_answers'],\
                                                                   train_data_dict['train_knowledges']
'''
val_questions, val_answers, val_knowledges = val_data_dict['val_questions'],\
                                             val_data_dict['val_answers'],\
                                             val_data_dict['val_knowledges']
print("loading data finished")
print("time used %d"%(time.time() - start))

# load data batch
train_data_batch_path = "tmp/train_data_batch"
if os.path.exists(train_data_batch_path):
    with open(train_data_batch_path, "rb") as f:
        train_data_batch_dict = pickle.load(f)
else:
    questions_knowledges, true_answers, false_answers = [], [], []
    for q, ta, fa in data_util.training_batch_iter(
            train_data_dict['train_questions'], train_data_dict['train_answers'], train_data_dict['train_knowledges'], word2idx, FLAGS.batch_size, FLAGS.k_neg,\
                FLAGS.max_sentence_len):
        questions_knowledges.append(q), true_answers.append(ta), false_answers.append(fa)
    train_data_batch_dict = {'questions_knowledges': questions_knowledges,
                 'true_answers': true_answers,
                 'false_answers': false_answers}
    with open(train_data_batch_path, "wb") as f:
        pickle.dump(train_data_batch_dict, f)
print("loading data batch finished")
print("time used %d" % (time.time() - start))
# --------------Data preprocess end--------------


# --------------Training begin--------------
print("training...")
with tf.Graph().as_default(), tf.device('/cpu:0'):

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_mem_usage
    )

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        gpu_options=gpu_options
    )

    with tf.Session(config=session_conf).as_default() as sess:
        globalStep = tf.Variable(0, name="globle_step", trainable=False)
        lstm = BiLSTM(
            FLAGS.batch_size,
            FLAGS.max_sentence_len,
            embedding,
            FLAGS.embedding_dim,
            FLAGS.rnn_size,
            FLAGS.margin
        )

        # define training procedure
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), FLAGS.max_grad_norm)
        saver = tf.train.Saver()

        # output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # summaries
        loss_summary = tf.summary.scalar("loss", lstm.loss)
        summary_op = tf.summary.merge([loss_summary])

        summary_dir = os.path.join(out_dir, "summary", "train")
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)


        # evaluating
        def evaluate():
            print("evaluating..")
            scores = []
            for val_q, val_a in data_util.testing_batch_iter(val_questions, val_answers, val_knowledges,\
                                            word2idx, FLAGS.batch_size, FLAGS.k_neg, FLAGS.max_sentence_len):
                val_feed_dict = {
                    lstm.inputTestQuestions: val_q,
                    lstm.inputTestAnswers: val_a,
                    lstm.dropout_keep_prob: 1.0
                }
                #print(np.shape(val_q), np.shape(val_a))
                _, score = sess.run([globalStep, lstm.result], val_feed_dict)
                scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for val_id in range(len(val_answers)):
                offset = val_id * FLAGS.k_neg
                predict_true_ix = np.argmax(scores[offset:offset + FLAGS.k_neg])
                if predict_true_ix == 0: # every k_neg answers, the first answer is true
                    cnt += 1
            print("evaluation acc: ", cnt / len(val_answers))
            '''
            scores = []
            for train_q, train_a in data_util.testing_batch_iter(train_questions, train_answers, train_knowledges,
                                                                 word2idx, FLAGS.batch_size, FLAGS.k_neg):
                val_feed_dict = {
                    lstm.inputTestQuestions: train_q,
                    lstm.inputTestAnswers: train_a,
                    lstm.dropout_keep_prob: 1.0
                }
                _, score = sess.run([globalStep, lstm.result], val_feed_dict)
                scores.extend(score)
            cnt = 0
            scores = np.absolute(scores)
            for train_id in range(len(train_answers)):
                offset = train_id * FLAGS.k_neg
                predict_true_ix = np.argmax(scores[offset:offset + FLAGS.k_neg])
                if predict_true_ix == 0:
                    cnt += 1
            print("evaluation acc(train): ", cnt / len(train_answers))
            '''

        # training
        sess.run(tf.global_variables_initializer())
        lr = FLAGS.learning_rate
        for i in range(FLAGS.lr_down_times):
            optimizer = tf.train.GradientDescentOptimizer(lr)
            optimizer.apply_gradients(zip(grads, tvars))
            trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
            for epoch in range(FLAGS.num_epochs):
                for question, trueAnswer, falseAnswer in zip(train_data_batch_dict['questions_knowledges'],\
                                                             train_data_batch_dict['true_answers'],\
                                                             train_data_batch_dict['false_answers']):
                    feed_dict = {
                        lstm.inputQuestions: question,
                        lstm.inputTrueAnswers: trueAnswer,
                        lstm.inputFalseAnswers: falseAnswer,
                        lstm.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                    _, step, _, _, loss, summary = \
                        sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op],
                                 feed_dict)
                    print("step:", step, "loss:", loss)
                    summary_writer.add_summary(summary, step)
                    if step % FLAGS.evaluate_every == 0:
                        evaluate()

                saver.save(sess, FLAGS.save_file)
            lr *= FLAGS.lr_down_rate

        # final evaluate
        evaluate()
# --------------Training end--------------
