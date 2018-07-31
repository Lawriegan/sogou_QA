import pandas as pd
import re
from collections import defaultdict
import jieba.posseg
import numpy as np
import codecs

#PATH = '/Users/ligen/PycharmProjects/Competition/Sogou/'

#raw_data = pd.read_json(PATH + 'data/train_factoid_1.json', lines=True)


def load_embedding(filename):
    '''
    # load word2vec matrix
    :param filename: word embedding matrix file path
    :return: numpy array of embedding matrix, word indexes(dict {word: index})
    '''
    embeddings = []
    word2idx = defaultdict(list)
    with open(filename, mode="r", encoding="utf-8") as rf:
        next(rf)
        for line in rf:
            arr = line.split(" ")

            embedding = [float(val.strip()) for val in arr[1: -1]]
            word2idx[arr[0]] = len(word2idx)

            embeddings.append(np.array(embedding))

    return np.array(embeddings), word2idx

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def tokenizer(raw_text, stop_words, remove_stopwords=False):
    '''
    :param raw_text:  texts that have not been processed, including questions, answers, passages. list of str
    :param stop_words:
    :return:  processed texts. list of list of word
    '''
    texts = []
    for line in raw_text:
        if remove_stopwords:
            texts.append([token for token, _ in jieba.posseg.cut(strQ2B(line.rstrip()), HMM=True) \
                          if token.strip() not in stop_words and token.strip() != ''])
        else:
            texts.append([token for token, _ in jieba.posseg.cut(strQ2B(line.rstrip()), HMM=True) \
                          if token.strip() != ''])

    return texts

def wordlist2index(word_list, word2idx, max_len):
    '''
    # word list to indexes in embeddings
    :param word_list:
    :param word2idx: the map of words to indexes in embeddings
    :param max_len:
    :return: <list>
    '''
    unknown = word2idx.get('UNKNOWN', 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len

    for i in range(len(word_list)):
        word = word_list[i]
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= max_len - 1:
            break
    return index

def load_data_text(train_file):
    '''
    :param train_file:  train data file path
    :param stop_words_file:  stop word file path
    :return:
    '''

    #stop_words = codecs.open(stop_words_file, 'r', encoding='utf8').readlines()
    #stop_words = [w.strip() for w in stop_words]

    # load questions, answers and passages
    raw_data = pd.read_json(train_file, encoding='utf-8',lines=True)
    raw_questions = list(raw_data['query'])
    raw_answers = list(raw_data['answer'])
    raw_questions_passages = [[passage['passage_text'] for passage in question_passage] \
                         for question_passage in raw_data['passages']] # [['', '', ...], [], ... , []]
    return (raw_questions, raw_answers, raw_questions_passages)

def load_data_index(datafilepath, stop_words_file, word2idx, max_question_len, max_answer_len, max_knowledge):
    #raw_questions, raw_answers, raw_questions_passages = load_data_text(train_file, stop_words_file)
    raw_data = load_data_text(datafilepath)
    return data_text2index(raw_data, stop_words_file, word2idx, max_question_len, max_answer_len, max_knowledge)

def data_text2index(raw_data,\
                    stop_words_file, word2idx, max_question_len, max_answer_len, max_knowledge):
    '''

    :param raw_data:
    :param stop_words_file:
    :param word2idx:
    :param max_question_len:
    :param max_answer_len:
    :param max_knowledge:
    :return:
     question_index: [[question1_index], [question2_index], ...]
     answer_index: [[answer1_index], [answer2_index], ...]
     knowledge_index: [[[q1_passage1_index], [q1_passage2_index], ..., [q1_index]], [[q2_passage1_index]...]]]
    '''
    raw_questions, raw_answers, raw_questions_passages = raw_data
    # load stop words
    with codecs.open(stop_words_file, 'r', encoding='utf8') as stf:
        stop_words = [w.strip() for w in stf.readlines()]

    q_length = len(raw_questions)
    knowledge_texts = [] # knowledge = question + passages
    questions_texts = tokenizer(raw_questions, stop_words)
    answers_texts = tokenizer(raw_answers, stop_words)

    for question_passages in raw_questions_passages:
        passages = ''
        for passage_str in question_passages:
            passages = passages + ' ' + passage_str # 'str(passage1) str(passage2) ... '
        knowledge_texts.append(passages)

    knowledge_wordlist = tokenizer(knowledge_texts, stop_words, remove_stopwords=True)
    knowledges = []
    for i in range(q_length):
        no_repeat_word_text = list(set(knowledge_wordlist[i])) # make every word appear only once
        # generally the word in question is not the answer
        no_w_in_Q = [word for word in no_repeat_word_text if word not in questions_texts[i]]
        knowledges.append(no_w_in_Q)

    question_index = []
    answer_index = []
    knowledge_index = []
    for i in range(q_length):
        question_index.append(wordlist2index(questions_texts[i], word2idx, max_question_len))
        answer_index.append(wordlist2index(answers_texts[i], word2idx, max_answer_len))
        knowledge_index.append(wordlist2index(knowledges[i], word2idx, max_knowledge))

    return question_index, answer_index, knowledge_index

def training_batch_iter(questions, answers, knowledges, word2idx, batch_size, k_neg, max_len):
    """
    :return q + -
    """
    question_num = len(questions)
    batch_num = int(question_num / batch_size) + 1
    for batch in range(batch_num):
        # for each batch
        question_knowledges, true_answers, false_answers = [], [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, question_num)):
            # every question needs 5 copies
            question_knowledges.extend([knowledges[i]] * k_neg)
            # 5 right answers
            true_answers.extend([answers[i]] * k_neg)
            # 5 wrong answers (random choosen)
            no_unknown_K = list(set(knowledges[i]))
            idx = np.random.randint(0, len(no_unknown_K))
            idx_list = set()
            #idx_list = []
            #print("第%d个question" % i)
            #print("knowledges: %s"%(no_unknown_K))
            #print("true_answers: %s"%(answers[i]))
            for j in range(k_neg):
                while len(idx_list) < len(no_unknown_K) - 2 and (no_unknown_K[idx] in answers[i] or no_unknown_K[idx] in idx_list):
                    idx = np.random.randint(0, len(no_unknown_K))
                if len(idx_list) < len(no_unknown_K) - 2:
                    break
                idx_list.add(no_unknown_K[idx])
            for j in range(k_neg - len(idx_list)):
                word_idx = np.random.randint(0, len(word2idx))
                while word_idx in answers[i] or word_idx in idx_list:
                    word_idx = np.random.randint(0, len(word2idx))
                idx_list.add(word_idx)
            for j in idx_list:
                false_answer = [word2idx.get('UNKNOWN', 0)] * max_len
                # Notice! The right answers may not be only one word. !!!(need to be modified)
                false_answer[0] = j
                false_answers.append(false_answer)
            #print(false_answer)
        yield np.array(question_knowledges), np.array(true_answers), np.array(false_answers)


def testing_batch_iter(questions, answers, knowledges, word2idx, batch_size, k_neg, max_len):
    question_num = len(questions)
    batch_num = int(question_num / batch_size) + 1
    # question + passages = knowledges

    test_answers = []
    for batch in range(batch_num):
        question_knowledges, test_answers = [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, question_num)):
            # every question needs 5 copies
            # put all relative passages in one list
            question_knowledges.extend([knowledges[i]] * k_neg)
            # 1 right answers
            test_answers.append(answers[i])

            # k_neg - 1 wrong answers (random choosen)
            no_unknown_K = list(set(knowledges[i]))
            idx = np.random.randint(0, len(no_unknown_K))
            idx_list = set()
            for j in range(k_neg - 1):
                while no_unknown_K[idx] in answers[i] or idx in idx_list:
                    idx = np.random.randint(0, len(no_unknown_K))
                idx_list.add(idx)
                false_answer = [word2idx.get('UNKNOWN', 0)] * max_len
                # Notice! The right answers may not be only one word. !!!(need to be modified)
                false_answer[0] = no_unknown_K[idx]
                test_answers.append(false_answer)
        yield np.array(question_knowledges), np.array(test_answers)
