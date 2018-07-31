import jieba.posseg
import os
import codecs
import pickle
from gensim import corpora, models, similarities
from pprint import pprint
from data_util import tokenizer, load_data_text

CUR_PATH = '/Users/ligen/PycharmProjects/Competition/Sogou/'

def generate_dic_and_corpus(filepath, stop_words): # not stop words file

    raw_questions, raw_answers, raw_questions_passages = load_data_text(filepath)
    q_length = len(raw_questions)
    knowledge_texts = []  # knowledge = question + passages
    questions_str = tokenizer(raw_questions, stop_words)
    #answers_str = tokenizer(raw_answers, stop_words)

    for idx in range(len(raw_questions_passages)):
        # pass_text = []
        # each passage of a question, put all passages' words together
        # for passage in raw_questions_passages[idx]:
        #    pass_text += passage
        temp = []
        q_kb = tokenizer(raw_questions_passages[idx], stop_words, remove_stopwords=True, \
                         remove_single_word=False)
        for kp in q_kb:
            temp += kp
        temp += questions_str[idx]
        knowledge_texts.append(temp)

    dictionary = corpora.Dictionary(knowledge_texts)  # dictionary of knowledge and train data
    os.path.join(CUR_PATH+'tmp/dictionary.dict')
    dictionary.save(CUR_PATH+'tmp/dictionary.dict')

    corpus = [dictionary.doc2bow(text) for text in knowledge_texts]  # corpus of knowledge
    corpora.MmCorpus.serialize(CUR_PATH+'tmp/knowledge_corpus.mm', corpus)


def topk_sim_ix(file_name, stop_words, k):
    """

    :param knowledge_file:
    :param file_name:
    :param stop_words:
    :param k:
    :return:
    """

    sim_path = CUR_PATH + "tmp/" +'train.sim'

    if os.path.exists(sim_path):
        with open(sim_path, "rb") as f:
            sim_ixs = pickle.load(f)
        return sim_ixs

    # load dictionary and corpus
    if not os.path.exists("tmp/dictionary.dict"):
        generate_dic_and_corpus(file_name, stop_words)
    dictionary = corpora.Dictionary.load("tmp/dictionary.dict")  # dictionary of knowledge and train data
    corpus = corpora.MmCorpus("tmp/knowledge_corpus.mm")  # corpus of knowledge

    # build Latent Semantic Indexing model
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)  # initialize an LSI transformation

    # similarity
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    sim_ixs = []  # topk related knowledge index of each question
    with open(file_name) as f:
        tmp = []  # background and question
        for i, line in enumerate(f):
            if i % 6 == 0:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
            if i % 6 == 1:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
                vec_lsi = lsi[dictionary.doc2bow(tmp)]  # convert the query to LSI space
                sim_ix = index[vec_lsi]  # perform a similarity query against the corpus
                sim_ix = [i for i, j in sorted(enumerate(sim_ix), key=lambda item: -item[1])[:k]]  # topk index
                sim_ixs.append(sim_ix)
                tmp.clear()
    with open(sim_path, "wb") as f:
        pickle.dump(sim_ixs, f)
    return sim_ixs


# module test
if __name__ == '__main__':
    stop_words_ = codecs.open("/Users/ligen/PycharmProjects/NLP/KB-QA-master/data/stop_words.txt", 'r', encoding='utf8').readlines()
    stop_words_ = [w.strip() for w in stop_words_]
    res = topk_sim_ix("/Users/ligen/PycharmProjects/NLP/KB-QA-master/data/knowledge.txt", "/Users/ligen/PycharmProjects/NLP/KB-QA-master/data/train.txt", stop_words_, 5)
    print(len(res))

