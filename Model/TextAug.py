from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import gensim

from google_trans_new import google_translator

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import nltk
nltk.download(['wordnet', 'punkt', 'stopwords', 'averaged_perceptron_tagger', 'words'])
from nltk.corpus import words, wordnet, comtrans
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetLMHeadModel, AutoTokenizer, AutoModelForMaskedLM
# from transformers import TFXLNetModel
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
from torch.nn import functional as F
import torch
import umls_api

from Model.word2vec.word2vec_hebrew.api.hebrew_w2v_api import HebrewSimilarWords
from Model.word2vec.wordembedding_hebrew.create_corpus import Convert

import pandas as pd
import numpy as np
from random import randrange
import os
from tqdm import tqdm
import time

import pickle  # with vpn
# import pickle5 as pickle  # when no vpn


# ---------------------------------------------------------------------------------------
# TextAug Class:
#
# Augmentation generations
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class TextAug:

    _model = None

    def __init__(self, ref_model):
        self._model = ref_model

        self.p_input = self._model.p_output + r'\output_parser'
        self.p_output = self._model.set_dir_get_path(self._model.p_output, 'output_text_aug')
        self.p_classifier = self._model.set_dir_get_path(self._model.p_output, 'output_classifier')
        self.p_features = self.p_input + r'\df_features_merged.csv'
        self.p_stopwords = self._model.p_resource + r'\stopwords.csv'
        self.p_onehot = self.p_output + r'\df_onehot.csv'
        self.p_tfidf = self.p_output + r'\x_sectors.csv'
        self.p_tta = self.p_output + r'\TTA'
        self.p_sectors = self.p_output + r'\sectors.csv'
        self.p_text = self.p_output + r'\df_text.csv'
        self.p_wiki_heb_vec = self.p_output + r'\wiki.he.vec'
        self.p_wiki_heb_bin = self.p_output + r'\wiki.he.bin'
        self.p_wiki_heb_txt = self.p_output + r'\wiki.he.text'
        self.p_glove = self.p_output + r'\wiki.he.text'
        self.p_xlnet = self._model.p_resource + r'\word2vec\xlnet_cased_L-24_H-1024_A-16'

        if self._model.b_vpn:
            self.p_input = self._model.set_vpn_dir(self.p_input)
            self.p_output = self._model.set_vpn_dir(self.p_output)
            self.p_classifier = self._model.set_vpn_dir(self.p_classifier)
            self.p_features = self._model.set_vpn_dir(self.p_features)
            self.p_stopwords = self._model.set_vpn_dir(self.p_stopwords)
            self.p_onehot = self._model.set_vpn_dir(self.p_onehot)
            self.p_tfidf = self._model.set_vpn_dir(self.p_tfidf)
            self.p_tta = self._model.set_vpn_dir(self.p_tta)
            self.p_sectors = self._model.set_vpn_dir(self.p_sectors)
            self.p_text = self._model.set_vpn_dir(self.p_text)
            self.p_wiki_heb_vec = self._model.set_vpn_dir(self.p_wiki_heb_vec)
            self.p_wiki_heb_bin = self._model.set_vpn_dir(self.p_wiki_heb_bin)
            self.p_wiki_heb_txt = self._model.set_vpn_dir(self.p_wiki_heb_txt)
            self.p_xlnet = self._model.set_vpn_dir(self.p_xlnet)

        self.translator = google_translator()
        self.get_synonym = naw.SynonymAug(aug_src='wordnet')
        # self.get_synonym = EDA()
        # self.py_dict = PyDictionary()
        self.get_hebrew_synonym = HebrewSimilarWords()

        self.demographic_size = 4
        self.examination_size = 3
        # self.general_size = self.demographic_size + self.examination_size
        self.general_size = 4  # Age, Gender, VariableAmount & GlucoseLevel

        self.emb_size = 0
        self.i = 0
        self.i_invalid = 0
        
        self.i_syn = 3
        # self.i_syn = 5
        
        self.i_w2v = 3
        # self.i_w2v = 5
        
        self.i_sleep = 2
        # self.i_sleep = 4

        # self.max_features = 100000
        # self.max_features = 99500
        self.max_features = 93000
        # self.max_features = None

        self.df_data = None
        self.df_tfidf = None
        self.df_normalized = None
        self.df_translated = None
        self.df_syn = None
        self.df_w2v = None
        self.df_backtrans = None
        self.m_w2v = None
        self.vocab_w2v = None
        self.vocab_tfidf = None

        self.s_text = ''
        self.curr_w2v_name = ''

        l_puncs = ["\"", '\"', ',', '"', '|', '?', '-', '_', '*', '`', '/', '@', ';', "'", '[', ']', '(', ')',
                   '{', '}', '<', '>', '~', '^', '?', '&', '!', '=', '+', '#', '$', '%', ':', '.']
        self.d_punc = self.set_puncwords(l_puncs)

        self.df_all_tfidf = pd.DataFrame(columns=['CaseID'])
        self.l_onehots = list()
        self.l_tfdf_cols_settings = ['Service', 'VariableLocation', 'VariableRange', 'TestSetting']
        self.l_targets_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']

        self.d_invalid = dict()

        if self._model.check_file_exists(self.p_stopwords):
            self.df_stopwords = pd.read_csv(self.p_stopwords)
            l_stopwords = self.df_stopwords.values.tolist()
            self.l_stopwords = [item for sublist in l_stopwords for item in sublist]
        else:
            print('stopwords file not found.')

    def apply_concat(self, value):
        new_value = ' '.join(value.dropna().astype(str))  # deletes all row if at least 1 NA exists
        self.s_text += new_value
        return new_value

    def merge_to_one_col(self, df_curr, filename):
        l_cols = self._model.l_tfdf_cols_features.copy()
        if 'sectors' not in filename:
            if '_' in filename:
                s_aug = filename.split('_')[1]
            else:
                s_aug = filename
            for i in range(len(l_cols)):
                l_cols[i] = s_aug+l_cols[i]
        df_curr[l_cols] = df_curr[l_cols].replace(np.nan, '', regex=True)
        s_corpus_sectors = df_curr[l_cols].T.agg(' '.join)  # merges multiple columns to one column
        s_corpus_sectors = s_corpus_sectors.str.lower()

        # (1) with validation
        df_corpus_sectors = pd.DataFrame(columns=['CaseID', 'Text'])
        s_case_id = df_curr['CaseID']
        df_corpus_sectors['CaseID'] = s_case_id
        df_corpus_sectors['Text'] = s_corpus_sectors

        # (2) without validation
        # df_corpus_sectors = pd.DataFrame(s_corpus_sectors, columns=['Text'])
        # s_corpus_sectors = df_curr[self._model.l_tfdf_cols_features].agg(lambda x: ' '.join(x.values), axis=1).T  # v2

        self._model.set_df_to_csv(df_corpus_sectors, filename, self.p_output, s_na='', b_append=False, b_header=True)
        print('Merged to one column.')
        return s_corpus_sectors

    def merge_to_one_cell(self, df_curr, filename):
        s_curr = df_curr['Text']
        s_curr = s_curr.str.lower()
        df_text = pd.DataFrame({'Text': [' '.join(s_curr)]})  # merges multiple rows to one row
        # df_text = pd.DataFrame({'Text': [' '.join(s_curr.str.strip('"').tolist())]})  # v2
        self._model.set_df_to_csv(df_text, filename, self.p_output, s_na='', b_append=False, b_header=True)
        print('Merged to one cell.')
        return df_text

    def extract_filename(self, p_curr, curr_aug):
        if self._model.b_vpn:
            i = p_curr.rfind('/')
        else:
            i = p_curr.rfind('\\')
        j = p_curr.rfind('.')
        name = p_curr[i+1:j]
        # name = name.replace(curr_aug, '')
        return name

    @staticmethod
    def set_file_list(filename, p_curr):
        l_files = []
        for root, dirs, files in os.walk(p_curr):
            for file in files:
                if filename in file:
                    curr_file_path = os.path.join(root, file)
                    if 'backup' not in curr_file_path:
                        l_files.append(curr_file_path)
        return l_files

    def merge_aug(self):
        for curr_aug in self._model.l_tta_types:
            filename = 'Merged'+curr_aug
            l_curr_files = self.set_file_list(curr_aug, self.p_output)
            df_org = pd.DataFrame()
            for p_file in l_curr_files:
                p_file = self._model.set_vpn_dir(p_file)
                df_new = pd.read_csv(p_file)
                l_cols = list(df_new.columns)  # first row gets header name
                for i in range(len(l_cols)):
                    if 'Unnamed' in l_cols[i]:
                        l_cols[i] = ''
                df_new.loc[-1] = l_cols
                df_new.index = df_new.index + 1
                df_new = df_new.sort_index()
                col_name = self.extract_filename(p_file, curr_aug)
                df_new.columns = [col_name]

                df_org = df_org.merge(df_new, left_index=True, right_index=True)
                # df_org = pd.concat([df_org, df_new], axis=1)  # 0 by row, 1 by col

            self._model.set_df_to_csv(df_org, filename, self.p_output, s_na='', b_append=False, b_header=True)

    @staticmethod
    def set_puncwords(l_curr_puncs):
        d_puncwords = {}
        if l_curr_puncs is not None:
            for word in l_curr_puncs:
                d_puncwords[word] = None
            del l_curr_puncs
        return d_puncwords

    def normalize_text(self, text):
        return text.translate(str.maketrans(self.d_punc)).strip()

    @staticmethod
    def toss(threshold=0.677):
        uniform = np.random.uniform(0, 1)
        if uniform >= threshold:
            return False
        else:
            return True

    def synonym_apply(self, value):
        syn_sequence = value
        if value != 'False' and value != '':
            value = self.normalize_text(value)
            l_sequence = value.split()
            syn_sequence = ''

            # for i_term in range(len(l_sequence)):  # v1
            #     term = l_sequence[i_term]
            #     if i_term % self.i_syn == 0:
            #         syn_sequence += self.get_synonym.augment(term) + ' '
            #     else:
            #         syn_sequence += term + ' '

            # for term in l_sequence:  # v2
            #     if self.toss(0.677):
            #         syn_sequence += self.get_synonym.augment(term) + ' '
            #         # syn_sequence += self.get_synonym.synonym_replacement(term)
            #     else:
            #         syn_sequence += term + ' '

            for i_term in range(len(l_sequence)):  # v3
                term = l_sequence[i_term]
                if i_term % self.i_syn == 0:
                    l_results = list()
                    chosen_candidate = ''
                    term_error = ''
                    b_stop = False

                    try:
                        l_results = self.get_hebrew_synonym.get_similar(term)
                    except KeyError:
                        term_error = term
                        term = term[1:]
                        try:
                            l_results = self.get_hebrew_synonym.get_similar(term)
                        except KeyError:
                            chosen_candidate = term
                            b_stop = True

                    if not b_stop:
                        try:
                            l_candidates = list()
                            for candidate in l_results:
                                curr_candidate = candidate['word']
                                if term not in curr_candidate:
                                    if len(curr_candidate) > 1:
                                        if '~' in curr_candidate:
                                            curr_candidate = curr_candidate.replace('~', ' ')
                                        l_candidates.append(curr_candidate)
                            i_random = randrange(0, len(l_candidates))  # includes only the left edge
                            chosen_candidate = l_candidates[i_random]
                        except (KeyError, ValueError):
                            self.i_invalid += 1
                            if term_error != '':
                                print(f'Could not synonym the term: {term_error}')
                                self.d_invalid[term_error] = ''
                            else:
                                print(f'Could not synonym the term: {term}')
                                self.d_invalid[term] = ''
                            chosen_candidate = term

                    syn_sequence += chosen_candidate + ' '
                else:
                    syn_sequence += term + ' '

        syn_sequence = self.normalize_text(syn_sequence)
        syn_sequence = syn_sequence.lower()

        return syn_sequence

    def set_candidate(self, l_curr_window, i_top=10):
        i_random = randrange(i_top)

        # v1-gensim #
        preds = self.m_w2v.predict_output_word(l_curr_window, topn=i_top)
        curr_tuple = preds[i_random]
        top_candidate = curr_tuple[0]

        # v2-nlpaug #
        # text = ' '.join(l_curr_window)
        # preds = self.m_w2v.augment(text)
        # if ' ' in preds:
        #     l_preds = preds.split(' ')
        #     top_candidate = l_preds[:-1]
        # else:
        #     top_candidate = preds

        # b_contains = False
        # threshold = 10
        # top_candidate = preds
        # if ' ' in preds:  # do
        #     l_preds = preds.split(' ')
        #     if len(l_preds) > 1:
        #         for curr_pred in l_preds:
        #             if curr_pred in text:
        #                 b_contains = True
        # elif preds in text:
        #     b_contains = True
        # if b_contains:
        #     i = 0
        #     while b_contains and i < threshold:  # while
        #         preds = self.m_w2v.augment(text)
        #         if ' ' in preds:
        #             l_preds = preds.split(' ')
        #             if len(l_preds) > 1:
        #                 for curr_pred in l_preds:
        #                     if curr_pred in text:
        #                         b_contains = True
        #         elif preds in text:
        #             b_contains = True
        #         else:
        #             b_contains = False
        #             top_candidate = preds
        #         i += 1
        # else:
        #     top_candidate = preds

        return top_candidate

    def set_similar(self, curr_term, i_top=5):
        preds = self.m_w2v.wv.most_similar(positive=[curr_term], topn=i_top)
        i_random = randrange(5)
        curr_tuple = preds[i_random]
        top_candidate = curr_tuple[0]
        return top_candidate

    def xlnet_apply(self, value, tokenizer):
        s_xlnet = value
        if value != 'False' and value != '':
            inputs = tokenizer(value, return_tensors="pt")
            outputs = self.m_w2v(**inputs)
            last_hidden_states = outputs.last_hidden_state
            i_random = randrange(len(outputs))
            s_xlnet = outputs[i_random]

            # text = s_xlnet + tokenizer.mask_token
            # mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
            # inputs = tokenizer.encode_plus(text, return_tensors="pt")
            # outputs = self.m_w2v(**inputs).logits
            # softmax = F.softmax(outputs, dim=-1)
            # mask_word = softmax[0, mask_index, :]
            # top_10 = torch.topk(mask_word, 10, dim=1)[1][0]
            # i_random = randrange(len(outputs))
            # for token in top_10:
            #     word = tokenizer.decode([token])
            #     s_xlnet = text.replace(tokenizer.mask_token, word)
            #     print(s_xlnet)
            # s_xlnet = top_10[i_random]

        s_xlnet = self.normalize_text(s_xlnet)
        s_xlnet = s_xlnet.lower()
        return s_xlnet

    def word2vec_apply(self, value):
        s_w2v = value
        i_top = 10
        if value != 'False' and value != '':
            s_w2v = ''
            value = self.normalize_text(value)
            l_w2v_terms = value.split(' ')
            i_length = len(l_w2v_terms)
            i_modulo = self.i_w2v

            # v1.5-keyVectors
            for i_term in range(len(l_w2v_terms)):
                term = l_w2v_terms[i_term]
                if i_term % i_modulo == 0:
                    try:
                        l_preds = self.m_w2v.most_similar(term, topn=i_top)
                        i_random = randrange(i_top)
                        curr_tuple = l_preds[i_random]
                        chosen_candidate = curr_tuple[0]
                    except KeyError:
                        chosen_candidate = term
                    s_w2v += chosen_candidate + ' '
                else:
                    s_w2v += term + ' '

            # v1-gensim
            # i_steps = i_length // self.i_w2v
            # d_index_candidates = dict()
            # i_modulo = self.i_w2v
            # if i_steps > 0:
            #     i_count = 0
            #     while i_count < i_steps:
            #         if i_modulo - 2 < i_length:
            #             l_curr_window = l_w2v_terms[:i_modulo-1]
            #             chosen_candidate = self.set_candidate(set_candidate(l_curr_window)
            #             d_index_candidates[i_modulo-1] = chosen_candidate
            #             i_modulo += 3
            #             i_count += 1
            #     for i_chosen, s_chosen in d_index_candidates.items():
            #         l_w2v_terms[i_chosen] = s_chosen
            #     s_w2v = ' '.join(l_w2v_terms)
            # elif i_length <= 2:
            #     l_candidates = list()
            #     for curr_term in l_w2v_terms:
            #         l_candidates.append(self.set_similar(curr_term))
            #     s_w2v = ' '.join(l_candidates)

            # v2-nlpaug
            # s_w2v = self.m_w2v.augment(value)

        s_w2v = self.normalize_text(s_w2v)
        s_w2v = s_w2v.lower()

        return s_w2v

    def translate_apply(self, value, src, tgt):
        self.i += 1
        seq_trans = value
        if value != 'False' and value != '':
            s_normalized_sentence = self.normalize_text(value)
            seq_trans = ''
            time.sleep(self.i_sleep)  # per sequence
            try:
                seq_trans = self.translator.translate(s_normalized_sentence, lang_src=src, lang_tgt=tgt)
                if seq_trans != '':
                    seq_trans = self.normalize_text(seq_trans)
                    seq_trans = seq_trans.lower()
                else:
                    print(f'Invalid sequence for translation: {seq_trans}')
                    self.i_invalid += 1
            except TypeError as te:
                seq_trans = str(te)
        return seq_trans

    def word2vec_train(self):
        # self.emb_size = 100
        self.emb_size = 300
        # self.emb_size = 1000
        # self.emb_size = 10000
        p_child = 'm_word2vec' + '_' + str(self.emb_size)
        self.curr_w2v_name = self._model.validate_path(self.p_output, p_child, 'model')

        if not self._model.check_file_exists(self.curr_w2v_name):
            # v1
            # df_data = pd.read_csv(self.p_text)  # loads input corpus
            # normalized_corpus = df_data.at[0, 'Text']  # has punctuations?
            # tokenizer = RegexpTokenizer(r'\w+')
            # tokens = tokenizer.tokenize(normalized_corpus)
            # print(f'Number of terms: {len(tokens)}')  # 399,852
            # corpus = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(normalized_corpus)]

            # v2
            corpus = open(self.p_wiki_heb_txt, "r")
            m_w2v = Word2Vec(corpus)

            sg = 0  # CBOW
            cbow_mean = 1  # if cbow applied

            # sg = 1  # SkipGram
            # cbow_mean = 0  # sum of context word vectors

            hs = 0
            # hs = 1  # Hierarchical Softmax

            negative_size = 10
            # negative_size = 20
            # negative_size = 5

            f_sample = 0.00001  # downsamples high tf terms randomly [0, 0.00001]
            # f_sample = 0.00006

            window_size = 5
            # min_size = 0  # ignores terms with tf lower than min_size
            min_size = 5

            # epoch_size = 50
            # epoch_size = 100
            epoch_size = 500

            f_lr = 0.03
            f_min_lr = 0.0007

            # (v1)
            # m_w2v = Word2Vec(corpus,
            #                  min_count=min_size,
            #                  window=window_size,
            #                  sample=f_sample,
            #                  alpha=f_lr,
            #                  min_alpha=f_min_lr,
            #                  negative=negative_size,
            #                  workers=4)

            # m_w2v.build_vocab(corpus, progress_per=10000)  # builds vocabulary
            # self.vocab_w2v = m_w2v.wv.key_to_index  # saves vocabulary
            # p_child_vocab = 'word2vec_vocab' + '_' + str(self.emb_size)
            # self._model.set_pickle(self.vocab_w2v, self.p_output, p_child_vocab)

            i_term_count = m_w2v.corpus_count
            m_w2v.train(corpus, total_words=i_term_count, epochs=epoch_size, report_delay=1)  # trains word2vec model

            self.word2vec_save(m_w2v, p_child)  # saves word2vec model

            corpus.close()
        else:
            print(f'Word2Vec Model Exists Already: {p_child}')

    def word2vec_save(self, m_w2v, p_child):
        m_w2v.save(self.curr_w2v_name)
        print(f'Word2Vec Model Completed: {p_child}')

    def word2vec_load(self, p):
        if self._model.check_file_exists(p):
            self.m_w2v = Word2Vec.load(p)
        else:
            print('Word2Vec model has not been found.')

    def vocabulary_load(self, s_emb='100'):
        vocabulary = dict()
        s_vocabulary = 'word2vec_vocab' + '_' + s_emb
        p_vocabulary = self._model.validate_path(self.p_output, s_vocabulary, 'pkl')
        if self._model.check_file_exists(p_vocabulary):
            vocabulary = self._model.get_pickle(self.p_output, s_vocabulary)
        else:
            print(f'Word2Vec Model: {s_vocabulary} has not been found.')
        return vocabulary

    def tfidf(self, value, i_features):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=self.l_stopwords, max_features=i_features,
                                     analyzer='word', encoding='utf-8', decode_error='strict',
                                     lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False)

        # if self.vocab_tfidf is not None:  # v2
        #     vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=self.l_stopwords, max_features=i_features,
        #                                  analyzer='word', encoding='utf-8', decode_error='strict',
        #                                  lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False,
        #                                  vocabulary=self.vocab_tfidf)

        m_tfidf = vectorizer.fit_transform(value)
        feature_names = vectorizer.get_feature_names()
        m_tfidf = pd.DataFrame(m_tfidf.toarray(), columns=feature_names)
        return m_tfidf

    def get_sample(self, data, sample=1000):
        data_sample = data.copy()
        data_sample = data_sample.sample(frac=1).reset_index(drop=True)
        data_sample = data_sample.iloc[:sample]
        return data_sample

    def umls_apply(self, value):
        s_umls = ''
        if value != 'False' and value != '':
            resp1 = umls_api.API(api_key=value).get_cui(cui='C0007107')  # symbolic id
            resp2 = umls_api.API(api_key=value).get_tui(cui='C0007107')  # semantic id
            s_umls = resp1['result']['name']
        return s_umls

    def create_cui_dict(self, voc_updated, tokenizer):
        tui_ids = dict()
        id_to_tui = torch.zeros(len(tokenizer), dtype=torch.long)
        voc_size = 0
        with open(voc_updated, 'r') as reader:
            for line in reader.readlines():
                voc_size = voc_size + 1
                line = line.replace("\n", "").replace(" ", "")
                line_list = line.split("|")
                if len(line_list) > 1:
                    if len(line_list[1]) > 2 and len(line_list[0]) > 2:
                        word_id = tokenizer.convert_tokens_to_ids(line_list[0])
                        if line_list[2] not in tui_ids:
                            tui_ids[line_list[2]] = len(tui_ids) + 1
                        id_to_tui[word_id] = tui_ids[line_list[2]]
        return id_to_tui

    def load_umls(self):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        if (
                os.path.exists(training_args.output_dir)
                and os.listdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        set_seed(training_args.seed)
        try:
            num_labels = 2
            output_mode = 'classification'
        except KeyError:
            raise ValueError("Task not found: %s" % (data_args.task_name))

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        tui_ids = None
        if model_args.umls:
            tui_ids = self.create_cui_dict(voc_updated=model_args.med_document, tokenizer=tokenizer)

        train_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir,
                        tui=tui_ids) if training_args.do_train else None
        )
        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, tui=tui_ids)
            if training_args.do_eval
            else None
        )
        test_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir, tui=tui_ids)
            if training_args.do_predict
            else None
        )

        def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                if output_mode == "classification":
                    preds = np.argmax(p.predictions, axis=1)
                elif output_mode == "regression":
                    preds = np.squeeze(p.predictions)
                return glue_compute_metrics(task_name, preds, p.label_ids)

            return compute_metrics_fn

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name),
        )

        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            # trainer.save_model()
            # if trainer.is_world_master():
            #     tokenizer.save_pretrained(training_args.output_dir)

        eval_results = {}
        if training_args.do_eval:

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_datasets = [eval_dataset]
            if data_args.task_name == "mnli":
                pass
                # mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
                # eval_datasets.append(
                #     GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
                # )

            for eval_dataset in eval_datasets:
                trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = os.path.join(
                    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        for key, value in eval_result.items():
                            writer.write("%s = %s\n" % (key, value))

                eval_results.update(eval_result)

        labels_id = []
        if training_args.do_predict:
            test_datasets = [test_dataset]
            if data_args.task_name == "mnli":
                pass
                # mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
                # test_datasets.append(
                #     GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
                # )

            for test_dataset in test_datasets:
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                if output_mode == "classification":
                    predictions = np.argmax(predictions, axis=1)
                for i in range(0, len(test_dataset.features)):
                    labels_id.append(test_dataset.features[i].label)

                metric = glue_compute_metrics(data_args.task_name, predictions, labels_id)
                output_test_file = os.path.join(
                    training_args.output_dir, f"test_results_{test_dataset.args.task_name}_metric.txt"
                )
                if trainer.is_world_master():
                    with open(output_test_file, "w") as writer:
                        for key, value in metric.items():
                            writer.write("%s = %s\n" % (key, value))

                eval_results.update(eval_result)

                output_test_file = os.path.join(
                    training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_test_file, "w") as writer:
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if output_mode == "regression":
                                writer.write("%d\t%3.3f\n" % (index, item))
                            else:
                                item = test_dataset.get_labels()[item]
                                writer.write("%d\t%s\n" % (index, item))
        return eval_results

    def aug_umls(self, col_key, umls_col_name):
        # p_emb = self._model.p_resource + '/umls/bert_multi_cased/multi_cased_L-12_H-768_A-12'
        p_emb = self._model.p_resource + '/umls/bert_umls/umlsbert/pytorch_model.bin'
        tokenizer = AutoTokenizer.from_pretrained(p_emb)
        m_umls = AutoModelForMaskedLM.from_pretrained(p_emb)
        p_ulms_file = self._model.validate_path(self.p_tta, umls_col_name, 'csv')
        if self._model.check_file_exists(p_ulms_file):
            self.i_invalid = 0
            print(f'Finished Augmentation: UMLS {umls_col_name}.')
        else:
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), umls_col_name, self.p_tta, s_na='',
                                      b_append=True, b_header=True)
            self.i_invalid = 0
            # (1) no chunks
            # df_umls = df_curr[col_key].progress_apply(self.umls_apply)
            # self._model.set_df_to_csv(df_umls, umls_col_name, self.p_output, s_na='', b_append=False, b_header=False)

            # (2) with chunks
            with open(self.p_sectors) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_umls = df_curr_chunk[col_key].copy()
                        s_curr_to_umls = s_curr_to_umls.fillna(value='')
                        s_curr_umls = s_curr_to_umls.progress_apply(self.umls_apply)

                        # (1) with validation
                        df_curr_umls = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_umls = df_curr_umls.merge(s_curr_umls, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_umls = pd.DataFrame(s_curr_umls)

                        self._model.set_df_to_csv(df_curr_umls, umls_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Augmentation: UMLS {umls_col_name}, with {self.i_invalid} invalid terms.')

    def glove_train(self, texts):
        embeddings_index = {}
        max_words = 10000

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))
        f = self.p_glove
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_dim = 100

        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

    def convert_bin_to_txt(self):
        if not self._model.check_file_exists(self.p_wiki_heb_txt):
            p_data = self._model.p_project + '/Model/word2vec/wordembedding_hebrew'
            converter = Convert()
            converter.apply_convert(p_data, self.p_output)

    def aug_w2v(self, col_key, w2v_col_name):
        # self.curr_w2v_name = 'm_word2vec_100'
        # self.curr_w2v_name = 'm_word2vec_300'
        # self.curr_w2v_name = 'm_word2vec_1000'

        # (1) gensim: word2vec (input corpus)
        # p_m_w2v = self._model.validate_path(self.p_output, self.curr_w2v_name, 'model')
        # self.word2vec_load(p_m_w2v)

        # (2) nlp-aug: fasttext (wiki)
        # self.m_w2v = naw.WordEmbsAug(model_type='fasttext', model_path=self.p_wiki_heb_vec)

        # (3) nlp-aug: word2vec (wiki)
        # (3.1)
        # self.m_w2v = naw.WordEmbsAug(model_type='word2vec', model_path=self.p_wiki_heb_txt)

        # (3.2) gensim: word2vec (wiki)
        # self.m_w2v = KeyedVectors.load_word2vec_format(self.p_wiki_heb_bin, binary=True, unicode_errors='ignore', encoding='utf8')
        # self.m_w2v = KeyedVectors.load_word2vec_format(self.p_wiki_heb_vec, binary=False, unicode_errors='ignore', encoding='utf8')

        # (3.3) gensim: word2vec (wiki)
        # self.m_w2v = Word2Vec.load(self.p_wiki_heb_txt)  # .bin / .txt

        # (4) nlp-aug: glove (wiki)
        # self.m_w2v = naw.WordEmbsAug(model_type='glove', model_path=self.p_wiki_heb_txt)

        # (5) nlp-aug: BERT
        self.m_w2v = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased', aug_p=0.1)  # tf 2.3

        p_w2v_file = self._model.validate_path(self.p_tta, w2v_col_name, 'csv')
        if self._model.check_file_exists(p_w2v_file):
            self.i_invalid = 0
            print(f'Finished Augmentation: Word2Vec {w2v_col_name}.')
        else:
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), w2v_col_name, self.p_tta, s_na='',
                                      b_append=True, b_header=True)
            self.i_invalid = 0
            # (1) no chunks
            # df_w2v = df_curr[col_key].progress_apply(self.word2vec_apply)
            # self._model.set_df_to_csv(df_w2v, w2v_col_name, self.p_output, s_na='', b_append=False, b_header=False)

            # (2) with chunks
            p_sectors_trans = self.p_tta + '/translation' + '.csv'
            with open(p_sectors_trans) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_w2v = df_curr_chunk[col_key].copy()
                        s_curr_to_w2v = s_curr_to_w2v.fillna(value='')

                        # s_curr_w2v = s_curr_to_w2v.progress_apply(self.word2vec_apply)
                        s_curr_w2v = s_curr_to_w2v.progress_apply(self.xlnet_apply, args=(m_tokenizer,))

                        # (1) with validation
                        df_curr_w2v = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_w2v = df_curr_w2v.merge(s_curr_w2v, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_w2v = pd.DataFrame(s_curr_w2v)

                        self._model.set_df_to_csv(df_curr_w2v, w2v_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Augmentation: Word2Vec {w2v_col_name}, with {self.i_invalid} invalid terms.')

    def aug_backtranslate(self, col_key, trans_col_name, backtrans_col_name):
        p_backtrans_file = self._model.validate_path(self.p_tta, backtrans_col_name, 'csv')
        p_trans_file = self._model.validate_path(self.p_tta, trans_col_name, 'csv')
        if self._model.check_file_exists(p_backtrans_file):
            self.i_invalid = 0
            print(f'Finished Augmentation: Back Translations {backtrans_col_name}.')
        else:
            self.i_invalid = 0
            # (1) no chunks
            # df_curr_translated_heb = df_curr[col_key].progress_apply(self.translate_apply, src='en', tgt='iw')
            # self._model.set_df_to_csv(df_curr_translated_heb, backtrans_col_name, self.p_output, s_na='', b_append=True, b_header=False)

            # (2) with chunks
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), backtrans_col_name, self.p_tta, s_na='',
                                      b_append=True, b_header=True)
            with open(p_trans_file) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                i = 0
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty and i > 1:
                        s_curr_to_translate = df_curr_chunk[col_key].copy()
                        s_curr_to_translate = s_curr_to_translate.fillna(value='')
                        s_curr_translated_heb = s_curr_to_translate.progress_apply(self.translate_apply, src='en', tgt='iw')

                        # (1) with validation
                        df_curr_translated_heb = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_translated_heb = df_curr_translated_heb.merge(s_curr_translated_heb, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_translated_heb = pd.DataFrame(s_curr_translated_heb)

                        self._model.set_df_to_csv(df_curr_translated_heb, backtrans_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
                    i += 1
            print(f'Finished Augmentation: Back Translation {backtrans_col_name}, with {self.i_invalid} invalid terms.')

    def aug_hebrew_synonym(self, col_key, synheb_col_name):
        self.i_invalid = 0
        self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), synheb_col_name, self.p_tta, s_na='',
                                  b_append=True, b_header=True)
        with open(self.p_sectors) as read_chunk:
            chunk_iter = pd.read_csv(read_chunk, chunksize=500)
            tqdm.pandas()
            for df_curr_chunk in tqdm(chunk_iter):
                if not df_curr_chunk.empty:
                    s_curr_to_syn_heb = df_curr_chunk[col_key].copy()
                    s_curr_to_syn_heb = s_curr_to_syn_heb.fillna(value='')
                    s_curr_syn_heb = s_curr_to_syn_heb.progress_apply(self.synonym_apply)

                    # (1) with validation
                    df_syn = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                    df_syn = df_syn.merge(s_curr_syn_heb, left_index=True, right_index=True)

                    # (2) no validation
                    # df_syn = pd.DataFrame(s_curr_syn_en)

                    self._model.set_df_to_csv(df_syn, synheb_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
        self._model.set_dict_to_csv(self.d_invalid, 'invalid_hebrew_synonym_terms', self.p_output)
        print(f'Hebrew synonyms invalid terms: {int(self.i_invalid)}')

    def aug_synonym(self, col_key, trans_col_name, syn_col_name, syn_heb_col_name):
        p_trans_file = self._model.validate_path(self.p_tta, trans_col_name, 'csv')
        p_syn_file = self._model.validate_path(self.p_tta, syn_col_name, 'csv')
        p_syn_heb_file = self._model.validate_path(self.p_tta, syn_heb_col_name, 'csv')

        if self._model.check_file_exists(p_syn_heb_file):
            self.i_invalid = 0
            print(f'Finished Augmentation: Hebrew Synonyms {syn_heb_col_name}.')

        elif self._model.check_file_exists(p_syn_file):  # (1) if only Synonym English file exists -> Translate to Heb
            self.i_invalid = 0
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), syn_heb_col_name, self.p_tta, s_na='', b_append=True, b_header=True)
            with open(p_syn_file) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_syn_heb = df_curr_chunk[col_key].copy()
                        s_curr_to_syn_heb = s_curr_to_syn_heb.fillna(value='')
                        s_curr_syn_heb = s_curr_to_syn_heb.progress_apply(self.translate_apply, src='en', tgt='iw')

                        # (1) with validation
                        df_curr_syn_heb = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_syn_heb = df_curr_syn_heb.merge(s_curr_syn_heb, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_syn_heb = pd.DataFrame(s_curr_syn_heb)

                        self._model.set_df_to_csv(df_curr_syn_heb, syn_heb_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Augmentation: Hebrew Synonyms {syn_heb_col_name}, with {self.i_invalid} invalid terms.')

        else:
            self.i_invalid = 0
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), syn_col_name, self.p_tta, s_na='', b_append=True, b_header=True)
            with open(p_trans_file) as read_chunk:  # (2) if only Translated English file exists -> English Synonym
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_syn_en = df_curr_chunk[col_key].copy()
                        s_curr_to_syn_en = s_curr_to_syn_en.fillna(value='')
                        s_curr_syn_en = s_curr_to_syn_en.progress_apply(self.synonym_apply)

                        # (1) with validation
                        df_syn = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_syn = df_syn.merge(s_curr_syn_en, left_index=True, right_index=True)

                        # (2) no validation
                        # df_syn = pd.DataFrame(s_curr_syn_en)

                        self._model.set_df_to_csv(df_syn, syn_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Generating English Synonyms {syn_col_name}, with {self.i_invalid} invalid terms.')
            self.i_invalid = 0
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), syn_heb_col_name, self.p_tta, s_na='', b_append=True, b_header=True)
            with open(p_syn_file) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)  # (3) Then, we can apply hebrew synonyms
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_syn_heb = df_curr_chunk[col_key].copy()
                        s_curr_to_syn_heb = s_curr_to_syn_heb.fillna(value='')
                        s_curr_syn_heb = s_curr_to_syn_heb.progress_apply(self.translate_apply, src='en', tgt='iw')

                        # (1) with validation
                        df_curr_syn_heb = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_syn_heb = df_curr_syn_heb.merge(s_curr_syn_heb, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_syn_heb = pd.DataFrame(s_curr_syn_heb)

                        self._model.set_df_to_csv(df_curr_syn_heb, syn_heb_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Augmentation: Hebrew Synonyms {syn_heb_col_name}, with {self.i_invalid} invalid terms.')

    def aug_translate(self, col_key, trans_col_name):
        self.i_invalid = 0
        # print(self.translator.LANGUAGES) -> English = 'en', Hebrew = 'heb' or 'iw'
        p_trans_file = self._model.validate_path(self.p_tta, trans_col_name, 'csv')
        if self._model.check_file_exists(p_trans_file):
            print(f'Finished Translation to English {trans_col_name}')
        else:
            # (1) no chunks
            # df_translated = df_curr[col_key].progress_apply(self.translate_apply, src='iw', tgt='en')
            # self._model.set_df_to_csv(df_curr_translated, trans_col_name, self.p_output, s_na='', b_append=True, b_header=False)

            # (2) with chunks
            self._model.set_df_to_csv(pd.DataFrame(columns=['CaseID', 'Text']), trans_col_name, self.p_tta, s_na='', b_append=True, b_header=True)
            with open(self.p_sectors) as read_chunk:
                chunk_iter = pd.read_csv(read_chunk, chunksize=500)
                tqdm.pandas()
                for df_curr_chunk in tqdm(chunk_iter):
                    if not df_curr_chunk.empty:
                        s_curr_to_translate = df_curr_chunk[col_key].copy()
                        s_curr_to_translate = s_curr_to_translate.fillna(value='')
                        s_curr_translated = s_curr_to_translate.progress_apply(self.translate_apply, src='iw', tgt='en')

                        # (1) with validation
                        df_curr_translated = pd.DataFrame(df_curr_chunk['CaseID'], columns=['CaseID'])
                        df_curr_translated = df_curr_translated.merge(s_curr_translated, left_index=True, right_index=True)

                        # (2) no validation
                        # df_curr_translated = pd.DataFrame(s_curr_translated)

                        self._model.set_df_to_csv(df_curr_translated, trans_col_name, self.p_tta, s_na='', b_append=True, b_header=False)
            print(f'Finished Translation to English {trans_col_name}, with {self.i_invalid} invalid terms.')

    @staticmethod
    def get_chunk_index(p):
        # 4 indexes
        df_curr = pd.read_csv(p)
        length = df_curr.shape[0]
        if length < 500:
            i = 0
        elif 500 < length < 1000:
            i = 1
        elif 1000 < length < 1500:
            i = 2
        else:
            i = 3
        return i

    @staticmethod
    def remove_na_rows(df_curr, col_key):
        df_curr = df_curr[df_curr[col_key].str.strip().astype(bool)]  # deletes null
        df_curr = df_curr.reset_index(drop=True)
        df_curr = pd.DataFrame(df_curr, columns=[col_key])
        return df_curr

    def generate_aug_per_model(self):
        df_data = pd.read_csv(self._model.p_features_merged)
        # self.vocab_w2v = self.vocabulary_load('100')
        self.vocab_w2v = self.vocabulary_load('1000')
        self.i_invalid = 0
        i_row = 0
        tqdm.pandas()
        for col_key, col_values in tqdm(df_data.iteritems()):
            if col_key in self._model.l_tfdf_cols_features:
                trans_col_name = 'Translation' + col_key
                backtrans_col_name = 'BackTrans' + col_key
                syn_col_name = 'Synonym' + col_key
                syn_heb_col_name = 'SynonymHeb' + col_key
                w2v_col_name = 'w2v' + col_key
                umls_col_name = 'umls' + col_key

                df_curr = pd.DataFrame()  # replaces nan with empty string for format validation
                heb_col_srs = col_values.fillna(value='')
                df_curr[col_key] = heb_col_srs

                self.aug_translate(col_key, df_curr, trans_col_name)
                self.aug_synonym(col_key, trans_col_name, syn_col_name, syn_heb_col_name)
                self.aug_backtranslate(col_key, trans_col_name, backtrans_col_name)
                self.aug_w2v(col_key, df_curr, w2v_col_name)
                self.aug_umls(col_key, df_curr, umls_col_name)

                i_row += 1
                print(f'Done Augmentations on {col_key}.')
        self.merge_aug()  # generates merged augmentation files (pre-vectorization)

    def load_index_file(self):
        d_indexes = dict()
        p_curr = self.p_output + r'\d_indexes.pkl'
        p_indexes = self._model.set_vpn_dir(p_curr)
        if os.path.exists(p_indexes):
            d_indexes = self._model.get_pickle(self.p_output, 'd_indexes')
            print(f'Dictionary Indexes loaded (already exists).')
        return d_indexes

    def apply_tfidf(self, df_corpus, s_aug):
        p_curr = self._model.validate_path(self.p_output, s_aug, 'csv')
        if self._model.check_file_exists(p_curr):
            print(f'TF-IDF file for {s_aug} already exists.')
        else:  # sectors: 116,718 | backtrans: 167,409 | syn: 213,775 | w2v: 99,573
            # sectors: 116,718 | backtrans: 177,634 | syn: 222,669 | w2v: 99,569

            token_pattern_ = r'([a-zA-Z0-9-/]{1,})'  # remove numbers / alphanumerics
            # token_pattern_ = r'(?u)\b[A-Za-z]+\b'
            # token_pattern_ = u'(?u)\b\w*[a-zA-Z]\w*\b'

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=self.l_stopwords, max_features=self.max_features,
                                         analyzer='word', encoding='utf-8', decode_error='strict',
                                         lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False)
            s_corpus = df_corpus['Text']
            tfidf = vectorizer.fit_transform(s_corpus)
            feature_names = vectorizer.get_feature_names_out()
            tfidf = pd.DataFrame(tfidf.toarray(), columns=feature_names)
            print(f'file {s_aug}, shape: {tfidf.shape}')
            self._model.set_df_to_csv(tfidf, s_aug, self.p_output, s_na='', b_append=False, b_header=True)
            p_save_vocabulary = self._model.validate_path(self.p_output, 'tfidf_vocab_'+s_aug, 'pkl')
            pickle.dump(vectorizer.vocabulary_, open(p_save_vocabulary, 'wb'))

    def apply_tfidf_per_model(self, df_input, i_features_sectors, i_features_settings, s_aug, b_tta=False):
        # Method creates the following files:
        #
        # (1) "tfidf" TF-IDF file for all sectors + settings.
        # (2) "sectors" TF-IDF file of all sectors together.
        # (3) "settings" TF-IDF file for all settings
        # (4) "separate" TF-IDF files for each sector.
        # (5) "data" entire data put together.
        # (6) "rest" demographics and settings.

        # curr_exp_name = str(i_features_sectors) + '_' + str(i_features_sectors)  # different dirs for experiments
        # self.p_output = self._model.set_dir_get_path(self.p_output, curr_exp_name)

        df_tfidf = pd.DataFrame()

        d_indexes = self.load_index_file()

        p_load_vocabulary = self._model.validate_path(self.p_output, 'tfidf_vocab', 'pkl')
        if self._model.check_file_exists(p_load_vocabulary):
            self.vocab_tfidf = pickle.load(open(p_load_vocabulary, 'rb'))

        curr_size = 0
        i_start = 0
        p_output_aug_parent = ''

        if i_features_sectors is None:
            i_end = 0
        else:
            i_end = i_features_sectors

        if b_tta:
            p_output_aug_parent = self._model.set_dir_get_path(self.p_output, s_aug)

        tqdm.pandas()
        for col_key in tqdm(self._model.l_tfdf_cols_features):  # tf-idf applied per area
            if b_tta:
                col_key = s_aug + col_key
            col_values = df_input[col_key]
            p_file = self._model.validate_path(self.p_output, col_key, 'csv')
            col_values = col_values.fillna(value='')
            df_col_values = self.tfidf(col_values, i_features_sectors)
            l_col_values = list(df_col_values.columns)
            df_col_values[l_col_values] = df_col_values[l_col_values].astype(float)  # validates format

            df_tfidf = df_tfidf.merge(df_col_values, left_index=True, right_index=True)

            # s_start = l_col_values[0]  # causes overlapping error when saving columns with strings
            # s_end = l_col_values[len(l_col_values)-1]
            # i_start = df_tfidf.columns.get_loc(s_start)
            # i_end = df_tfidf.columns.get_loc(s_end)

            if i_features_sectors is None:
                curr_size = len(l_col_values)
                i_end += curr_size

            d_indexes[col_key] = [i_start, i_end]  # saves by index

            if i_features_sectors is None:
                i_start = i_end
            else:
                i_start = i_end
                i_end += i_features_sectors

            if b_tta:
                self._model.set_df_to_csv(df_col_values, col_key, p_output_aug_parent, s_na='', b_append=False, b_header=True)
            else:
                self._model.set_df_to_csv(df_col_values, col_key, self.p_output, s_na='', b_append=False, b_header=True)

        if not b_tta:
            df_multiple = df_input[self.l_tfdf_cols_settings].copy()  # settings in tf-idf format
            df_multiple[self.l_tfdf_cols_settings] = df_multiple[self.l_tfdf_cols_settings].astype(str)  # validates format
            df_multiple['Settings'] = df_multiple[self.l_tfdf_cols_settings].apply(lambda x: ' '.join(x), axis=1)
            df_multiple.drop(columns=self.l_tfdf_cols_settings, axis=1, inplace=True)
            df_multiple = df_multiple.fillna(value='')
            df_multiple_col_values = self.tfidf(df_multiple['Settings'], i_features_settings)
            l_multiple_col_values = list(df_multiple_col_values.columns)
            df_multiple_col_values[l_multiple_col_values] = df_multiple_col_values[l_multiple_col_values].astype(float)

            if i_features_sectors is None:
                i_end -= curr_size
                i_start = df_tfidf.shape[1]
                curr_size_settings = len(l_multiple_col_values)
                i_end = i_start + curr_size_settings
            else:
                i_end -= i_features_sectors
                i_start = (len(self._model.l_tfdf_cols_features) * i_features_sectors)
                # i_end = i_start + i_features_settings  # without tf idf settings

            self._model.set_df_to_csv(df_tfidf, 'x_sectors', self.p_output, s_na='', b_append=False, b_header=True)

            i_end = i_start + self.general_size  # Age, Gender, VariableAmount & GlucoseLevel
            d_indexes['General'] = [i_start, i_end]

            # d_indexes['Settings'] = [i_start, i_end]

            # i_start = i_end
            # i_end += self.demographic_size
            # d_indexes['Demographics'] = [i_start, i_end]

            # i_start = i_end
            # i_end += self.examination_size
            # d_indexes['Examinations'] = [i_start, i_end]

            # d_indexes['General'] = [d_indexes['Demographics'][0], d_indexes['Examinations'][1]]

            self._model.set_df_to_csv(df_multiple_col_values, 'x_settings', self.p_output, s_na='', b_append=False, b_header=True)  # settings

            self._model.set_pickle(d_indexes, self.p_output, 'd_indexes')

        else:  # if b_tta
            filename = 'x_' + s_aug
            self._model.set_df_to_csv(df_tfidf, filename, self.p_classifier, s_na='', b_append=False, b_header=True)

    def generate_aug(self):
        p_original = self._model.validate_path(self.p_output, 'sectors', 'csv')
        df_data = pd.read_csv(p_original)

        s_text = df_data.columns.tolist()[1]  # validates format
        df_data[s_text] = df_data[s_text].fillna(value='')

        p_y = self._model.validate_path(self.p_classifier, 'y', 'csv')
        y = pd.read_csv(p_y)

        # self.vocab_w2v = self.vocabulary_load('100')
        # self.vocab_w2v = self.vocabulary_load('300')
        # self.vocab_w2v = self.vocabulary_load('1000')

        # self.aug_translate(s_text, 'translation')
        # self.aug_synonym(s_text, 'translation', 'synonym', 'synonymheb')
        # self.aug_backtranslate(s_text, 'translation', 'backtrans')
        # self.aug_w2v(s_text, 'w2v')
        # self.aug_umls(s_text, 'umls')
        # self.aug_hebrew_synonym(s_text, 'synonymheb2')

        print(f'Done Augmentations.')
