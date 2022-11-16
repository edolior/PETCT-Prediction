from Model.Report import Report
import os
import pandas as pd
import numpy as np
from numpy import sort
from statistics import mean
from datetime import datetime
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import scikit_posthocs as sp
import stac

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import type_of_target
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.compose import make_column_selector
from sklearn.metrics import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.calibration import *
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.utils import *
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from fancyimpute import *
from autoimpute.imputations import MiceImputer
from autoimpute.analysis import MiLinearRegression
from bayes_opt import BayesianOptimization

import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import backend as K
# from tensorflow import layers
# from transformers import TFXLNetModel, XLNetTokenizer
# import tensorflow_addons as tfa

import SimpleITK as sitk
from skimage import io
from skimage.transform import resize
import nibabel as nib
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


# ---------------------------------------------------------------------------------------
# Classifier Class:
#
# Training & Testing Ensemble Model
#
#
# Files in output_text_aug:
#
# (1) Sectors
# (2) Demographics
# (3) Examinations
#
# Files in output_classifier:
# (1) df_filtered
# (2) df_imputed
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------
class History_Tensor(tf.keras.callbacks.Callback):
    # callback class and function adjustment for calculating test accuracy and loss

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        """
        function tracks on epochs
        :param epoch
        :param logs
        """
        loss, acc = self.model.evaluate(self.test_data)
        if 'test_loss' not in logs:
          logs['test_loss'] = []
        if 'test_accuracy' not in logs:
          logs['test_accuracy'] = []
        logs['test_loss'].append(loss)
        logs['test_accuracy'] = acc


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)  # normalizes feature vectors
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class Classifier:
    _model = None
    _report = None

    df_data = None
    df_metrics = None
    df_cms = None
    df_configs = None
    df_params = None
    df_best_fold = None

    def __init__(self, ref_model, b_tta):
        """
        Classifier Constructor
        """
        self._model = ref_model
        self._report = Report(ref_model)
        self.b_tta = b_tta
        self.p_input_parser = self._model.p_output + r'\output_parser'
        self.p_input_aug = self._model.p_output + r'\output_text_aug'
        self.p_output = self._model.set_dir_get_path(self._model.p_output, 'output_classifier')
        self.p_results = self.p_output + r'\df_results.csv'
        self.p_features = self.p_input_parser + r'\df_features_merged.csv'
        self.p_onehot = self.p_input_aug + r'\df_onehot.csv'
        self.p_tfidf = self.p_input_aug + r'\x_sectors.csv'
        self.p_sectors = self.p_input_aug + r'\sectors.csv'
        self.p_rest = self.p_input_aug + r'\df_rest.csv'
        self.p_imputed = self.p_output + r'\df_imputed.csv'
        self.p_standard = self.p_output + r'\df_standard.csv'
        self.p_filtered = self.p_output + r'\df_filtered.csv'
        self.p_ratio = self.p_output + r'\df_ratio.csv'
        self.p_data_final = self.p_output + r'\df_final.csv'
        self.p_stopwords = self._model.p_resource + r'\stopwords.csv'
        self.p_models = self._model.set_dir_get_path(self.p_output, 'models')
        self.p_plots = self._model.set_dir_get_path(self.p_output, 'plots')
        self.p_tta = self._model.set_dir_get_path(self.p_input_aug, 'TTA')
        self.p_params = self._model.set_dir_get_path(self.p_output, 'params')
        self.p_petct = self._model.set_dir_get_path(self.p_output, 'petct')

        if self._model.b_vpn:
            self.p_input_parser = self._model.set_vpn_dir(self.p_input_parser)
            self.p_input_aug = self._model.set_vpn_dir(self.p_input_aug)
            self.p_output = self._model.set_vpn_dir(self.p_output)
            self.p_results = self._model.set_vpn_dir(self.p_results)
            self.p_features = self._model.set_vpn_dir(self.p_features)
            self.p_onehot = self._model.set_vpn_dir(self.p_onehot)
            self.p_tfidf = self._model.set_vpn_dir(self.p_tfidf)
            self.p_sectors = self._model.set_vpn_dir(self.p_sectors)
            self.p_rest = self._model.set_vpn_dir(self.p_rest)
            self.p_imputed = self._model.set_vpn_dir(self.p_imputed)
            self.p_standard = self._model.set_vpn_dir(self.p_standard)
            self.p_filtered = self._model.set_vpn_dir(self.p_filtered)
            self.p_ratio = self._model.set_vpn_dir(self.p_ratio)
            self.p_data_final = self._model.set_vpn_dir(self.p_data_final)
            self.p_stopwords = self._model.set_vpn_dir(self.p_stopwords)
            self.p_models = self._model.set_vpn_dir(self.p_models)
            self.p_plots = self._model.set_vpn_dir(self.p_plots)
            self.p_tta = self._model.set_vpn_dir(self.p_tta)
            self.p_params = self._model.set_vpn_dir(self.p_params)
            self.p_petct = self._model.set_vpn_dir(self.p_petct)

        if self._model.check_file_exists(self.p_stopwords):
            self.df_stopwords = pd.read_csv(self.p_stopwords)
            l_stopwords = self.df_stopwords.values.tolist()
            self.l_stopwords = [item for sublist in l_stopwords for item in sublist]

        self.onehot_size = 12

        # self.i_stack_cv = 3
        self.i_stack_cv = 10

        # self.i_cv = 10
        self.i_cv = 5
        # self.i_cv = 3
        # self.i_cv = 2

        # self.s_metric = 'PRAUC'
        self.s_metric = 'AUC'
        self.threshold_score_probs = None
        self.i_top, self.f_top = -1, -1

        # lists #
        self.l_targets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.l_targets_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']
        self.l_demographics = ['Age', 'Gender', 'כללית', 'מכון איזוטופים יחידה ארגונית']
        self.l_examinations = ['VariableAmount', 'GlucoseLevel', 'מיפוי FDG PET גלוקוז מסומן']
        self.l_generals = ['Age', 'Gender', 'VariableAmount', 'GlucoseLevel']

        self.l_sectors = [
            'ArrivalReason',
            'BreastArmPit', 'Chest', 'Lung', 'ChestLung', 'HeadNeck', 'SkeletonTissue', 'StomachPelvis',
            'Summary'
        ]

        self.l_tta_types = ['backtrans', 'synonymheb', 'w2v', 'bert']
        self.l_test_types = ['original', 'backtrans', 'synonymheb', 'w2v', 'bert']
        self.i_datasets = len(self.l_test_types)
        self.l_formula_fold = list()

        self.l_cols_sparse = [
            'לא ידוע', 'לאומית', 'מאוחדת', 'מוסד רפאחר', 'מכבי', 'מסוק', 'פניה עצמית', 'צה"ל',
            'שרות בתי הסוהר',
            'מחלקה כירורגית', 'מחלקה כירורגית יחידה ארגונית', 'מחלקה פנימית א יחידה ארגונית',
            'מיפוי FDG PETגלוקוז מסומןCTללא חיו ב',
            'CaseID', 'Timestamp', 'TestStartTime'
        ]

        self.l_numerics = ['Age', 'VariableAmount', 'GlucoseLevel']
        self.l_categoricals = ['Gender', 'כללית', 'מכון איזוטופים יחידה ארגונית', 'מיפוי FDG PET גלוקוז מסומן']

        # self.l_models = ['ARRIVAL', 'SUMMARY', 'BREASTARMPIT', 'CHEST', 'CHESTLUNG', 'HEADNECK', 'LUNG',
        #                  'SKELETONTISSUE', 'STOMACHPELVIS', 'SETTINGS', 'DEMOGRAPHICS', 'EXAMINATIONS']

        # self.l_models = ['ARRIVAL', 'SUMMARY', 'BREASTARMPIT', 'CHEST', 'CHESTLUNG', 'HEADNECK', 'LUNG',
        #                  'SKELETONTISSUE', 'STOMACHPELVIS', 'SETTINGS', 'GENERAL']

        # self.l_models = ['ARRIVAL', 'SUMMARY', 'BREASTARMPIT', 'CHEST', 'CHESTLUNG', 'HEADNECK', 'LUNG',
        #                  'SKELETONTISSUE', 'STOMACHPELVIS', 'GENERAL']

        self.l_models = ['BREASTARMPIT', 'CHEST', 'CHESTLUNG', 'HEADNECK', 'LUNG',
                         'SKELETONTISSUE', 'STOMACHPELVIS']

        self.l_colors10 = ['red', 'yellow', 'green', 'gray', 'lightgreen', 'orange', 'lightgray', 'darkgreen', 'pink',
                           'darkgray']

        self.l_colors5 = ['red', 'yellow', 'green', 'blue', 'black']

        self.l_features, self.l_tfidf_features = list(), list()

        self.l_x_tta, self.l_tta = list(), list()

        self.l_org_indexes = list()

        self.l_types = ['original', 'backtrans', 'synonymheb', 'w2v', 'bert', 'TTAEnsemble']

        # dictionaries #
        self.d_datasets = dict()
        self.d_models = dict()
        self.d_configs = dict()
        self.d_level0 = dict()
        self.d_fold_scores = dict()
        self.d_scores = dict()

        self.d_model_results_nested = {'TestAcc': list(), 'AUC': list(), 'Precision': list(), 'Recall': list(),
                                       'F1': list(), 'PRAUC': list()}

        self.d_labels = {'A+B': list(), 'C+D': list(), 'E+F': list(), 'G': list(), 'H+I': list(), 'J': list(),
                         'K': list(), 'L': list(), 'M': list(), 'N': list()}

        # dataframes #
        self.l_cols_metrics = ['Model', 'Class', 'Fold', 'Train', 'Test', 'TestAcc', 'AUC', 'Precision', 'Recall', 'F1',
                               'PRAUC', 'Preds', 'Proba', 'YTest']
        self.l_cols_cm = [0, 1, 'Model', 'Class', 'Train', 'Test']
        self.l_cols_configs = ['Model', 'Train', 'Test', 'i_fold', 'x_train', 'y_train', 'x_test', 'y_test', 'x', 'y']
        self.l_cols_stacking = ['Class', 'Fold', 'Model', 'Test', 'AVG_Model', 'AVG_All']

        self.general = None

        self.curr_timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        self.df_metrics_name = 'df_metrics' + '_' + self.curr_timestamp
        self.df_metrics_models_name = 'df_metrics_models' + '_' + self.curr_timestamp
        self.df_metrics_targets_name = 'df_metrics_targets' + '_' + self.curr_timestamp
        self.df_cm_name = 'df_cms' + '_' + self.curr_timestamp
        self.df_configs_name = 'df_configs' + '_' + self.curr_timestamp
        self.df_stacking_name = 'df_stacking_scores' + '_' + self.curr_timestamp

        self.df_metrics = pd.DataFrame(columns=self.l_cols_metrics)
        self.df_metrics_models = pd.DataFrame(columns=self.l_cols_metrics)
        self.df_metrics_targets = pd.DataFrame()
        self.df_cms = pd.DataFrame(columns=self.l_cols_cm)
        self.df_configs = pd.DataFrame(columns=self.l_cols_configs)
        self.df_stacking = pd.DataFrame(columns=self.l_cols_stacking)
        self.df_ratios = pd.DataFrame(columns=['Label', 'Class1', 'Class0', 'B_Class1', 'B_Class0',
                                               'Count_Class1', 'Count_Class0', 'Count_B_Class1', 'Count_B_Class0'])

        self.m_xlnet = None
        self.m_tokenizer = None

        np.set_printoptions(precision=3)

    def get_models(self, s_target):
        """
        function initializes sub-models for Stacking
        :param s_target model name
        """
        self.d_models = dict()

        # self.d_models['DecisionTree'] = [
        #     DecisionTreeClassifier(criterion='entropy', splitter='best',
        #                            class_weight='balanced', max_features=None,
        #                            random_state=1),
        #     {'n_estimators': [10, 50, 100], 'max_depth': [2, 16, 64],
        #      'criterion': ['entropy', 'gini'], 'max_features': [None, 'auto']}
        #                                 ]

        # self.d_models['ExtraTree'] = [
        #     ExtraTreeClassifier(criterion='entropy', splitter='best', class_weight='balanced',
        #                         max_features=None, random_state=2),
        #     {'n_estimators': [10, 50, 100], 'max_dept': [2, 16, 64],
        #      'criterion': ['entropy', 'gini'], 'max_features': [None, 'auto']}
        #                              ]

        # self.d_models['RandomForest'] = [
        #     RandomForestClassifier(bootstrap=True, criterion='entropy', n_jobs=-1,
        #                            min_samples_split=20, class_weight='balanced',
        #                            max_features=None, random_state=3),
        #     {'n_estimators': [10, 50, 100], 'max_depth': [2, 16, 64],
        #      'criterion': ['entropy', 'gini'], 'max_features': [None, 'auto']}
        # ]

        self.d_models['XGBoost'] = [
            XGBClassifier(max_depth=4, eta=0.02, n_estimators=500, verbosity=0, objective='binary:logistic',
                          random_state=9, use_label_encoder=False),
            {'eta': [0.1, 0.05, 0.01]}
                                    ]

        # self.d_models['CatBoost'] = [
        #     CatBoostClassifier(iterations=5, learning_rate=0.1, depth=2, loss_function='CrossEntropy', random_state=5),
        #     dict()
        #                             ]

        # d_lightgbm_params = {'application': 'binary', 'objective': 'binary', 'metric': 'binary_logloss',
        #                      'is_unbalance': 'true', 'boosting_type': 'gbdt', 'num_leaves': 31, 'feature_fraction': 0.5,
        #                      'bagging_fraction': 0.5, 'bagging_freq': 20, 'learning_rate': 0.05, 'verbose': 2}
        # self.d_models['LightGBM'] = [
        #     lgb.LGBMClassifier(**d_lightgbm_params, max_depth=6, random_state=6),
        #     dict()
        # ]

        # self.d_models['SVC'] = [
        #     SVC(C=100, gamma=0.2, random_state=7),
        #     {'C': [1, 10, 100], 'gamma': [0.001, 0.5], 'kernel': ['rbf', 'linear']}
        #                        ]

        # self.d_models['KNN'] = [
        #     KNeighborsClassifier(n_neighbors=3),
        #     {"n_neighbors": [3, 5, 10]}
        #                        ]

        # self.d_models['GB'] = [
        #     GradientBoostingClassifier(n_estimators=100, max_leaf_nodes=None, max_depth=10,
        #                                random_state=9, min_samples_split=5),
        #     {"n_estimators": [10, 50, 100], "max_depth": [2, 16, 64]}
        #                       ]

        # self.d_models['MLP'] = [
        #     MLPClassifier(solver='lbfgs', random_state=10),
        #     {'hidden_layer_sizes': [(64, 32, 16), (32, 32, 16), (64, 32, 32), (62, 62, 32, 32)],
        #      'alpha': [0.01, 0.05, 0.005, 0.001, 0.0001]}
        #                       ]

        o_stacking_model = self.get_stacking()
        s_stacking_name = 'Stacking'+s_target
        return o_stacking_model, s_stacking_name

    def set_call_backs(self):
        """
        function sets call backs: early stopping / learning rate reduction pace
        """
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02,
                                             restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(self.warmup, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0,
                                                 mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6)
        ]

    def warmup(self, epoch, lr):
        """
        function helper for tokenization
        :param epoch
        :param lr learning rate
        """
        return max(lr + 1e-6, 2e-5)

    def get_inputs(self, text, max_len=512):
        """
        function tokenizes attention model text input
        :param text input
        :param max_len sequence length of iteration
        """
        inps = [self.m_tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in
                text]
        inp_tok = np.array([a['input_ids'] for a in inps])  # gets tensors from text using tokenizer
        ids = np.array([a['attention_mask'] for a in inps])
        segments = np.array([a['token_type_ids'] for a in inps])
        return inp_tok, ids, segments

    def load_xlnet(self):
        """
        function initializes xlnet model
        """
        # (6) nlp-aug: XLNET
        # (6.1)
        # self.m_xlnet = naw.ContextualWordEmbsAug(model_path='xlnet', aug_p=0.1)  # tf 2.3

        # (6.2) loads pretrained model
        # p_spiece = self.p_xlnet + '/spiece.model'
        # self.m_tokenizer = Tokenizer(p_spiece)
        # self.m_xlnet = load_trained_model_from_checkpoint(
        #     config_path=os.path.join(self.p_xlnet, 'xlnet_config.json'),
        #     checkpoint_path=os.path.join(self.p_xlnet, 'xlnet_model.ckpt'),
        #     batch_size=16,
        #     memory_len=512,
        #     target_len=128,
        #     in_train_phase=False,
        #     attention_type=ATTENTION_TYPE_BI,
        # )

        # (6.3) loads import model
        i_classes = 2
        mname = 'xlnet-large-cased'
        # input_shape = 120
        input_shape = 512

        self.m_tokenizer = XLNetTokenizer.from_pretrained(mname)
        word_inputs = tf.keras.Input(shape=(input_shape,), name='word_inputs', dtype='int32')
        self.m_xlnet = TFXLNetModel.from_pretrained(mname)
        xlnet_encodings = self.m_xlnet(word_inputs)[0]
        doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
        doc_encoding = tf.keras.layers.Dropout(0.1)(doc_encoding)
        outputs = tf.keras.layers.Dense(i_classes, activation='sigmoid', name='outputs')(doc_encoding)
        self.m_xlnet = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
        self.m_xlnet.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002), loss='binary_crossentropy',
                             metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        self.m_xlnet.summary()

    def train_xlnet(self, x_train, y_train, curr_label, epochs, batch_size, i_val_split):
        """
        function trains xlnet model
        :param x_train
        :param y_train
        :param curr_label
        :param epochs
        :param batch_size
        :param i_val_split ratio to split validation set
        """
        inp_tok, ids, segments = self.get_inputs(x_train)
        history = self.m_xlnet.fit(x=inp_tok, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=i_val_split)
        # self.m_xlnet.save_weights(self.p_input + '/xlnet.h5')
        # self.save_model(self.m_xlnet, curr_label)

    def infer_xlnet(self, x_test, y_test, s_target):
        """
        function infers xlnet model
        :param x_test
        :param y_test
        :param s_target
        """
        # self.m_xlnet = self.load_model(s_target, i_fold)
        inp_tok, ids, segments = self.get_inputs(x_test)
        y_preds = self.m_xlnet.predict(inp_tok, verbose=True)
        return y_preds

    def plot_metrics(self, y_preds, y_test):
        """
        function calculates and plots xlnet results
        :param y_preds
        :param y_test
        """
        acc = accuracy_score(y_test, np.array(y_preds.flatten() >= .5, dtype='int'))
        fpr, tpr, thresholds = roc_curve(y_test, y_preds)
        auc = roc_auc_score(y_test, y_preds)
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.plot(fpr, tpr, color='red')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        s_title = f'AUC: {auc}'
        ax.set_title(s_title)
        ax.show()
        self.save_plot(fig, s_title)

    def remove_sparse_features(self, f_threshold_na, d_remove):
        """
        function removes sparse features
        :param f_threshold_na threshold of missing values to remove
        :param d_remove dict of force remove
        """
        if self._model.check_file_exists(self.p_onehot):
            print(f'Chosen threshold for feature removal: {f_threshold_na*100}%')
            df_filtered = pd.read_csv(self.p_onehot)
            l_columns = list(df_filtered)
            i_total = len(df_filtered)
            for curr_col in l_columns:
                if curr_col in self.l_sectors:
                    continue
                if curr_col in self.l_targets_merged:
                    continue
                elif curr_col in d_remove:
                    df_filtered.drop(curr_col, axis=1, inplace=True)
                    print(f'Feature {curr_col} removed, force.')
                else:
                    curr_feature = df_filtered[curr_col]
                    i_null = curr_feature.isna().sum()  # True for None, np.nan
                    f_missing = i_null / i_total  # by missing values
                    if f_missing > f_threshold_na:
                        df_filtered.drop(curr_col, axis=1, inplace=True)
                        print(f'Feature {curr_col} removed, contains only {(1-f_missing)*100}% of data.')
                    elif df_filtered[curr_col].dtype == 'object':  # if str type column
                        i_space = (df_filtered[curr_col].values == '').sum()  # True for whitespaces
                        f_missing_spaces = i_space / i_total  # by missing values
                        if f_missing_spaces > f_threshold_na:
                            df_filtered.drop(curr_col, axis=1, inplace=True)
                            print(f'Feature {curr_col} removed, contains only {(1-f_missing_spaces)*100}% of data.')
                    else:  # if binary type columns
                        if df_filtered[curr_col].dtype == 'int64' or df_filtered[curr_col].dtype == 'int':
                            if curr_col != 'Gender':
                                i_sum_values = curr_feature.sum()  # by existing values
                                f_existing = i_sum_values / i_total
                                if f_existing < f_threshold_na:
                                    df_filtered.drop(curr_col, axis=1, inplace=True)
                                    print(f'Feature {curr_col} removed, contains only {f_existing*100}% of data.')
            self._model.set_df_to_csv(df_filtered, 'df_filtered', self.p_output, s_na='', b_append=False, b_header=True)

    def get_models_per_model(self):
        """
        function initializes sub-models
        """
        for i_sub_model in range(len(self.l_models)):
            self.d_models[self.l_models[i_sub_model]] = [
                XGBClassifier(max_depth=4, eta=0.02, n_estimators=500, verbosity=0, objective='binary:logistic',
                              random_state=i_sub_model + 1, use_label_encoder=False, scale_pos_weight=10),
                {'eta': [0.1, 0.05, 0.01]}
            ]

    def init(self):
        """
        function inits gpu settings
        """
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print(
                '\n\nThis error most likely means that this notebook is not '
                'configured to use a GPU.  Change this in Notebook Settings via the '
                'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
            raise SystemError('GPU device not found')

        with tf.device('/device:GPU:0'):
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            random_image_gpu = tf.random.normal((100, 100, 100, 3))
            net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
            return tf.math.reduce_sum(net_gpu)

    def save_model(self, o_model, s_model, i_top_fold):
        """
        function saves model to disk
        :param o_model model object
        :param s_model model name
        :param i_top_fold top fold
        """
        p_save = self.p_models + '\\' + s_model + '.pkl'
        if self._model.b_vpn:
            p_save = self.p_models + '/' + s_model + '.pkl'
        joblib.dump(o_model, p_save)
        print(f'Saved new top score model for class {s_model} (fold: {i_top_fold}).')

    def load_model(self, curr_label):
        """
        function loads model from disk
        :param curr_label model name
        """
        p_load = self.p_models + '\\' + curr_label + '.pkl'
        if self._model.b_vpn:
            p_load = self.p_models + '/' + curr_label + '.pkl'
        o_model = joblib.load(p_load)
        filename = self._model.get_filename(p_load)
        print(f'Model {filename} has been loaded.')
        return o_model

    def plot_confusion_matrix(self, y_preds, y_test, label, l_targets, s_model):
        """
        function runs confusion matrix calculations
        :param y_preds
        :param y_test
        :param label
        :param l_targets list of column names
        :param s_model model name
        """
        # cm = multilabel_confusion_matrix(y_test, y_preds)
        cm = confusion_matrix(y_test, y_preds)

        tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        df_cm = pd.DataFrame(cm, index=l_targets, columns=l_targets)

        # fig = plt.figure()
        # plt.figure(figsize=(5, 4))
        # sns.heatmap(df_cm, annot=True, fmt="g")  # fmt=".1f"
        # s_title = 'Confusion Matrix Model ' + s_model + ' For Class ' + label
        # plt.title(s_title)
        # plt.ylabel('Actual Values')
        # plt.xlabel('Predicted Values')
        # plt.show()
        # self.save_plot(fig, s_title)

        df_cm['Model'] = s_model
        df_cm['Class'] = label
        self.df_cms = self.df_cms.append(df_cm, ignore_index=True)

        return df_cm

    def save_plot(self, fig, s_title):
        """
        function saves plot to disk
        :param fig figure object
        :param s_title title name of figure
        """
        p_save = self.p_plots + '\\' + s_title + '.png'
        if self._model.b_vpn:
            p_save = self.p_plots + '/' + s_title + '.png'

        # if not self._model.check_file_exists(p_save):
        #     fig.savefig(p_save)

        # fig.savefig(p_save)
        fig.savefig(p_save, dpi=300)
        # fig.savefig(p_save, dpi=600)

    def get_stacking(self):
        """
        function initializes Stacking Ensemble Model
        """
        level0 = list()
        for s_model, o_model in self.d_models.items():
            o_builder = o_model[0]
            level0.append((s_model, o_builder))
        level1 = LogisticRegression()
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=self.i_stack_cv)
        l_model = [model, dict()]
        self.d_models['Stacking'] = l_model
        return l_model

    @staticmethod
    def validate_model(o_model, s_model):
        """
        function validates model settings
        :param o_model model object
        :param s_model model name
        """
        if hasattr(o_model, 'n_classes_'):
            if o_model.n_classes_ != 2:
                o_model.n_classes_ = 2
                o_model.classes_ = np.array([0, 1])
                if 'LightGBM' in s_model:
                    o_model.objective = 'binary'
                else:  # XGBoost
                    o_model.objective = 'binary:logistic'
        return o_model

    @staticmethod
    def get_lightgbm(x_train, y_train, x_test):
        """
        function initializes LightGBM Model
        :param x_train
        :param y_train
        :param x_test
        """
        lgb_train = lgb.Dataset(x_train, y_train)
        # lgb_val = lgb.Dataset(x_val, y_val)
        parameters = {
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'verbose': 0
        }
        # m_light_gbm = lgb.train(parameters, lgb_train, valid_sets=lgb_val, num_boost_round=5000, early_stopping_rounds=100)
        m_light_gbm = lgb.train(parameters, lgb_train, num_boost_round=5000, early_stopping_rounds=100)
        y_preds = m_light_gbm.predict(x_test, num_iteration=m_light_gbm.best_iteration)
        return m_light_gbm, y_preds

    @staticmethod
    def validate_type(df_data):
        """
        function validates features types
        :param df_data dataframe to validate its types
        """
        for col, row in df_data.iterrows():
            if col.dtype == 'object':
                df_data[col] = df_data[col].astype(str)
            if col.dtype == 'int' or col.dtype == 'int32' or col.dtype == 'int64':
                df_data[col] = df_data[col].astype(int)
            if col.dtype == 'float' or col.dtype == 'float32' or col.dtype == 'float64':
                df_data[col] = df_data[col].astype(float)
        return df_data

    def set_column_names(self, l_org_cols, ohe):
        """
        function returns feature names
        :param l_org_cols list of the original columns
        :param ohe one hot vector object
        """
        train_feature_names = ohe.get_feature_names_out(l_org_cols)
        # train_feature_names = ohe.feature_names_in_.tolist()
        for i in range(len(train_feature_names)):
            feature = train_feature_names[i]
            i_del = feature.rfind('_')
            train_feature_names[i] = feature[i_del + 1:]
        return train_feature_names

    def onehot_encoder(self, df_curr, df_test_curr=None):
        """
        function performs one hot encoding
        :param df_curr is performed one hot encoding on
        :param df_test_curr is performed one hot encoding appropriately to the training set one hot features
        """
        df_test_ohe = None
        df_train_ohe = df_curr.copy()
        l_org_train_cols = df_train_ohe.columns.tolist()
        ohe = OneHotEncoder(handle_unknown='ignore')

        if df_test_curr is not None:
            df_test_ohe = df_test_curr.copy()

        for curr_col in df_train_ohe.columns.tolist():
            if df_train_ohe[curr_col].dtype == object or '_id' in curr_col:
                train_col_ohe = ohe.fit_transform(df_train_ohe[curr_col].values.reshape(-1, 1))
                l_train_ohe_cols = self.set_column_names([curr_col], ohe)
                df_col_train_ohe = pd.DataFrame(train_col_ohe.toarray(), columns=l_train_ohe_cols)
                df_train_ohe.drop(curr_col, axis=1, inplace=True)
                df_train_ohe = df_train_ohe.merge(df_col_train_ohe, left_index=True, right_index=True)

                if df_test_ohe is not None:  # (1) one hot encoding test features that are in train
                    if curr_col in df_test_ohe.columns.tolist():
                        test_col_ohe = ohe.transform(df_test_ohe[curr_col].values.reshape(-1, 1))
                        l_test_ohe_cols = self.set_column_names([curr_col], ohe)
                        df_col_test_ohe = pd.DataFrame(test_col_ohe.toarray(), columns=l_test_ohe_cols)
                        df_test_ohe.drop(curr_col, axis=1, inplace=True)
                        df_test_ohe = df_test_ohe.merge(df_col_test_ohe, left_index=True, right_index=True)
                    else:  # (2) adding dummy category features from train that werent in test
                        for new_train_ohe_col in l_train_ohe_cols:
                            df_test_ohe[new_train_ohe_col] = 0

        if df_test_ohe is not None:
            for curr_col in df_test_ohe.columns.tolist():  # removes test categorical features that are not in train
                if df_test_ohe[curr_col].dtype == object or '_id' in curr_col and curr_col not in l_org_train_cols:
                    df_test_ohe.drop(curr_col, axis=1, inplace=True)

        if df_test_ohe is not None:
            return df_train_ohe, df_test_ohe
        else:
            return df_train_ohe

    def impute_mice(self, curr_general_train, curr_general_test, curr_y_train):
        """
        function performs imputation by: Multiple-Imputation
        :param curr_general_train is performed imputation on
        :param curr_general_test is performed imputation appropriately to the training set imputation
        :param curr_y_train prediction helper
        """
        general_train = curr_general_train.copy()
        general_test = curr_general_test.copy()
        y_train = curr_y_train.copy()
        y_train = y_train.to_frame()
        custom_imputer = MiceImputer(n=3, strategy="pmm", return_list=True)
        complex_lm = MiLinearRegression(mi=custom_imputer, model_lib="statsmodels")  # imputes passed to linear reg
        for curr_col in general_train.columns.tolist():
            i_na = general_train[curr_col].isnull().sum()
            if i_na > 0:
                srs_imp_train = general_train[curr_col].to_frame()
                complex_lm.fit(srs_imp_train, y_train)
                x_general_train = complex_lm.predict(srs_imp_train)
                general_train.loc[:, curr_col] = x_general_train
                srs_imp_test = general_test[curr_col].values.reshape(-1, 1)
                x_general_test = complex_lm.predict(srs_imp_test)
                general_test.loc[:, curr_col] = x_general_test
        general_train.reset_index(inplace=True, drop=True)
        general_test.reset_index(inplace=True, drop=True)
        return general_train, general_test

    def impute_knn(self, curr_general_train, curr_general_test):
        """
        function performs imputation by: KNN
        :param curr_general_train is performed imputation on
        :param curr_general_test is performed imputation appropriately to the training set imputation
        """
        general_train = curr_general_train.copy()
        general_test = curr_general_test.copy()
        imp_knn = KNNImputer(n_neighbors=20, weights="uniform")
        for curr_col in general_train.columns.tolist():
            i_na = general_train[curr_col].isnull().sum()
            if i_na > 1:
                srs_imp_train = general_train[curr_col].values.reshape(-1, 1)
                x_general_train = imp_knn.fit_transform(srs_imp_train)
                general_train.loc[:, curr_col] = x_general_train
                srs_imp_test = general_test[curr_col].values.reshape(-1, 1)
                x_general_test = imp_knn.transform(srs_imp_test)
                general_test.loc[:, curr_col] = x_general_test
        general_train.reset_index(inplace=True, drop=True)
        general_test.reset_index(inplace=True, drop=True)
        return general_train, general_test

    def impute_mean(self, curr_general_train, curr_general_test):
        """
        function performs imputation by: Mean
        :param curr_general_train is performed imputation on
        :param curr_general_test is performed imputation appropriately to the training set imputation
        """
        general_train = curr_general_train.copy()
        general_test = curr_general_test.copy()
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        for curr_col in general_train.columns.tolist():
            i_na = general_train[curr_col].isnull().sum()
            if i_na > 1:
                srs_imp_train = general_train[curr_col].values.reshape(-1, 1)
                x_general_train = imp_mean.fit_transform(srs_imp_train)
                general_train.loc[:, curr_col] = x_general_train
                srs_imp_test = general_test[curr_col].values.reshape(-1, 1)
                x_general_test = imp_mean.transform(srs_imp_test)
                general_test.loc[:, curr_col] = x_general_test
        general_train.reset_index(inplace=True, drop=True)
        general_test.reset_index(inplace=True, drop=True)
        return general_train, general_test

    def get_model_name(self, path):
        """
        function returns model name
        :param path of model
        """
        if self._model.b_vpn:
            index1 = path.rfind('/')
            index2 = path.rfind('.')
        else:
            index1 = path.rfind('\\')
            index2 = path.rfind('.')
        s_model = path[index1+1:index2]
        return s_model

    def init_models(self, b_sample=False):
        """
        function for loading datasets to insert to the models
        :param b_sample when using sampled records
        """
        sectors, general, x, y = None, None, None, None

        p_general = self._model.validate_path(self.p_input_aug, 'x_general', 'csv')
        if self._model.check_file_exists(p_general):
            if b_sample:
                general = self.read_csv_sample(p_general, 500)
            else:
                general = pd.read_csv(p_general)

        else:
            df_filtered = pd.read_csv(self.p_filtered)
            general = df_filtered[self.l_generals].copy()
            self._model.set_df_to_csv(general, 'x_general', self.p_input_aug, s_na='', b_append=False, b_header=True)

        p_sectors = self._model.validate_path(self.p_input_aug, 'sectors', 'csv')
        if self._model.check_file_exists(p_sectors):
            # self._model.get_csv_to_df_header_only(p_sectors)
            if b_sample:
                sectors = self.read_csv_sample(p_sectors, 500)
            else:
                sectors = pd.read_csv(p_sectors)

            x = pd.DataFrame(sectors['Text'], columns=['Text'])

        p_y = self._model.validate_path(self.p_output, 'y', 'csv')
        if self._model.check_file_exists(p_y):
            if b_sample:
                y = self.read_csv_sample(p_y, 500)
            else:
                y = pd.read_csv(p_y)

        else:
            df_data = pd.read_csv(self.p_filtered)
            y = df_data[self.l_targets_merged].copy()
            self.get_data_counts(y)
            self._model.set_df_to_csv(y, 'y', self.p_output, s_na='', b_append=False, b_header=True)

        if self.b_tta:
            self.l_x_tta = self.init_models_tta(b_sample)
            l_curr_x_mrg = list()
            for curr_tta in self.l_x_tta:
                l_curr_x_mrg.append(curr_tta)
            self.l_x_tta = l_curr_x_mrg.copy()
        else:
            l_curr_x_mrg = None

        return x, general, y, l_curr_x_mrg

    def show_values(self, axs, results, orient, space=.01):
        """
        function displays auc score text on top of plots
        :param axs plot object
        :param results float auc scores
        :param orient type of plot: bar/box/scatter
        :param space between shape and text
        """
        def _single(ax):
            if orient == 'bar':
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = '{:.1f}'.format(p.get_height())
                    ax.text(_x, _y, value, ha="center")

            if orient == 'box':
                # nobs = list(self.d_fold_scores.items())
                nobs = list(self.d_fold_scores.values())

                medians = list()
                for i in range(len(nobs)):
                    l_class_fold_scores = nobs[i]
                    curr_class_mean = mean(l_class_fold_scores)
                    curr_class_mean = curr_class_mean * 100
                    curr_class_mean = round(curr_class_mean, 4)
                    medians.append(curr_class_mean)

                # vertical_offset = list()
                # for i in range(len(nobs)):
                #     l_class_fold_scores = nobs[i]
                #     curr_class_mean = mean(l_class_fold_scores)
                #     curr_class_mean = curr_class_mean * 100
                #     curr_class_mean = curr_class_mean * 0.03
                #     curr_class_mean = round(curr_class_mean, 4)
                #     vertical_offset.append(curr_class_mean)

                medians = [round(mean(x), 4) for x in nobs]
                vertical_offset = [round(mean(x) * 0.03, 2) for x in nobs]  # offset from median for display

                for xtick in ax.get_xticks():
                    ax.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick],
                            horizontalalignment='center', size='x-small', color='black', weight='semibold')

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)

    def plot(self, df_results, l_names, s_x_axis, s_y_axis, s_title):
        """
        function displays results
        :param df_results df of prediction results
        :param l_names list of model names
        :param s_x_axis graph x axis title
        :param s_y_axis graph y axis title
        :param s_title graph main title
        """
        sns.set_theme(style="whitegrid")
        s_title = s_y_axis + ' ' + s_title

        # bar plot #
        # plot_type = 'bar'
        # sns_plot = sns.barplot(data=df_results)

        # count plot #
        # plot_type = 'bar'
        # sns_plot = sns.countplot(
        #                    x=s_x_axis,
        #                    y=s_y_axis,
        #                    data=df_results
        #                    )
        # sns_plot.bar_label(sns_plot.containers[0])

        # box plot #
        plot_type = 'box'
        # l_colors = ['#78C850', '#F08030', '#6890F0', '#F8D030', '#F85888', '#705898']

        # sns_plot = sns.boxplot(
        #             x=s_x_axis,
        #             y=s_y_axis,
        #             hue='Datasets',
        #             # palette=l_colors,
        #             showmeans=True,
        #             data=df_results
        # )

        sns_plot = sns.boxplot(data=df_results, showfliers=False)
        self.show_values(sns_plot, df_results, plot_type)
        plt.title(s_title)
        plt.xticks(rotation=45)
        plt.show()
        fig = plt.figure()
        self.save_plot(fig, s_title)

    def plot_results(self, l_results, l_names, s_title, s_axis):
        """
        function displays results
        :param l_results list of prediction results
        :param l_names list of model names
        :param s_title graph main title
        :param s_axis graph axis title
        """
        fig = plt.figure()
        if s_axis == 'AUC':
            sns.set_theme(style="darkgrid")
            sns.scatterplot(data=l_results, sizes=(30, 200), legend='brief').set(title=s_axis + ' ' + s_title)
            # sns.scatterplot(data=l_results, x='Class', y='AUC', sizes=(30, 200), legend='brief').set(title=s_axis + ' ' + s_title)
        else:
            fig.suptitle(s_title, fontsize=20)  # v1
            plt.ylabel(s_axis, fontsize=16)
            plt.xticks(rotation=45, ha='right')
            l_elements = list(range(1, len(l_names) + 1))
            plt.bar(l_elements, l_results, tick_label=l_names, width=0.8, color=self.l_colors5)
            for i in range(len(l_results)):
                l_results[i] = float("{:.3f}".format(l_results[i]))
                plt.annotate(str(l_results[i]), xy=(l_elements[i], l_results[i]), ha='center', va='bottom')

            # box = plt.boxplot(l_results, labels=l_names, showmeans=True)  # v2
            # # ax = sns.boxplot(x=l_names, y="AUC score", data=l_results)
            # i_box = 0
            # l_caps = box['caps']
            # for line in box['medians']:
            #     curr_caps = l_caps[i_box]
            #     y_top = curr_caps.get_ydata()[0]
            #     x, y = line.get_xydata()[1]  # top of median line
            #     plt.text(x, y, '%.2f' % y, horizontalalignment='center')  # draw above and centered
            #     i_box += 1

        plt.show()
        self.save_plot(fig, s_title)

    def get_indexes(self, x):
        """
        function reads range of indexes from saved pickle of index dictionary
        :param x
        """
        d_indexes = self._model.get_pickle(self.p_input_aug, 'd_indexes')
        # d_indexes['General'] = [d_indexes['General'][0], d_indexes['General'][1] - 1]  # range fix
        d_indexes['StomachPelvis'] = [d_indexes['StomachPelvis'][0], d_indexes['StomachPelvis'][1] - 1]  # range fix
        d_features = {
            # 'ARRIVAL': x.iloc[:, d_indexes['ArrivalReason'][0]:d_indexes['ArrivalReason'][1]],
            # 'SUMMARY': x.iloc[:, d_indexes['Summary'][0]:d_indexes['Summary'][1]],
            'BREASTARMPIT': x.iloc[:, d_indexes['BreastArmPit'][0]:d_indexes['BreastArmPit'][1]],
            'CHEST': x.iloc[:, d_indexes['Chest'][0]:d_indexes['Chest'][1]],
            'CHESTLUNG': x.iloc[:, d_indexes['ChestLung'][0]:d_indexes['ChestLung'][1]],
            'HEADNECK': x.iloc[:, d_indexes['HeadNeck'][0]:d_indexes['HeadNeck'][1]],
            'LUNG': x.iloc[:, d_indexes['Lung'][0]:d_indexes['Lung'][1]],
            'SKELETONTISSUE': x.iloc[:, d_indexes['SkeletonTissue'][0]:d_indexes['SkeletonTissue'][1]],
            'STOMACHPELVIS': x.iloc[:, d_indexes['StomachPelvis'][0]:d_indexes['StomachPelvis'][1]],
            # 'SETTINGS': x.iloc[:, d_indexes['Settings'][0]:d_indexes['Settings'][1]],
            # 'DEMOGRAPHICS': x.iloc[:, d_indexes['Demographics'][0]:d_indexes['Demographics'][1]],
            # 'EXAMINATIONS': x.iloc[:, d_indexes['Examinations'][0]:d_indexes['Examinations'][1]],
            # 'GENERAL': x.iloc[:, d_indexes['General'][0]:d_indexes['General'][1]-1]
        }
        return d_indexes

    def down_sample(self, x_curr, y_curr, s_target):
        """
        function performs down-sampling: fixes ration by erasing occurrences of non-diseased patients
        :param x_curr
        :param y_curr
        :param s_target class name
        """
        threshold_ratio = 10
        df_majority, df_minority = None, None

        i_ratio1_new, i_ratio0_new = None, None
        i_class1_ratio_new, i_class0_ratio_new = None, None
        i_range = None

        if isinstance(y_curr, np.ndarray):
            df_majority = y_curr[y_curr == 0]
            df_minority = y_curr[y_curr == 1]
        elif isinstance(y_curr, pd.Series):
            df_majority = y_curr.where(y_curr == 0)
            df_minority = y_curr.where(y_curr == 1)
        elif isinstance(y_curr, pd.DataFrame):
            df_majority = y_curr[y_curr[s_target] == 0]
            df_minority = y_curr[y_curr[s_target] == 1]

        i_ratio1 = df_minority.shape[0]
        i_ratio0 = df_majority.shape[0]
        i_class1_ratio = int(i_ratio1/i_ratio1)
        i_class0_ratio = i_ratio0/i_ratio1
        i_diff_ratio = i_class0_ratio - i_class1_ratio
        i_class0_ratio = float("{:.2f}".format(i_class0_ratio))
        print(f'Current ratio for label {s_target} Class 1:Class 0 is: \n {i_class1_ratio}:{i_class0_ratio}')

        if i_diff_ratio > threshold_ratio:  # 1:100 imbalance -> 1:10 | {'E+F': 30, 'G': 12, 'K': 20, 'M': 131, 'N': 18}

            if i_ratio1 < 100:
                # (1.1) enables training by augmentations
                # {'E+F': [53, 1671], 'K': [80, 1644], 'M': [13, 1711], 'N': [88, 1636]}  -> overfits
                i_split = i_ratio0 // threshold_ratio
                i_range = i_split - i_ratio1

                # (1.2) over sample (generates new class1 minority samples -> overfits)
                # smote over: randomly duplicating examples from the minority class and adding them
                # f_over = 1 / threshold_ratio
                # over_sample = SMOTE(sampling_strategy=f_over)  # over_sample = RandomOverSampler(sampling_strategy=f_over)
                # x_new, y_new = over_sample.fit_resample(x_curr, y_curr)

            # (2) smote random under: randomly selecting examples from majority class to delete
            f_under = 1 / threshold_ratio
            under_sample = RandomUnderSampler(sampling_strategy=f_under)
            x_new, y_new = under_sample.fit_resample(x_curr, y_curr)

            # (3) smote over + under:
            # oversample minority class (class=1) to have 10% of the majority class (class=0) (class1)1:10(class0)
            # undersample to reduce the number of examples in the majority class (class=0) to have 50% more than the minority class (class=1) (class1)1:1.5(class0)
            # f_over = 1 / threshold_ratio
            # f_under = 2 / threshold_ratio
            # over = SMOTE(sampling_strategy=f_over)
            # under = RandomUnderSampler(sampling_strategy=f_under)
            # steps = [('o', over), ('u', under)]
            # pipeline = Pipeline(steps=steps)
            # x_new, y_new = pipeline.fit_resample(x_curr, y_curr)

            # (4) resample
            # majority_size = int(df_minority.shape[0] * ratio)  # erases zeros
            # df_majority_downsampled = resample(df_majority,
            #                                    replace=False,  # sample without replacement
            #                                    n_samples=majority_size,  # how much remains to match minority class
            #                                    random_state=None)  # reproducible results
            # downsampled = pd.concat([df_majority_downsampled, df_minority])
            # downsampled = df_majority_downsampled.merge(df_minority, left_index=True, right_index=True)
            # x_new = x_curr.iloc[downsampled.index, ]
            # x_new.reset_index(inplace=True, drop=True)
            # df_y_curr = pd.DataFrame(data=y_curr)
            # y_new = df_y_curr.iloc[downsampled.index, ]
            # y_new.reset_index(inplace=True, drop=True)

            i_ratio1_new = np.sum(y_new)
            i_ratio0_new = y_new.size - i_ratio1_new
            i_class1_ratio_new = int(i_ratio1_new / i_ratio1_new)
            i_class0_ratio_new = i_ratio0_new / i_ratio1_new
            i_class0_ratio_new = float("{:.2f}".format(i_class0_ratio_new))
            print(f'New ratio for label {s_target} Class 1:Class 0 is: \n {i_class1_ratio_new}:{i_class0_ratio_new}')

            if isinstance(y_new, pd.Series):
                y_new = pd.DataFrame(data=y_new, columns=[s_target])

            l_new_indexes = under_sample.sample_indices_.tolist()
            self.set_indexes(l_new_indexes)

            x_new.reset_index(inplace=True, drop=True)  # takes affect if over-sampling is applied
            y_new.reset_index(inplace=True, drop=True)
        else:
            x_new, y_new = x_curr, y_curr

        self.df_ratios = self.df_ratios.append({'Label': s_target, 'Class1': i_class1_ratio, 'Class0': i_class0_ratio,
                                                'B_Class1': i_class1_ratio_new, 'B_Class0': i_class0_ratio_new,
                                                'Count_Class1': i_ratio1, 'Count_Class0': i_ratio0,
                                                'Count_B_Class1': i_ratio1_new, 'Count_B_Class0': i_ratio0_new
                                                },
                                               ignore_index=True)
        return x_new, y_new, i_range

    def set_indexes(self, element_indexes, b_org_indexes=False):
        """
        function sets index of records before randomization to keep original indexes in a list
        :param element_indexes original index order of records
        :param b_org_indexes boolean flag for re-setting indexes of randomizes records back to the original order
        """
        l_indexes = None
        if isinstance(element_indexes, pd.DataFrame):
            element_indexes = element_indexes.index
            element_indexes = element_indexes.to_numpy()
            if b_org_indexes:
                self.l_org_indexes = element_indexes.tolist()
        if not isinstance(element_indexes, list):
            l_indexes = element_indexes.tolist()
        else:
            l_indexes = element_indexes
        for i in range(len(self.l_x_tta)):
            self.l_x_tta[i] = self.l_x_tta[i].reindex(l_indexes)
            self.l_x_tta[i].reset_index(inplace=True, drop=True)
        self.general = self.general.reindex(l_indexes)
        self.general.reset_index(inplace=True, drop=True)

    def get_data_counts(self, y):
        """
        function calculates class ratios
        :param y
        """
        s_target_count = y[self.l_targets_merged].sum()
        for index, val in s_target_count.iteritems():
            print(index, val)

    @staticmethod
    def get_transformers(d_indexes):
        """
        function initializes sub-models of each sector
        :param d_indexes dict of index range of the entire dataframe
        """
        d_transformers = {
            # 'ARRIVAL': ColumnTransformer(
            #     transformers=[('ARRIVAL', 'passthrough', d_indexes['ArrivalReason']), ], remainder='drop', ),
            # 'SUMMARY': ColumnTransformer(
            #     transformers=[('SUMMARY', 'passthrough', d_indexes['Summary']), ], remainder='drop', ),
            'BREASTARMPIT': ColumnTransformer(
                transformers=[('BREASTARMPIT', 'passthrough', d_indexes['BreastArmPit']), ], remainder='drop', ),
            'CHEST': ColumnTransformer(transformers=[('CHEST', 'passthrough', d_indexes['Chest']), ],
                                       remainder='drop', ),
            'CHESTLUNG': ColumnTransformer(transformers=[('CHESTLUNG', 'passthrough', d_indexes['ChestLung']), ],
                                           remainder='drop', ),
            'HEADNECK': ColumnTransformer(transformers=[('HEADNECK', 'passthrough', d_indexes['HeadNeck']), ],
                                          remainder='drop', ),
            'LUNG': ColumnTransformer(transformers=[('LUNG', 'passthrough', d_indexes['Lung']), ],
                                      remainder='drop', ),
            'SKELETONTISSUE': ColumnTransformer(
                transformers=[('SKELETONTISSUE', 'passthrough', d_indexes['SkeletonTissue']), ],
                remainder='drop', ),
            'STOMACHPELVIS': ColumnTransformer(
                transformers=[('STOMACHPELVIS', 'passthrough', d_indexes['StomachPelvis']), ], remainder='drop', ),
            # 'SETTINGS': ColumnTransformer(transformers=[('SETTINGS', 'passthrough', d_indexes['Settings']), ],
            #                               remainder='drop', ),
            # 'DEMOGRAPHICS': ColumnTransformer(
            #     transformers=[('DEMOGRAPHICS', 'passthrough', d_indexes['Demographics']), ], remainder='drop', ),
            # 'EXAMINATIONS': ColumnTransformer(
            #     transformers=[('EXAMINATIONS', 'passthrough', d_indexes['Examinations']), ], remainder='drop', )
            # 'GENERAL': ColumnTransformer(
            #     transformers=[('GENERAL', 'passthrough', d_indexes['General']), ], remainder='drop', )
        }
        return d_transformers

    def set_ensemble_model(self, d_transformers, s_target):
        """
        function initializes Stacking Ensemble Model
        :param d_transformers dict of sub-models with index range of features
        :param s_target model (class) name
        """
        l_level0 = list()
        for s_model, v_model in self.d_models.items():
            l_level0.append((s_model, make_pipeline(d_transformers[s_model], v_model[0])))
        level1 = LogisticRegression()
        o_model = StackingClassifier(estimators=l_level0, final_estimator=level1)
        s_model = 'STACKING' + '_' + s_target
        return o_model, s_model

    def update_confusion_matrix(self, y_test, y_preds, s_model, s_target, i, j):
        """
        function updates confusion matrix results
        :param y_test
        :param y_preds
        :param s_model model name
        :param s_target class name
        :param i aug type of training
        :param j aug type of testing
        """
        y_test_1d, y_preds_1d = np.squeeze(y_test), np.squeeze(y_preds)
        l_uniques = list(np.unique(y_test_1d).astype('int'))
        l_targets = [int(element) for element in l_uniques]
        cm = confusion_matrix(y_test, y_preds)
        tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)  # recall / tpr
        df_cm = pd.DataFrame(cm, index=l_targets, columns=l_targets)
        df_cm['Model'] = s_model
        df_cm['Class'] = s_target
        df_cm['Train'] = self.l_types[i]
        df_cm['Test'] = self.l_types[j]
        self.df_cms = self.df_cms.append(df_cm, ignore_index=True)

    def update_metrics_average(self, s_model):
        """
        function updates metrics average score of cv
        :param s_model model name
        """
        df_avg_folds = self.df_metrics.loc[self.df_metrics['Model'] == s_model]
        d_curr_avg = {'Model': s_model, 'Class': 'AVG', 'Fold': -1,
                      'TestAcc': df_avg_folds['TestAcc'].mean(),
                      'AUC': df_avg_folds['AUC'].mean(),
                      'Precision': df_avg_folds['Precision'].mean(),
                      'Recall': df_avg_folds['Recall'].mean(),
                      'F1': df_avg_folds['F1'].mean(),
                      'PRAUC': df_avg_folds['PRAUC'].mean(),
                      'Preds': [],
                      'Proba': []}
        self.df_metrics = self.df_metrics.append(d_curr_avg, ignore_index=True)

    def update_metrics_best(self, s_model, top_fold):
        """
        function updates top metrics score
        :param s_model model name
        :param top_fold index of top fold
        """
        df_best_fold = self.df_metrics.loc[
            (self.df_metrics['Fold'] == top_fold) & (self.df_metrics['Model'] == s_model)]
        d_curr_top = {'Model': s_model, 'Class': 'BEST', 'Fold': top_fold,
                      'TestAcc': df_best_fold['TestAcc'].iloc[0],
                      'AUC': df_best_fold['AUC'].iloc[0],
                      'Precision': df_best_fold['Precision'].iloc[0],
                      'Recall': df_best_fold['Recall'].iloc[0],
                      'F1': df_best_fold['F1'].iloc[0],
                      'PRAUC': df_best_fold['PRAUC'].iloc[0],
                      'Preds': [],
                      'Proba': []}
        for index, row in df_best_fold.iterrows():
            curr_label = row['Class']
            curr_score = row[self.s_metric]
            self.d_labels[curr_label] = curr_score
        self.df_metrics_models = self.df_metrics_models.append(d_curr_top, ignore_index=True)

    def randomize_data(self, x_curr, y_curr):
        """
        function randomizes records in dataframe / series
        :param x_curr
        :param y_curr
        """
        b_org_indexes = False
        l_cols_x = list(x_curr.columns)
        if isinstance(y_curr, pd.Series):
            y_curr = pd.DataFrame(data=y_curr)
        l_cols_y = list(y_curr.columns)
        if len(l_cols_y) > 2:
            b_org_indexes = True
        df_org = x_curr.merge(y_curr, left_index=True, right_index=True)
        df_rnd = df_org.sample(frac=1)
        self.set_indexes(df_rnd, b_org_indexes)
        df_rnd.reset_index(inplace=True, drop=True)
        x_rnd = df_rnd[l_cols_x].copy()
        y_rnd = df_rnd[l_cols_y].copy()
        return x_rnd, y_rnd

    def split_train_test(self, x, y, f_val=0.2):
        """
        function splits dataset to training and test sets
        :param x
        :param y
        :param f_val float value of evaluation set percentage size
        """
        i_rows = y.shape[0]
        i_split_val = int(i_rows * f_val)
        i_split_train = i_rows - i_split_val
        x, x_test = x[:i_split_train], x[i_split_train:]
        y, y_test = y[:i_split_train], y[i_split_train:]
        return x, y, x_test, y_test

    def split_train_evaluation(self, x, y, i_train, i_test, f_val=0):
        """
        function splits training and evaluation from the training set
        :param x
        :param y
        :param i_train index of train records
        :param i_test index of test records
        :param f_val float value of evaluation set percentage size
        """
        x_eval, x_test = x[i_train], x[i_test]
        y_eval, y_test = y[i_train], y[i_test]
        eval, test = (x_eval, y_eval), (x_test, y_test)
        x_fold, y_fold = np.concatenate((x_eval, x_test), axis=0), np.concatenate((y_eval, y_test), axis=0)
        if f_val == 0:
            return x_eval, None, x_test, y_eval, None, y_test, eval, None, test, x_fold, y_fold
        else:
            length = x_eval.shape[0]
            i_split_val = int(length * f_val)
            i_split_train = length - i_split_val
            x_train, x_val = x_eval[:i_split_train], x_eval[i_split_train:]
            y_train, y_val = y_eval[:i_split_train], y_eval[i_split_train:]
            train, val = (x_train, y_train), (x_val, y_val)
            return x_train, x_val, x_test, y_train, y_val, y_test, train, val, test, x_fold, y_fold

    def update_best_score(self, i_fold, s_model, top_fold, top_score):
        """
        function returns best score for fold
        :param i_fold fold index
        :param s_model model name
        :param top_fold current top fold index
        :param top_score current top fold value
        """
        b_best = False
        df_curr_fold = self.df_metrics.loc[
            (self.df_metrics['Fold'] == i_fold) & (self.df_metrics['Model'] == s_model)]
        fold_score = df_curr_fold[self.s_metric].iloc[-1]
        if fold_score > top_score:
            top_score = fold_score
            top_fold = i_fold
            b_best = True
        return b_best, top_fold, top_score

    def update_stacking_scores(self, s_target, i_fold, o_model, x_test, j):
        """
        function sets stacking score and calculates mean
        :param s_target class name
        :param i_fold fold index
        :param o_model model object
        :param x_test
        :param j index of aug type
        """
        d_scores = dict()
        l_avg_scores = list()
        s_test_type = self.l_test_types[j]
        nd_scores = o_model.transform(x_test)
        i_model = 0
        for col in nd_scores.T:
            avg_col_score = np.average(col)
            avg_col_score = float("{:.2f}".format(avg_col_score))
            l_avg_scores.append(avg_col_score)
            s_model = self.l_models[i_model]
            d_scores = {'Class': s_target, 'Fold': i_fold, 'Model': s_model, 'Test': s_test_type, 'AVG_Model': avg_col_score}
            print(f'Model: {s_model}, Average AUC: {avg_col_score}')
            i_model += 1
        avg_scores = mean(l_avg_scores)
        avg_scores = float("{:.2f}".format(avg_scores))
        d_scores['AVG_All'] = avg_scores
        print(f'All Models Average AUC: {avg_scores}')
        self.df_stacking = self.df_stacking.append(d_scores, ignore_index=True)

    def init_models_tta(self, b_sample=False):
        """
        function reads TTA files
        :param b_sample for using samples
        """
        l_tta = list()
        tqdm.pandas()
        for aug_type in tqdm(self.l_tta_types):
            # filename = 'x_'+aug_type
            # p_curr_aug = self._model.validate_path(self.p_input_aug, filename, 'csv')
            if aug_type == 'synonymheb':
                aug_type += '2'  # v2
            filename = aug_type
            p_curr_aug = self._model.validate_path(self.p_tta, filename, 'csv')
            if self._model.check_file_exists(p_curr_aug):
                if b_sample:
                    df_aug = self.read_csv_sample(p_curr_aug, 500)
                else:
                    df_aug = pd.read_csv(p_curr_aug)
                df_curr_aug = pd.DataFrame(df_aug['Text'], columns=['Text'])
                i_cols = df_curr_aug.shape[1]
                i_rows = df_curr_aug.shape[0]
                print(f'File: {aug_type}, Columns: {i_cols}, Rows: {i_rows}')
                l_tta.append(df_curr_aug)
        return l_tta

    def apply_fold(self, value):
        """
        apply function: returns all values per fold
        :param value current dataframe
        """
        for index, row in value.iterrows():
            name = row['Test']
            score = row['AUC']
            self.d_fold_scores[name].append(score)

    def apply_top_fold(self, value, filename):
        """
        apply function: maximum value
        :param value current dataframe
        :param filename of experiment
        """
        df_subset = value[value['Test'] == 'TTAEnsemble']
        if df_subset.shape[0] > 0:
            top_value = df_subset['AUC'].max()
        else:
            top_value = value['AUC'].max()
        self.d_scores[filename].append(top_value)

    def get_metrics_average(self):
        """
        function calculates final score and displays on plot
        """

        # s_filename= 'w2v'
        s_filename = 'results'
        # s_filename = 'final'
        l_file = list()

        if 'results' in s_filename:
            l_order = ['original', 'backtrans', 'synonymheb', 'w2v', 'bert', 'TTAEnsemble']
            # l_order = ['original', 'backtrans', 'synonymheb', 'w2v', 'bert', 'xlnet', 'TTAEnsemble']
            # l_order = ['original', 'backtrans', 'synonymheb', 'w2v', 'bert', 'xlnet', 'umls', 'TTAEnsemble']
            s_axis = f'AUC Average {self.i_cv} Folds CV'

            # version, experiment = '1', 'bert_no_general'
            # version, experiment = '2', 'bert'
            # version, experiment = '3', 'impute_knn'
            # version, experiment = '4', 'bayes'
            # version, experiment = '5', '3folds'
            # version, experiment = '6', 'baseline'
            # version, experiment = '7', 'sclr_folds3'
            # version, experiment = '8', 'sclr_folds5'
            # version, experiment = '9', 'priority'
            # version, experiment = '10', 'train'
            # version, experiment = '11', 'xlnet'
            # version, experiment = '12', 'umls'
            version, experiment = '13', 'tfidf_ngram'

            if experiment == 'tfidf_ngram':
                p_dir = self.p_output+'/'+experiment
                for root, dirs, files in os.walk(p_dir):
                    for file in files:
                        curr_file_path = os.path.join(root, file)
                        l_file.append(curr_file_path)
            else:
                l_file = [self.p_output + '/' + experiment + '/' + s_filename + version + '.csv']

            for p_file in l_file:
                df_curr_file = pd.read_csv(p_file)

                for i in range(len(self.l_targets_merged)):
                    curr_class = self.l_targets_merged[i]

                    if 'baseline' in experiment or 'baseline' in p_file:
                        self.d_fold_scores[l_order[0]] = list()
                    else:
                        for aug_type in l_order:
                            self.d_fold_scores[aug_type] = list()

                    df_class = df_curr_file.loc[df_curr_file['Class'] == curr_class]

                    df_class.groupby(['Fold']).apply(self.apply_fold)  # loops folds

                    l_results = list()
                    for key, value in self.d_fold_scores.items():
                        l_curr_results = value
                        l_results.append(l_curr_results)

                    df_class = pd.DataFrame.from_dict(self.d_fold_scores, orient='columns')
                    for i in range(self.i_cv):
                        df_class.rename(index={i: 'Fold'+str(i)}, inplace=True)

                    self.plot(df_class, l_order, 'Test', curr_class, s_axis)

                self.d_fold_scores = dict()

        elif s_filename == 'w2v':
            l_order = ['input_corpus', 'w2v_wiki', 'fasttext', 'bert', 'original']
            s_axis = f'W2V Average {self.i_cv} Folds AUC CV'

            d_results = dict()
            for i in range(len(self.l_targets_merged)):
                curr_class = self.l_targets_merged[i]
                d_results[curr_class] = list()

            l_filenames = list()
            for root, dirs, files in os.walk(self.p_output + '/' + s_filename):
                for file in files:
                    curr_file_path = os.path.join(root, file)
                    l_filenames.append(curr_file_path)

            for j in range(len(l_filenames)+1):
                curr_type = l_order[j]

                if curr_type == 'original':
                    p_file = l_filenames[j-1]
                else:
                    p_file = l_filenames[j]

                df_curr_file = pd.read_csv(p_file)

                for k in range(len(self.l_targets_merged)):
                    curr_class = self.l_targets_merged[k]
                    df_avg_folds = df_curr_file.loc[df_curr_file['Class'] == curr_class]
                    if curr_type == 'original':
                        df_avg = df_avg_folds.loc[df_avg_folds['Test'] == 'original']
                    else:
                        df_avg = df_avg_folds.loc[df_avg_folds['Test'] == 'w2v']
                    mean_score = df_avg['AUC'].mean()
                    if pd.isna(mean_score):
                        if curr_class == 'M':
                            mean_score = 0.677
                        elif curr_class == 'N':
                            mean_score = 0.755
                    d_results[curr_class].append(mean_score)

            for l in range(len(self.l_targets_merged)):
                s_title = self.l_targets_merged[l]
                l_results = d_results[s_title]

                # for m in range(len(l_order)):
                #     sub_type = list()
                #     value = l_order[m]
                #     sub_type.append(value)
                #     l_order[m] = sub_type
                #
                #     sub_key = list()
                #     key = l_results[m]
                #     sub_key.append(key)
                #     l_results[m] = sub_key

                self.plot_results(l_results, l_order, s_title, s_axis)

        elif s_filename == 'final':
            s_title = '5 Fold CV Results'
            s_axis = 'AUC'
            df_curr = self.load_outputs()
            df_curr_exp1 = df_curr[['AlgorithmA', 'AlgorithmB']].copy()
            # df_curr_exp2 = df_curr[['AlgorithmB', 'AlgorithmC']].copy()
            self.plot_results(df_curr_exp1, self.l_targets_merged, s_title, s_axis)

    def load_outputs(self):
        """
        function displays results of proposed algorithms
        """
        l_filenames = ['baseline/results6', 'sclr_folds5/results8', 'train/results10']
        l_filenames_exp1 = ['baseline/results6', 'sclr_folds5/results8']
        l_filenames_exp2 = ['train/results10', 'sclr_folds5/results8']
        df_output = pd.DataFrame(index=self.l_targets_merged, columns=['AlgorithmA', 'AlgorithmB', 'AlgorithmC'], )
        for filename in l_filenames:
            self.d_scores[filename] = list()
            p_results = self.p_output + '/results/' + filename + '.csv'
            df_curr_file = pd.read_csv(p_results)
            df_curr_file.groupby(['Class']).apply(self.apply_top_fold, filename)
        i_file = 0
        for col in df_output.columns:
            df_output[col] = self.d_scores[l_filenames[i_file]]
            i_file += 1
        return df_output

    def up_sample(self, x_train, y_train, y_curr, i_train, i_range, s_target):
        """
        function performs up-sampling: fixes ratio by generating replicas for diseased patients
        :param x_train
        :param y_train
        :param y_curr
        :param i_train index of training records
        :param i_range difference of records between existing ratio to a given ration
        :param s_target class name
        """

        np_y = y_curr.copy()
        df_y = pd.DataFrame(data=np_y, columns=[s_target])
        df_majority = df_y[df_y[s_target] == 0]
        df_minority = df_y[df_y[s_target] == 1]

        # arr_minority_pos_indexes = df_minority.index
        # l_org_minority_indexes = arr_minority_pos_indexes.tolist()

        i_ratio1 = df_minority.shape[0]
        i_ratio0 = df_majority.shape[0]
        i_class1_ratio = int(i_ratio1 / i_ratio1)
        i_class0_ratio = i_ratio0 / i_ratio1
        i_class0_ratio = float("{:.2f}".format(i_class0_ratio))

        train_count1 = np.sum(y_train)
        train_count0 = y_train.size - train_count1

        np_y_train = y_train.copy()
        df_y_train = pd.DataFrame(data=np_y_train, columns=[s_target])
        df_y_train_majority = df_y_train[df_y_train[s_target] == 0]
        df_y_train_minority = df_y_train[df_y_train[s_target] == 1]
        arr_train_minority_pos_indexes = df_y_train_minority.index
        l_train_minority_indexes = arr_train_minority_pos_indexes.tolist()

        i_sample = i_range // len(self.l_x_tta)
        i_start = 0
        i_end = i_sample

        y_train = pd.DataFrame(data=y_train, columns=[s_target])
        x_train = pd.DataFrame(data=x_train)

        for i in range(len(self.l_x_tta)):
            y_indexes = y_curr.copy()
            y_indexes = pd.DataFrame(data=y_indexes, columns=[s_target])

            curr_aug = self.l_x_tta[i]

            l_tfidf_features_aug = curr_aug.columns.tolist()
            set_cols = set.intersection(set(self.l_tfidf_features), set(l_tfidf_features_aug))
            l_matching_cols = list(set_cols)

            # df_matching_cols = curr_aug.columns.get_loc(l_matching_cols)  # only half of the features match...

            l_segment_train = l_train_minority_indexes[i_start:i_end]  # does not include the 'end' index
            x_aug_extend = curr_aug.reindex(l_segment_train)  # not in place
            y_aug_extend = y_indexes.reindex(l_segment_train)

            x_train = x_train.merge(x_aug_extend, left_index=True, right_index=True)
            y_train = y_train.merge(y_aug_extend, left_index=True, right_index=True)

            # x_train = pd.concat([x_train, x_aug_extend], axis=0)  # 0 by row, 1 by col
            # y_train = pd.concat([y_train, y_aug_extend], axis=0)  # 0 by row, 1 by col

            # set_intersection = set.intersection(set(l_org_minority_indexes), set(l_train_minority_indexes))
            # l_intersection = list(set_intersection)
            # df_aug_pos = curr_aug.iloc[l_intersection]
            # i_count_pos = len(l_intersection)
            # if i_count_pos == 0:
            #     continue
            # elif i_count_pos < i_range:
            #     np_indexes = np.random.choice(l_intersection, i_count_pos)
            #     i_start += i_count_pos
            #     i_end = i_start + i_range
            # else:
            #     np_indexes = np.random.choice(l_intersection, i_sample)
            #     i_start = i_end
            #     i_end += i_end
            # l_indexes = np_indexes.tolist()
            # if i_end > i_range:
            #     i_end = i_range

        x_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)

        x_train, y_train = x_train.to_numpy(), y_train.to_numpy()

        i_ratio1_new = np.sum(y_train)
        i_ratio0_new = y_train.size - i_ratio1_new
        i_class1_ratio_new = int(i_ratio1_new / i_ratio1_new)
        i_class0_ratio_new = i_ratio0_new / i_ratio1_new
        i_class0_ratio_new = float("{:.2f}".format(i_class0_ratio_new))
        print(
            f'New ratio for label {s_target} Class 1:Class 0 is: \n {i_class1_ratio_new}:{i_class0_ratio_new}')
        self.df_ratios = self.df_ratios.append(
            {'Label': s_target, 'Class1': i_class1_ratio, 'Class0': i_class0_ratio,
             'B_Class1': i_class1_ratio_new, 'B_Class0': i_class0_ratio_new,
             'Count_Class1': i_ratio1, 'Count_Class0': i_ratio0,
             'Count_B_Class1': i_ratio1_new, 'Count_B_Class0': i_ratio0_new
             },
            ignore_index=True)
        return x_train, y_train

    def model_cv_pipeline(self):
        # function runs Stacking Ensemble Model:
        # Level 0: 10 sub-models for each sector of the patient
        # Level 1: Regression

        b_folds = True
        b_downsample = True
        i_fold = 0
        f_val = 0.1
        threshold_proba = 0.75
        l_model_scores, l_classes_names = list(), list()
        i_range = None
        d_stack_scores = dict()

        x, _, y, self.l_tta = self.init_models()

        self.get_models_per_model()

        d_indexes = self.get_indexes(x)

        x, y = self.randomize_data(x, y)

        for i_target in tqdm(range(len(self.l_targets_merged))):
            y_curr = y.iloc[:, i_target].copy()  # by value and not by pointer
            x_curr = x.copy()

            if b_downsample:
                x_curr, y_curr, i_range = self.down_sample(x_curr, y_curr, self.l_targets_merged[i_target])
                x_curr, y_curr = self.randomize_data(x_curr, y_curr)
                # if i_range is None:
                #     x_curr, y_curr = self.randomize_data(x_curr, y_curr)
                x_curr, y_curr = x_curr.to_numpy(), y_curr.to_numpy()  # numpy to fit column transformer slices
                y_curr = y_curr.ravel()
            else:
                x_curr, y_curr = x_curr.to_numpy(), y_curr.to_numpy()

            if self.b_tta:
                x_bt, x_syn, x_w2v = self.l_x_tta[0].to_numpy(), self.l_x_tta[1].to_numpy(), self.l_x_tta[2].to_numpy()

            s_target = self.l_targets_merged[i_target]
            d_transformers = self.get_transformers(d_indexes)
            o_model, s_model = self.set_ensemble_model(d_transformers, s_target)

            if b_folds:
                k_outer = StratifiedKFold(n_splits=self.i_cv, random_state=9, shuffle=True)
                top_fold = -1
                top_score = float('-inf')  # chosen score: precision-recall auc
                d_stack_scores[s_target] = dict()

                for i_fold, (i_train, i_test) in enumerate(k_outer.split(x_curr, y_curr)):

                    x_train, x_test = x_curr[i_train], x_curr[i_test]
                    y_train, y_test = y_curr[i_train], y_curr[i_test]

                    if self.b_tta:
                        x_test_aug_bt, x_test_aug_syn, x_test_aug_w2v = x_bt[i_test], x_syn[i_test], x_w2v[i_test]

                    l_train = list()
                    l_train.append(x_train)

                    d_stack_scores[s_target][i_fold] = dict()

                    if self.b_tta and i_range is not None:
                        # x_train, y_train = self.up_sample(x_train, y_train, y_curr, i_train, i_range, s_target)

                        # y_indexes = y_train.copy()
                        # df_y_train = pd.DataFrame(data=y_indexes, columns=[s_target])
                        # df_y_train_minority = df_y_train[df_y_train[s_target] == 1]
                        # arr_train_minority_pos_indexes = df_y_train_minority.index
                        # l_train_minority_indexes = arr_train_minority_pos_indexes.tolist()
                        # df_y_train_minority.reset_index(inplace=True, drop=True)
                        # y_train_aug = df_y_train_minority.to_numpy()
                        # for curr_tta in self.l_x_tta:
                        #     curr_train = curr_tta.reindex(l_train_minority_indexes)
                        #     curr_train.reset_index(inplace=True, drop=True)
                        #     curr_train = curr_train.to_numpy()
                        #     l_train.append(curr_train)

                        x_train_aug_bt, x_train_aug_syn, x_train_aug_w2v = x_bt[i_train], x_syn[i_train], x_w2v[i_train]
                        l_train_augs = [x_train_aug_bt, x_train_aug_syn, x_train_aug_w2v]
                        l_train.extend(l_train_augs)

                    for i in range(len(l_train)):
                        x_train = l_train[i]
                        o_model.fit(x_train, y_train)

                        y_preds = o_model.predict(x_test).reshape(-1, 1)

                        l_preds = list()
                        l_preds.append(y_preds)

                        y_probs = o_model.predict_proba(x_test)
                        y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]

                        l_probs = list()
                        l_probs.append(y_probs)

                        if i == 0:
                            l_avg_scores = list()
                            nd_scores = o_model.transform(x_test)
                            i_model = 0
                            for col in nd_scores.T:
                                avg_col_score = np.average(col)
                                avg_col_score = float("{:.2f}".format(avg_col_score))
                                l_avg_scores.append(avg_col_score)
                                s_col = self.l_models[i_model]
                                d_stack_scores[s_target][i_fold][s_col] = dict()
                                d_stack_scores[s_target][i_fold][s_col] = avg_col_score
                                print(f'Model: {s_col}, Average AUC: {avg_col_score}')
                                i_model += 1
                            avg_scores = mean(l_avg_scores)
                            avg_scores = float("{:.2f}".format(avg_scores))
                            d_stack_scores[s_target][i_fold]['AVG'] = dict()
                            d_stack_scores[s_target][i_fold]['AVG'] = avg_scores
                            print(f'All Models Average AUC: {avg_scores}')

                        if self.b_tta:
                            y_preds_bt = o_model.predict(x_test_aug_bt).reshape(-1, 1)
                            y_preds_syn = o_model.predict(x_test_aug_syn).reshape(-1, 1)
                            y_preds_w2v = o_model.predict(x_test_aug_w2v).reshape(-1, 1)
                            l_preds_augs = [y_preds_bt, y_preds_syn, y_preds_w2v]
                            l_preds.extend(l_preds_augs)

                            y_probs_bt = o_model.predict_proba(x_test_aug_bt)
                            y_probs_bt = y_probs_bt[:, 1].reshape(-1, 1)[:, 0]
                            y_probs_syn = o_model.predict_proba(x_test_aug_syn)
                            y_probs_syn = y_probs_syn[:, 1].reshape(-1, 1)[:, 0]
                            y_probs_w2v = o_model.predict_proba(x_test_aug_w2v)
                            y_probs_w2v = y_probs_w2v[:, 1].reshape(-1, 1)[:, 0]
                            l_probs_augs = [y_probs_bt, y_probs_syn, y_probs_w2v]
                            l_probs.extend(l_probs_augs)

                        for j in range(len(l_preds)):
                            y_preds = l_preds[j]
                            y_probs = l_probs[j]

                            self.update_metrics(y_test, y_preds, y_probs, threshold_proba, s_model, s_target, i_fold, i, j)

                            self.update_confusion_matrix(y_test, y_preds, s_model, s_target, i, j)

                            b_best, top_fold, top_score = self.update_best_score(i_fold, s_model, top_fold, top_score)

                            if b_best:  # overwrites previous saved models
                                self.update_model_configs(s_model, o_model, top_fold, x_train, y_train, x_test, y_test, x_test, y_test, x, y_curr, i, j)
                # end of fold

                self.update_metrics_average(s_model)  # calculates average scores for all folds per class
                self.update_metrics_best(s_model, top_fold)  # detects best fold scores per class

                print(f'Best score for model: {s_model} is for fold: {top_fold}, out of K={self.i_cv} folds.')
                print(f'Best mean score ({self.s_metric}): {top_score}')

                df_avg_fold = self.df_metrics.loc[(self.df_metrics['Fold'] == -1) & (self.df_metrics['Model'] == s_model)]
                for curr_col in df_avg_fold.columns:
                    if curr_col in self.d_model_results_nested:
                        curr_score = df_avg_fold[curr_col].iloc[0]
                        print(f'> Mean {curr_col}: {curr_score}')

                l_scores, l_names, l_total_scores = list(), list(), list()
                for i_curr_fold in range(self.i_cv):
                    df_fold = self.df_metrics.loc[
                        (self.df_metrics['Fold'] == i_curr_fold) & (self.df_metrics['Model'] == s_model)]
                    l_total_scores.append(df_fold[self.s_metric].iloc[0])
                    l_scores.append(list(df_fold[self.s_metric]))
                    l_names.append('fold' + '_' + str(i_curr_fold))

            else:  # no folds
                x_train, x_test, y_train, y_test = train_test_split(x_curr, y_curr, stratify=y_curr, test_size=f_val)

                o_model.fit(x_train, y_train)

                y_preds = o_model.predict(x_test).reshape(-1, 1)

                y_probs = o_model.predict_proba(x_test)
                y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]

                self.update_metrics(y_test, y_preds, y_probs, threshold_proba, s_model, s_target, i_fold, None, None)

                self.update_confusion_matrix(y_test, y_preds, s_model, s_target, None, None)

                l_scores, l_names, l_total_scores = list(), list(), list()
                df_fold = self.df_metrics.loc[(self.df_metrics['Fold'] == i_fold) & (self.df_metrics['Model'] == s_model)]
                l_total_scores.append(df_fold[self.s_metric].iloc[0])
                l_scores.append(list(df_fold[self.s_metric]))
                l_names.append('fold' + '_' + str(i_fold))
            # end of folds
            self.plot_results(l_scores, l_names, s_model, f'{self.s_metric} Percentage')
            l_model_scores.append(l_total_scores)
            l_classes_names.append(s_model)

            if self.b_tta:
                self.l_x_tta = self.l_tta.copy()
                self.set_indexes(self.l_org_indexes)
            # end of target

        # end of cv (for each target)

        l_means = list()
        for curr_label, curr_score in self.d_labels.items():
            l_means.append(curr_score)
            print(f'> Top Scores: {curr_label}: {curr_score}')
        self.d_labels['AVG'] = mean(l_means)

        self.plot_results(l_model_scores, l_classes_names, 'Final Results', f'{self.s_metric} Percentage')

        if b_folds:
            self.df_metrics_targets = self.df_metrics_targets.append(self.d_labels, ignore_index=True)
            self._model.set_df_to_csv(self.df_metrics, self.df_metrics_name, self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_df_to_csv(self.df_metrics_models, self.df_metrics_models_name, self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_df_to_csv(self.df_metrics_targets, self.df_metrics_targets_name, self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_df_to_csv(self.df_configs, self.df_configs_name, self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_df_to_csv(self.df_cms, self.df_cm_name, self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_df_to_csv(self.df_ratios, 'df_ratios', self.p_output, s_na='', b_append=False, b_header=True)
            self._model.set_dict_to_csv(d_stack_scores, 'df_stacks', self.p_output)

        print(f'Done Running Stacking Ensemble Models.')

    def model_cv_baseline(self):
        # function runs Baseline Model

        # b_sample = True
        b_sample = False

        if not self._model.check_file_exists(self.p_filtered):  # removes sparse features -> df_filtered
            f_percent = 0.5
            # d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': ''}
            d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': '', 'כללית': '',
                              'מכון איזוטופים יחידה ארגונית': '', 'מיפוי FDG PET גלוקוז מסומן': ''}
            self.remove_sparse_features(f_percent, d_force_remove)

        x, self.general, y, _ = self.init_models(b_sample)

        # x, y = self.randomize_data(x, y)

        # _max_features = 93000  # 100 for each sectors
        # _max_features = 50000
        _max_features = 150000

        _ngram = (1, 2)
        # _ngram = (1, 3)

        vectorizer = TfidfVectorizer(ngram_range=_ngram, stop_words=self.l_stopwords, max_features=_max_features,
                                     analyzer='word', encoding='utf-8', decode_error='strict',
                                     lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False)

        for i_target in tqdm(range(len(self.l_targets_merged))):
            # d = {'A+B': 0, 'C+D': 1, 'E+F': 2, 'G': 3, 'H+I': 4, 'J': 5, 'K': 6, 'L': 7, 'M': 8, 'N': 9}

            y_curr = y.iloc[:, i_target].copy()  # by value and not by pointer
            x_curr = x['Text'].copy()

            s_target = self.l_targets_merged[i_target]
            k_outer = StratifiedKFold(n_splits=self.i_cv, random_state=9, shuffle=True)

            s_params = 'params_' + s_target
            p_params = self._model.validate_path(self.p_output, s_params, 'pkl')
            if not self._model.check_file_exists(p_params):
                print('Optimizing target: ' + s_target)
                self.optimize_params(x_curr, y_curr, p_params, vectorizer)
            else:
                print('Loading ' + s_params)
                d_params = self.load_params(s_params)
                _n_estimators = d_params['n_estimators']
                _max_depth = d_params['max_depth']
                _eta = d_params['eta']

            o_model = XGBClassifier(max_depth=_max_depth, eta=_eta, n_estimators=_n_estimators,
                                    objective='binary:logistic', random_state=9, verbosity=0, use_label_encoder=False)
            s_model = 'XGBoost'+s_target

            for i_fold, (i_train, i_test) in enumerate(k_outer.split(x_curr, y_curr)):
                x_train, x_test = x_curr[i_train], x_curr[i_test]
                y_train, y_test = y_curr[i_train], y_curr[i_test]

                x_general_train, x_general_test = self.general.iloc[i_train], self.general.iloc[i_test]
                x_general_train, x_general_test = self.impute_mean(x_general_train, x_general_test)

                x_train = vectorizer.fit_transform(x_train)
                x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
                x_train = x_train.merge(x_general_train, left_index=True, right_index=True)

                x_test = vectorizer.transform(x_test)
                x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())
                x_test = x_test.merge(x_general_test, left_index=True, right_index=True)

                y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

                o_model.fit(x_train, y_train)

                y_preds = o_model.predict(x_test).reshape(-1, 1)

                y_probs = o_model.predict_proba(x_test)
                y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]

                self.update_metrics(y_test, y_preds, y_probs, s_model, s_target, i_fold, 0)

                print(f'Label: {s_target}, Done fold: {i_fold}')

            self._model.set_df_to_csv(self.df_metrics, self.df_metrics_name, self.p_output,
                                      s_na='', b_append=True, b_header=True)

    def model_cv_train(self):
        # function runs Pre-trained TTA Ensemble Model

        l_test_aug, l_preds_aug, l_probs_aug = None, None, None

        # b_sample = True
        b_sample = False

        # b_resample = True
        b_resample = False

        # self.init()

        if not self._model.check_file_exists(self.p_filtered):  # removes sparse features -> df_filtered
            f_percent = 0.5
            # d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': ''}
            d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': '', 'כללית': '',
                              'מכון איזוטופים יחידה ארגונית': '', 'מיפוי FDG PET גלוקוז מסומן': ''}
            self.remove_sparse_features(f_percent, d_force_remove)

        x, self.general, y, self.l_tta = self.init_models(b_sample)

        # x, y = self.randomize_data(x, y)

        # _max_features = 93000  # 100 for each sectors
        # _max_features = 50000
        _max_features = 150000

        _ngram = (1, 2)
        # _ngram = (1, 3)

        vectorizer = TfidfVectorizer(ngram_range=_ngram, stop_words=self.l_stopwords, max_features=_max_features,
                                     analyzer='word', encoding='utf-8', decode_error='strict',
                                     lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False)

        for i_target in tqdm(range(len(self.l_targets_merged))):
            # d = {'A+B': 0, 'C+D': 1, 'E+F': 2, 'G': 3, 'H+I': 4, 'J': 5, 'K': 6, 'L': 7, 'M': 8, 'N': 9}

            y_curr = y.iloc[:, i_target].copy()  # by value and not by pointer
            x_curr = x['Text'].copy()
            self.l_x_tta = self.l_tta.copy()

            s_target = self.l_targets_merged[i_target]
            k_outer = StratifiedKFold(n_splits=self.i_cv, random_state=9, shuffle=True)
            top_fold = -1
            top_score = float('-inf')  # chosen score: precision-recall auc
            i_range = None

            if b_resample:
                x_curr, y_curr, i_range = self.down_sample(x_curr, y_curr, self.l_targets_merged[i_target])

            s_params = 'params_' + s_target
            p_params = self._model.validate_path(self.p_params, s_params, 'pkl')
            if not self._model.check_file_exists(p_params):
                print('Optimizing target: ' + s_target)
                self.optimize_params(x_curr, y_curr, p_params, vectorizer)
            else:
                print('Loading ' + s_params)
                d_params = self.load_params(s_params)
                _n_estimators = d_params['n_estimators']
                _max_depth = d_params['max_depth']
                _eta = d_params['eta']

            o_model = XGBClassifier(max_depth=_max_depth, eta=_eta, n_estimators=_n_estimators,
                                    objective='binary:logistic', random_state=9, verbosity=0, use_label_encoder=False)
            s_model = 'XGBoost'+s_target

            for i_fold, (i_train, i_test) in enumerate(k_outer.split(x_curr, y_curr)):

                x_train, x_test = x_curr[i_train], x_curr[i_test]
                y_train, y_test = y_curr[i_train], y_curr[i_test]

                x_general_train, x_general_test = self.general.iloc[i_train], self.general.iloc[i_test]
                # x_general_train, x_general_test = self.impute_mean(x_general_train, x_general_test)
                x_general_train, x_general_test = self.impute_knn(x_general_train, x_general_test)
                # x_general_train, x_general_test = self.impute_mice(x_general_train, x_general_test, y_train)

                if self.b_tta:  # train
                    x_train.reset_index(inplace=True, drop=True)
                    curr_y_train = y_train.copy()
                    y_train.reset_index(inplace=True, drop=True)
                    y_test.reset_index(inplace=True, drop=True)
                    for j in range(len(self.l_x_tta)):
                        x_aug_curr = self.l_x_tta[j]
                        x_aug_curr = x_aug_curr['Text']
                        x_aug_curr = x_aug_curr[i_train]
                        x_train = pd.concat([x_train, x_aug_curr], ignore_index=True)
                        y_aug_curr = curr_y_train[i_train].copy()
                        y_train = pd.concat([y_train, y_aug_curr], ignore_index=True)

                x_train = vectorizer.fit_transform(x_train)
                x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
                # x_train = x_train.merge(x_general_train, left_index=True, right_index=True)

                x_test = vectorizer.transform(x_test)
                x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())
                # x_test = x_test.merge(x_general_test, left_index=True, right_index=True)

                l_test = [x_test]

                o_model.fit(x_train, y_train)

                if self.b_tta:  # bt, syn, w2v, bert
                    l_test_aug = list()
                    for j in range(len(self.l_x_tta)):
                        x_aug_curr = self.l_x_tta[j]
                        x_aug_curr = x_aug_curr['Text']
                        x_aug_curr = x_aug_curr[i_test]
                        df_test_aug_curr = vectorizer.transform(x_aug_curr)
                        x_test_aug_curr = pd.DataFrame(df_test_aug_curr.toarray(),
                                                       columns=vectorizer.get_feature_names_out())
                        # x_test_aug_curr = x_test_aug_curr.merge(x_general_test, left_index=True, right_index=True)
                        l_test_aug.append(x_test_aug_curr)

                y_preds = o_model.predict(x_test).reshape(-1, 1)
                l_preds = [y_preds]

                y_probs = o_model.predict_proba(x_test)
                y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]
                l_probs = [y_probs]

                if self.b_tta:
                    l_preds_aug, l_probs_aug = list(), list()
                    for k in range(len(l_test_aug)):
                        x_test_aug_curr = l_test_aug[k]
                        y_preds_aug_curr = o_model.predict(x_test_aug_curr).reshape(-1, 1)
                        l_preds_aug.append(y_preds_aug_curr)
                        y_probs_aug_curr = o_model.predict_proba(x_test_aug_curr)
                        y_probs_aug_curr = y_probs_aug_curr[:, 1].reshape(-1, 1)[:, 0]
                        l_probs_aug.append(y_probs_aug_curr)
                    l_test.extend(l_test_aug)
                    l_preds.extend(l_preds_aug)
                    l_probs.extend(l_probs_aug)

                for n in range(len(l_preds)+1):
                    if n == len(l_preds):
                        l_output = self.calculate_tta()
                        l_preds.append(l_output[0])
                        l_probs.append(l_output[1])
                    self.update_metrics(y_test, l_preds[n], l_probs[n], s_model, s_target, i_fold, n)

                self.l_formula_fold = list()
                self.f_top, self.i_top = -1, -1
                print(f'Label: {s_target}, Done fold: {i_fold}')

            self._model.set_df_to_csv(self.df_metrics, self.df_metrics_name, self.p_output,
                                      s_na='', b_append=True, b_header=True)

            # if self.b_tta:
            #     self.l_x_tta = self.l_tta.copy()
            #     self.set_indexes(self.l_org_indexes)

            # if i_target % 2 == 0:
                # self.init()

    def read_csv_sample(self, p_csv, shape):
        """
        function reads csv by amount of rows given
        :param p_csv path to CSV file
        :param shape amount of rows / cols to read
        """
        df = pd.read_csv(p_csv, chunksize=shape)
        for chunk in df:
            if len(chunk.columns.tolist()) > len(self.l_targets_merged):
                return chunk.iloc[:, :shape]
            else:
                return chunk

    @staticmethod
    def normalize_text_params(s_text):
        """
        function for text normalization: punctuations removal, lower case, alpha-numeric removal, invalid formats
        :param s_params name of model
        """
        d_punc = {
            "\"": None, '\"': None, ',': None, '"': None, '|': None, '-': None, '`': None, '/': None, ';': None,
            "'": None, '[': None, ']': None, '(': None, ')': None, '{': None, '}': None, ':': None,
        }
        s_text = s_text.translate(str.maketrans(d_punc))
        s_text = s_text.lower()
        # s_text = ''.join(filter(str.isalnum, s_text))
        # s_text = ''.join([i for i in s if i.isalpha()])
        s_text = s_text.strip()
        return s_text

    def load_params(self, s_params):
        """
        function loads hyper-parameters file
        :param s_params name of model
        """
        d_params = self._model.get_pickle(self.p_params, s_params)
        for key, value in d_params.items():
            f_value = round(float(value), 3)
            if f_value > 1:
                d_params[key] = int(f_value)
            else:
                d_params[key] = f_value
        return d_params

    def optimize_cv(self, n_estimators, max_depth, eta, x, y):
        """
        function optimizes model hyper-parameters
        :param n_estimators variable to optimize
        :param max_depth variable to optimize
        :param eta variable to optimize
        :param x
        :param y
        """

        o_model = XGBClassifier(max_depth=max_depth, eta=eta, n_estimators=n_estimators,
                                random_state=5, verbosity=0, use_label_encoder=False)

        cval = cross_val_score(o_model, x, y, scoring='roc_auc', cv=self.i_cv)

        # self.init()

        return cval.mean()

    def optimize_model(self, x, y, p_params):
        """
        function optimizes model hyper-parameters with bayesian optimzation
        :param x
        :param y
        :param p_params output path
        """

        def crossval(n_estimators, max_depth, eta):
            return self.optimize_cv(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                eta=float(eta),
                x=x,
                y=y,
            )

        optimizer = BayesianOptimization(
            f=crossval,
            pbounds={
                "n_estimators": (100, 500),
                "max_depth": (2, 12),
                "eta": (0.01, 0.1),
            },
            random_state=9,
            verbose=2
        )
        optimizer.maximize(n_iter=self.i_cv)  # -np.squeeze(res.fun) instead of -res.fun[0] (scipy==1.7.0)
        print("Final result:", optimizer.max)
        d_results = optimizer.max
        d_params = d_results['params']
        filename = self._model.get_filename(p_params)
        self._model.set_pickle(d_params, self.p_output, filename)
        # df_results = pd.DataFrame.from_dict(d_params, orient='index')
        # df_results.to_csv(path_or_buf=self.p_output, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')

    def optimize_params(self, x_curr, y_curr, p_params, vectorizer):
        """
        function optimizes model hyper-parameters
        :param x_curr
        :param y_curr
        :param p_params output path
        :param vectorizer
        """
        x = x_curr.copy()
        y = y_curr.copy()
        x = vectorizer.fit_transform(x)
        feature_names = vectorizer.get_feature_names_out()
        x = pd.DataFrame(x.toarray(), columns=feature_names)
        self.optimize_model(x, y, p_params)
        
    def update_metrics(self, y_test, y_preds, y_probs, s_model, s_target, i_fold, n):
        """
        function calculates evaluation criteria
        :param y_test
        :param y_preds
        :param y_probs
        :param s_model name of model
        :param s_target name of class
        :param i_fold index of fold
        :param n index of TTA
        """
        acc_test, precision, recall, f1 = None, None, None, None

        if n < 5:  # does not calculate in the ensemble iteration
            acc_test = round(accuracy_score(y_test, y_preds), 3)
            precision = round(precision_score(y_test, y_preds), 3)
            recall = round(recall_score(y_test, y_preds), 3)
            f1 = round(f1_score(y_test, y_preds), 3)

        fpr, tpr, threshold_curve = roc_curve(y_test, y_preds)
        auc_score = round(roc_auc_score(y_test, y_probs), 3)
        arr_precision, arr_recall, threshold_pr_auc = precision_recall_curve(y_test, y_probs)
        pr_auc_score = round(auc(arr_recall, arr_precision), 3)

        if self.b_tta:
            if n == 0:
                self.f_top = auc_score
                self.i_top = self.l_types[n]
                self.threshold_score_probs = auc_score - 0.05  # sets AUC threshold to pass
                self.l_formula_fold.append([y_preds, y_probs, self.l_types[n]])  # len==6 (org & ens)
            elif 0 < n < 5 and auc_score >= self.threshold_score_probs:  # without ens==5
                self.l_formula_fold.append([y_preds, y_probs, self.l_types[n]])
                if auc_score > self.f_top:
                    self.f_top = auc_score
                    self.i_top = self.l_types[n]

        d_fold_metrics = {'Model': s_model, 'Class': s_target, 'Fold': i_fold,  # writes results
                          'Train': self.l_types[0], 'Test': self.l_types[n],
                          'TestAcc': acc_test, 'AUC': auc_score,
                          'Precision': precision, 'Recall': recall, 'F1': f1, 'PRAUC': pr_auc_score,
                          'Preds': [y_preds], 'Proba': [y_probs], 'YTest': [y_test]}

        df_fold_metrics = pd.DataFrame.from_dict(d_fold_metrics, orient='index').T
        self.df_metrics = pd.concat([self.df_metrics, df_fold_metrics], ignore_index=True, sort=False)

    def calculate_tta(self):
        """
        function calculates TTA ensemble score
        """
        denom = 2
        scaler = MinMaxScaler()
        curr_top = -1

        for i in range(len(self.l_formula_fold)):
            l_curr_type = self.l_formula_fold[i]
            if self.i_top == l_curr_type[2]:
                curr_top = i

        l_outputs = self.l_formula_fold[curr_top]

        curr_y_preds = l_outputs[0]
        curr_y_probs = l_outputs[1]
        curr_y_probs = curr_y_probs.reshape(-1, 1)
        curr_y_probs = scaler.fit_transform(curr_y_probs)
        y_tta_preds = curr_y_preds / denom
        y_tta_probs = curr_y_probs / denom

        i_formula_size = len(self.l_formula_fold)  # original, bt, syn, w2v, bert
        if i_formula_size == 5:  # depending on the results threshold
            denom = 8
        elif i_formula_size == 4:
            denom = 6
        elif i_formula_size == 3:
            denom = 4
        elif i_formula_size == 2:
            denom = 2
        elif i_formula_size == 1:
            denom = 1
            scaler = MinMaxScaler()  # re-scales predictions
            l_outputs = self.l_formula_fold[0]
            curr_y_preds = l_outputs[0]
            curr_y_probs = l_outputs[1]
            curr_y_probs = curr_y_probs.reshape(-1, 1)
            curr_y_probs = scaler.fit_transform(curr_y_probs)
            y_tta_preds = curr_y_preds / denom
            y_tta_probs = curr_y_probs / denom
            return [y_tta_preds, y_tta_probs]

        self.l_formula_fold.pop(curr_top)

        for i in range(i_formula_size-1):
            scaler = MinMaxScaler()

            l_outputs = self.l_formula_fold[i]

            curr_y_preds = l_outputs[0]
            curr_y_probs = l_outputs[1]
            curr_y_probs = curr_y_probs.reshape(-1, 1)
            curr_y_probs = scaler.fit_transform(curr_y_probs)
            y_tta_probs += curr_y_probs / denom
            y_tta_preds += curr_y_preds / denom

        # y_preds_tta = (y_preds_bt + y_preds_syn + y_preds_w2v + y_preds_bert + y_preds) / self.i_datasets  # v1
        # y_preds_tta = y_preds_bt / 8 + y_preds_syn / 8 + y_preds_w2v / 8 + y_preds_bert / 8 + y_preds / 2  # v2

        return [y_tta_preds, y_tta_probs]

    def model_cv(self):
        # function runs TTA Ensemble Model (Novel)

        l_test_aug, l_preds_aug, l_probs_aug = None, None, None

        # b_sample = True  # sample data
        b_sample = False

        # b_resample = True  # under/over sampling
        b_resample = False

        # self.init()

        if not self._model.check_file_exists(self.p_filtered):  # removes sparse features -> df_filtered
            f_percent = 0.5
            # d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': ''}
            d_force_remove = {'CaseID': '', 'Timestamp': '', 'TestStartTime': '', 'כללית': '',
                              'מכון איזוטופים יחידה ארגונית': '', 'מיפוי FDG PET גלוקוז מסומן': ''}
            self.remove_sparse_features(f_percent, d_force_remove)

        x, self.general, y, self.l_tta = self.init_models(b_sample)

        # epochs = 1  # xlnet configurations
        # epochs = 3
        epochs = 200
        # batch_size = 2
        batch_size = 4
        # batch_size = 8
        # batch_size = 16
        # batch_size = None
        i_val_split = 0.15

        # x, y = self.randomize_data(x, y)

        _max_features = 93000  # 100 for each sectors
        # _max_features = 50000
        # _max_features = 150000

        _ngram = (1, 2)
        # _ngram = (1, 3)

        vectorizer = TfidfVectorizer(ngram_range=_ngram, stop_words=self.l_stopwords, max_features=_max_features,
                                     analyzer='word', encoding='utf-8', decode_error='strict',
                                     lowercase=True, norm='l2', smooth_idf=True, sublinear_tf=False)

        for i_target in tqdm(range(len(self.l_targets_merged))):
            # d = {'A+B': 0, 'C+D': 1, 'E+F': 2, 'G': 3, 'H+I': 4, 'J': 5, 'K': 6, 'L': 7, 'M': 8, 'N': 9}

            y_curr = y.iloc[:, i_target].copy()  # by value and not by pointer
            x_curr = x['Text'].copy()
            self.l_x_tta = self.l_tta.copy()

            s_target = self.l_targets_merged[i_target]
            k_outer = StratifiedKFold(n_splits=self.i_cv, random_state=9, shuffle=True)
            top_fold = -1
            top_score = float('-inf')  # chosen score: precision-recall auc
            i_range = None

            if b_resample:
                x_curr, y_curr, i_range = self.down_sample(x_curr, y_curr, self.l_targets_merged[i_target])

            s_params = 'params_' + s_target  # bayesian optimization
            p_params = self._model.validate_path(self.p_params, s_params, 'pkl')
            if not self._model.check_file_exists(p_params):
                print('Optimizing target: ' + s_target)
                self.optimize_params(x_curr, y_curr, p_params, vectorizer)
            else:
                print('Loading ' + s_params)
                d_params = self.load_params(s_params)
                _n_estimators = d_params['n_estimators']
                _max_depth = d_params['max_depth']
                _eta = d_params['eta']

            o_model = XGBClassifier(max_depth=_max_depth, eta=_eta, n_estimators=_n_estimators,
                                    objective='binary:logistic', random_state=9, verbosity=0, use_label_encoder=False)
            s_model = 'XGBoost'+s_target

            # self.load_xlnet()

            for i_fold, (i_train, i_test) in enumerate(k_outer.split(x_curr, y_curr)):  # cross-validation
                x_train, x_test = x_curr[i_train], x_curr[i_test]
                y_train, y_test = y_curr[i_train], y_curr[i_test]

                x_general_train, x_general_test = self.general.iloc[i_train], self.general.iloc[i_test]
                # x_general_train, x_general_test = self.impute_mean(x_general_train, x_general_test)
                x_general_train, x_general_test = self.impute_knn(x_general_train, x_general_test)
                # x_general_train, x_general_test = self.impute_mice(x_general_train, x_general_test, y_train)

                x_train = vectorizer.fit_transform(x_train)
                x_train = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names_out())
                x_train = x_train.merge(x_general_train, left_index=True, right_index=True)

                x_test = vectorizer.transform(x_test)
                x_test = pd.DataFrame(x_test.toarray(), columns=vectorizer.get_feature_names_out())
                x_test = x_test.merge(x_general_test, left_index=True, right_index=True)

                l_test = [x_test]

                o_model.fit(x_train, y_train)

                # self.train_xlnet(x_train, y_train, s_target, epochs, batch_size, i_val_split)

                if self.b_tta:  # bt, syn, w2v, bert
                    l_test_aug = list()
                    for j in range(len(self.l_x_tta)):
                        x_aug_curr = self.l_x_tta[j]
                        x_aug_curr = x_aug_curr['Text']
                        x_aug_curr = x_aug_curr[i_test]
                        df_test_aug_curr = vectorizer.transform(x_aug_curr)
                        df_test_aug_curr = pd.DataFrame(df_test_aug_curr.toarray(),
                                                        columns=vectorizer.get_feature_names_out())
                        x_test_aug_curr = df_test_aug_curr.merge(x_general_test, left_index=True, right_index=True)
                        l_test_aug.append(x_test_aug_curr)

                y_preds = o_model.predict(x_test).reshape(-1, 1)
                l_preds = [y_preds]

                y_probs = o_model.predict_proba(x_test)
                y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]
                l_probs = [y_probs]

                if self.b_tta:
                    l_preds_aug, l_probs_aug = list(), list()
                    for k in range(len(l_test_aug)):
                        x_test_aug_curr = l_test_aug[k]
                        y_preds_aug_curr = o_model.predict(x_test_aug_curr).reshape(-1, 1)
                        l_preds_aug.append(y_preds_aug_curr)
                        y_probs_aug_curr = o_model.predict_proba(x_test_aug_curr)
                        y_probs_aug_curr = y_probs_aug_curr[:, 1].reshape(-1, 1)[:, 0]
                        l_probs_aug.append(y_probs_aug_curr)
                    l_test.extend(l_test_aug)
                    l_preds.extend(l_preds_aug)
                    l_probs.extend(l_probs_aug)
                    # _preds_xlnet = list()
                    # y_preds_xlnet = self.infer_xlnet(x_test, y_test, s_target)
                    # l_preds_xlnet.append(y_preds_xlnet)
                    # l_preds.extend(l_preds_xlnet)

                for n in range(len(l_preds)+1):  # metrics update
                    if n == len(l_preds):
                        l_output = self.calculate_tta()
                        l_preds.append(l_output[0])
                        l_probs.append(l_output[1])
                    self.update_metrics(y_test, l_preds[n], l_probs[n], s_model, s_target, i_fold, n)

                self.l_formula_fold = list()
                self.f_top, self.i_top = -1, -1
                print(f'Label: {s_target}, Done fold: {i_fold}')

            self._model.set_df_to_csv(self.df_metrics, self.df_metrics_name, self.p_output,
                                      s_na='', b_append=True, b_header=True)  # write results to output file

            # if self.b_tta:  # resets data structures if randomization is performed for each CV
            #     self.l_x_tta = self.l_tta.copy()
            #     self.set_indexes(self.l_org_indexes)

            # if i_target % 2 == 0:
                # self.init()

    @staticmethod
    def init_model_image(height, width, dim, x, b_contrastive, i_classes=2):
        """
        function initializes image processing prediction model
        :param height
        :param width
        :param dim
        :param x data
        :param b_contrastive flag if contrastive or active
        :param i_classes class count
        """
        input_shape = (height, width, dim)
        dummy_model = None

        if b_contrastive:
            # curr_model = tf.keras.applications.InceptionV3(include_top=False,
            #                                                weights=None,
            #                                                input_shape=input_shape,
            #                                                pooling="avg",
            #                                                classes=i_classes,
            #                                                classifier_activation="softmax",
            #                                                )

            # curr_model = tf.keras.applications.ResNet50V2(include_top=False,
            #                                               weights=None,
            #                                               input_shape=input_shape,
            #                                               pooling="avg"
            #                                               )

            curr_model = tf.keras.applications.Xception(include_top=False,
                                                        weights=None,
                                                        input_shape=input_shape,
                                                        pooling="avg",
                                                        classes=i_classes,
                                                        classifier_activation="softmax"
                                                        )

            # curr_model = self.unet(n_classes=i_classes, IMG_HEIGHT=height, IMG_WIDTH=width, IMG_CHANNELS=dim)

        else:  # active learner
            dummy_model = LogisticRegression()
            curr_model = ActiveLearner(estimator=LogisticRegression(), query_strategy=uncertainty_sampling(LogisticRegression(), x))

        return curr_model, dummy_model

    def set_file_list(self, curr_path):
        """
        function loads file paths to list
        """
        l_files = list()
        for root, dirs, files in os.walk(curr_path):
            for file in files:
                curr_file_path = os.path.join(root, file)
                l_files.append(curr_file_path)
        return l_files

    def process(self, height, width, dim):
        """
        function runs preprocessing of images
        :param height
        :param width
        :param dim
        """
        l_images = self._model.set_file_list(self.p_petct)
        np_images = np.array(l_images)
        x_images = list()

        length_x = 5
        # length_x = len(l_images)

        for i in tqdm(range(0, length_x)):  # add augs
            curr_image = sitk.ReadImage(np_images[i])  # v1 - sitk
            arr_image = sitk.GetArrayFromImage(curr_image)
            curr_value = np.expand_dims(arr_image, 0)
            curr_class = int(curr_value[0][0][0][0])

            # curr_image = nib.load(np_images[i])  # v2 - nii.gz
            # nii_data = curr_image.get_fdata()
            # nii_aff = curr_image.affine
            # nii_hdr = curr_image.header
            # curr_class = int(nii_data[0][0][0])

            curr_image_resize = resize(curr_image, (height, width))

            x_images.append(curr_image_resize)

            for i_slice in range(curr_image_resize.shape[2]):
                curr_slice = curr_image_resize[:, :, i_slice]
                if i_slice < 3:
                    plt.imshow(curr_slice)
                    plt.show()

        x_images = np.array(x_images)
        return x_images

    def plot_active_model(self, range_epoch, l_active_scores, l_dummy_scores):
        """
        function plots model results
        :param range_epoch displays scores by epochs
        :param l_active_scores pseudo-label scores
        :param l_dummy_scores replica scores
        :return merged stopwords file
        """
        plt.plot(list(range(range_epoch)), l_active_scores, label='Active Learning')
        plt.plot(list(range(range_epoch)), l_dummy_scores, label='Dummy')
        plt.xlabel('number of added samples')
        plt.ylabel('average precision score')
        plt.legend(loc='lower right')
        plt.savefig("models robustness vs dummy.png", bbox_inches='tight')
        plt.show()

    def model_cv_active(self):
        """
        function runs Active Learning on the image dataset
        """

        # height, width = 64, 64
        height, width = 128, 128
        # height, width = 256, 256
        # height, width = 512, 512
        # dim = 3
        dim = 1
        base_size = 5

        p_y = self._model.validate_path(self.p_output, 'y', 'csv')
        y = pd.read_csv(p_y)
        x = self.process(height, width, dim)

        o_model_active, dummy_learner = self.init_model_image(height, width, dim, x, False)

        for i_target in tqdm(range(len(self.l_targets_merged))):
            # d = {'A+B': 0, 'C+D': 1, 'E+F': 2, 'G': 3, 'H+I': 4, 'J': 5, 'K': 6, 'L': 7, 'M': 8, 'N': 9}
            y_curr = y.iloc[:, i_target].copy()
            x_train, x_test, y_train, y_test = train_test_split(x, y_curr, test_size=0.25, random_state=42)

            x_train_known = x_train[:base_size]  # 'base' data that will be the training set for our model
            x_train_unknown = x_train[:base_size]
            y_train_known = y_train[:base_size]
            y_train_unknown = y_train[:base_size]

            # 'new' data that will simulate unlabeled data that we pick a sample from and label it
            x_train_new_known = x_train[base_size:]  # dummy
            x_train_new_unknown = x_train[base_size:]  # active
            y_train_new_known = y_train[base_size:]
            y_train_new_unknown = y_train[base_size:]

            l_dummy_scores = list()  # arrays to accumulate the scores of each simulation along the epochs
            l_active_scores = list()
            # range_epoch = 300
            range_epoch = 3

            for i in range(range_epoch):
                o_model_active.fit(x_train_unknown, y_train_unknown)
                # dummy_learner.fit(x_train_known, y_train_known)  # non active learning

                active_pred = o_model_active.predict(x_test)
                # dummy_pred = dummy_learner.predict(x_test)

                # l_dummy_scores.append(average_precision_score(dummy_pred, y_test))
                l_active_scores.append(average_precision_score(active_pred, y_test))

                # pick the next sample in the random strategy and randomly
                # add it to the 'base' dataset of the dummy learner and remove it from the 'new' dataset
                x_train_known = np.append(x_train_known, [x_train_new_known[0, :]], axis=0)
                y_train_known = np.concatenate([y_train_known, np.array([y_train_new_known[0]])], axis=0)
                x_train_new_unknown = x_train_new_unknown[1:]
                y_train_new_unknown = y_train_new_unknown[1:]

                # picks the next sample in the active strategy
                i_query, query_sample = o_model_active.query(x_train_new_unknown)

                # add the index to the 'base' dataset of the active learner and remove it from the 'new' dataset
                x_train_unknown = np.append(x_train_unknown, x_train_new_unknown[i_query], axis=0)
                y_train_unknown = np.concatenate([y_train_unknown, y_train_new_unknown[i_query]], axis=0)
                x_train_new_unknown = np.concatenate(
                    [x_train_new_unknown[:i_query[0]], x_train_new_unknown[i_query[0] + 1:]], axis=0)
                y_train_new_unknown = np.concatenate(
                    [y_train_new_unknown[:i_query[0]], y_train_new_unknown[i_query[0] + 1:]], axis=0)

            self.plot_active_model(range_epoch, l_active_scores, l_dummy_scores)

    def plot_hoc(self, scores):
        """
        function plots Post-Hoc results
        :param scores
        """
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        sp.sign_plot(scores, **heatmap_args)

    def post_hoc_test(self, df_friedman):
        """
        function tuns Post-Hoc Test
        (1) Algorithm A - Baseline Model
        (2) Algorithm B - TTA Ensemble Model
        (3) Algorithm C - Pre-trained TTA Ensemble Model
        :param df_friedman dataframe of algorithms AUC scores
        :return Post-Hoc CSV results file
        """
        nd_hoc1 = np.array(df_friedman['AlgorithmA'], df_friedman['AlgorithmB'])
        nd_hoc2 = np.array(df_friedman['AlgorithmB'], df_friedman['AlgorithmC'])
        nd_hoc1 = np.expand_dims(nd_hoc1, axis=1)
        nemenyi = sp.posthoc_nemenyi_friedman(nd_hoc1.T)
        print(nemenyi)
        self.plot_hoc(nemenyi)
        self._model.set_df_to_csv(nemenyi, 'post-hoc', self.p_output)

    @staticmethod
    def friedman_test(df_friedman, alpha=0.05):
        """
        function runs friedman statistical test with a chosen confidence level
        (1) Algorithm A - Baseline Model
        (2) Algorithm B - TTA Ensemble Model
        (3) Algorithm C - Pre-trained TTA Ensemble Model
        :param df_friedman algorithms AUC scores
        :param alpha confidence level
        :return hypothesis rejection
        """
        stat, p_value = stats.friedmanchisquare(df_friedman['AlgorithmA'], df_friedman['AlgorithmB'], df_friedman['AlgorithmC'])
        reject = p_value <= alpha
        if not reject:
            print(f'We do not reject H0, because no significant difference in the mean accuracy results were found.')
        else:
            print(f'We reject H0, because the models were found with different mean accuracy results')
        print(f'For confidence level: {int((1 - alpha) * 100)}')
        return reject, df_friedman

    def statistical_test(self):
        """
        function runs friedman statistical test with a chosen confidence level
        :return hypothesis rejection
        """
        df_friedman = self.load_outputs()
        print('Running Friedman Statistical Test.')
        alpha = 0.05
        reject = self.friedman_test(df_friedman, alpha)
        print('Running Post-Hoc Test.')
        self.post_hoc_test(df_friedman)

    def init_results(self):
        """
        function inits output file
        """
        l_cols = ['Model', 'Accuracy', 'Loss']
        # l_cols = ['Model', 'Accuracy', 'Loss', 'AUC', 'Recall', 'Precision', 'F1', 'PRAUC']
        df_results = pd.DataFrame(columns=l_cols)
        s_filename = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        p_output = self.p_project + '/' + s_filename + '.csv'
        return df_results, p_output

    def init(self):
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            print(
                '\n\nThis error most likely means that this notebook is not '
                'configured to use a GPU.  Change this in Notebook Settings via the '
                'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
            raise SystemError('GPU device not found')

        with tf.device('/device:GPU:0'):
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            random_image_gpu = tf.random.normal((100, 100, 100, 3))
            net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
            return tf.math.reduce_sum(net_gpu)

    def read_data(self, p_curr):
        """
        function reads data
        :param p_curr current path
        """
        CTres_Path = os.path.join(p_curr, "CTres.nii.gz")
        imgCTres = sitk.ReadImage(CTres_Path)
        imgCTres = np.expand_dims(sitk.GetArrayFromImage(imgCTres), 0)

        SUV_Path = os.path.join(p_curr, "SUV.nii.gz")
        imgSUV = sitk.ReadImage(SUV_Path)
        imgSUV = np.expand_dims(sitk.GetArrayFromImage(imgSUV), 0)
        return np.concatenate((imgCTres, imgSUV), 0)

    def to_uint8(self, data):
        """
        function converts format to uint8
        :param data
        """
        data -= data.min()
        data /= data.max()
        data *= 255
        return data.astype(np.uint8)

    def nii_to_jpgs(self, input_path, output_dir, rgb=False):
        """
        function converts format nii to jpg
        :param input_path
        :param output_dir
        :param rgb
        """
        output_dir = Path(output_dir)
        data = nib.load(input_path).get_fdata()
        *_, num_slices, num_channels = data.shape
        for channel in range(num_channels):
            volume = data[..., channel]
            volume = self.to_uint8(volume)
            channel_dir = output_dir / f'channel_{channel}'
            channel_dir.mkdir(exist_ok=True, parents=True)
            for slice in range(num_slices):
                slice_data = volume[..., slice]
                if rgb:
                    slice_data = np.stack(3 * [slice_data], axis=2)
                output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
                io.imsave(output_path, slice_data)

    def unet(self, n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS=3):
        """
        function inits U-Net model
        :param n_classes
        :param IMG_HEIGHT
        :param IMG_WIDTH
        :param IMG_CHANNELS
        """
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # s = Lambda(lambda x: x / 255)(inputs)  # normalizes
        s = inputs

        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def nii2numpy(self, nii_path):
        """
        function converts nii to numpy
        :param nii_path
        """
        # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
        mask_nii = nib.load(str(nii_path))
        mask = mask_nii.get_fdata()
        pixdim = mask_nii.header['pixdim']
        voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
        return mask, voxel_vol

    def con_comp(self, seg_array):
        """
        function input a binary segmentation array output: an array with separated (indexed) connected components of the segmentation array
        :param seg_array
        """
        connectivity = 18
        conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
        return conn_comp

    def false_pos_pix(self, gt_array, pred_array):
        """
        function computes number of voxels of false positive connected components in prediction mask
        :param gt_array
        :param pred_array
        """
        pred_conn_comp = self.con_comp(pred_array)
        false_pos = 0
        for idx in range(1, pred_conn_comp.max() + 1):
            comp_mask = np.isin(pred_conn_comp, idx)
            if (comp_mask * gt_array).sum() == 0:
                false_pos = false_pos + comp_mask.sum()
        return false_pos

    def false_neg_pix(self, gt_array, pred_array):
        """
        function computes number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
        :param gt_array
        :param pred_array
        """
        gt_conn_comp = self.con_comp(gt_array)
        false_neg = 0
        for idx in range(1, gt_conn_comp.max() + 1):
            comp_mask = np.isin(gt_conn_comp, idx)
            if (comp_mask * pred_array).sum() == 0:
                false_neg = false_neg + comp_mask.sum()
        return false_neg

    def dice_score(self, mask1, mask2):
        """
        function computes foreground Dice coefficient
        :param mask1
        :param mask2
        """
        overlap = (mask1 * mask2).sum()
        sum = mask1.sum() + mask2.sum()
        dice_score = 2 * overlap / sum
        return dice_score

    def compute_metrics(self, nii_gt_path, nii_pred_path):
        """
        function computes evaluation scores
        :param nii_gt_path
        :param nii_pred_path
        """
        gt_array, voxel_vol = self.nii2numpy(nii_gt_path)
        pred_array, voxel_vol = self.nii2numpy(nii_pred_path)

        false_neg_vol = self.false_neg_pix(gt_array, pred_array) * voxel_vol
        false_pos_vol = self.false_pos_pix(gt_array, pred_array) * voxel_vol
        dice_sc = self.dice_score(gt_array, pred_array)

        return dice_sc, false_pos_vol, false_neg_vol

    def get_augmentation(self, x_train):
        """
        function sets augmentations
        :param x_train
        """
        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2),
        #     ]
        # )

        data_augmentation = keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.Normalization(),
                tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.02),
                tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
                tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
            ]
        )

        # data_augmentation = keras.Sequential(
        #     [
        #         layers.Normalization(),
        #         layers.RandomCrop(128, 128),
        #         layers.RandomZoom(0.5, 0.2),
        #         layers.RandomContrast(0.2),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomRotation(0.02),
        #         layers.RandomWidth(0.2),
        #         layers.RandomHeight(0.2)
        #     ]
        # )

        data_augmentation.layers[0].adapt(x_train)  # sets the state of the normalization layer

        return data_augmentation

    def create_encoder(self, x_train, height, width, dim, b_contrastive):
        """
        function inits encoder model
        :param x_train
        :param height
        :param width
        :param dim
        :param b_contrastive
        """
        shape = (height, width, dim)
        curr_model, _ = self.init_model_image(height, width, dim, None, b_contrastive)
        data_augmentation = self.get_augmentation(x_train)
        inputs = keras.Input(shape=shape)
        augmented = data_augmentation(inputs)
        outputs = curr_model(augmented)
        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-encoder")
        return model

    def add_projection_head(self, encoder, height, width, dim):
        """
        function inits projection model
        :param encoder
        :param height
        :param width
        :param dim
        """
        shape = (height, width, dim)
        projection_units = 128
        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        outputs = layers.Dense(projection_units, activation="relu")(features)
        model = keras.Model(
            inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
        )
        return model

    def create_classifier(self, encoder, height, width, dim, trainable=True):
        """
        function inits classifier
        :param encoder
        :param height
        :param width
        :param dim
        :param trainable
        """
        shape = (height, width, dim)
        learning_rate = 0.001
        hidden_units = 512
        num_classes = 2
        dropout_rate = 0.5

        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = keras.Input(shape=shape)
        features = encoder(inputs)
        features = layers.Dropout(dropout_rate)(features)
        features = layers.Dense(hidden_units, activation="relu")(features)
        features = layers.Dropout(dropout_rate)(features)
        outputs = layers.Dense(num_classes, activation="softmax")(features)

        model = keras.Model(inputs=inputs, outputs=outputs, name="contrastive-clf")

        # _metrics = [keras.metrics.SparseCategoricalAccuracy(),
        #             keras.metrics.Recall(),
        #             keras.metrics.Precision(),
        #             tfa.metrics.F1Score(num_classes=num_classes, average='weighted'),
        #             keras.metrics.AUC()
        #             ]

        _metrics = [keras.metrics.SparseCategoricalAccuracy()]

        # _loss = 'binary_crossentropy'

        _loss = keras.losses.SparseCategoricalCrossentropy()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=_loss,
            metrics=_metrics
        )

        return model

    def plot_acc_loss(self, history, m_name, b_contrastive):
        """
        function plots model results
        :param history
        :param m_name model name
        :param b_contrastive flag for contrastive learning
        """
        if b_contrastive:
            plt.plot(history.history['sparse_categorical_accuracy'])
            # plt.plot(history.history['val_sparse_categorical_accuracy'])
        else:
            plt.plot(history.history['accuracy'])
            # plt.plot(history.history['val_accuracy'])
        # plt.plot(history.history['test_accuracy'])

        # accuracy plot
        plt.title('Model ' + m_name + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_acc = self.p_resource + '/' + m_name + '_acc.png'
        plt.savefig(p_save_acc)
        plt.show()
        plt.clf()

        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.plot(history.history['test_loss'])

        # loss plot
        plt.title('Model ' + m_name + ' Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        p_save_loss = self.p_resource + '/' + m_name + '_loss.png'
        plt.savefig(p_save_loss)
        plt.show()
        plt.clf()

    def save_model(self, o_model, s_model):
        """
        function saves model
        :param o_model model object
        :param s_model model name
        """
        print(f'Saved {s_model}.')
        p_output_model = self.p_project + '/contrastive.pkl'
        joblib.dump(o_model, p_output_model)

    def set_file_list(self):
        """
        function sets files in a list
        """
        l_pet, l_ct, l_ct_res, l_seg, l_suv = list(), list(), list(), list(), list()
        for root, dirs, files in os.walk(self._model.p_resource):
            for dir in dirs:
                child_root = root + '/' + dir
                for sub_root, sub_dirs, sub_files in os.walk(child_root):
                    for child_dir in sub_dirs:
                        grand_root = child_root + '/' + child_dir
                        for sub_root_child, sub_dirs_child, sub_files_child in os.walk(grand_root):
                            for curr_file in sub_files_child:
                                curr_file_path = os.path.join(sub_root_child, curr_file)
                                if 'PET' in curr_file:
                                    l_pet.append(curr_file_path)
                                elif 'CT' in curr_file and 'res' not in curr_file:
                                    l_ct.append(curr_file_path)
                                elif 'CT' in curr_file and 'res' in curr_file:
                                    l_ct_res.append(curr_file_path)
                                if 'SEG' in curr_file:
                                    l_seg.append(curr_file_path)
                                if 'SUV' in curr_file:
                                    l_suv.append(curr_file_path)
        return {'pet': l_pet, 'ct': l_ct, 'ct_res': l_ct_res, 'seg': l_seg, 'suv': l_suv}

    def plot_image(self, x, y):
        """
        function plots images
        :param x model
        :param y model
        """
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))  # plots examples
        for i in tqdm(range(0, 5)):
            rand = np.random.randint(len(x))
            ax[i].imshow(x[rand])
            ax[i].axis('off')
            a = y[rand]
            if a == 1:
                ax[i].set_title('Diseased')
            else:
                ax[i].set_title('Non_Diseased')

    def set_callbacks(self, b_contrastive, test=None):
        """
        function sets callbacks
        :param b_contrastive
        :param test
        """
        _epoch_limit = 10

        if b_contrastive:
            _monitor = 'val_sparse_categorical_accuracy'
        else:
            _monitor = 'val_accuracy'

        early_stopping = EarlyStopping(monitor=_monitor,
                                       mode='max',
                                       patience=_epoch_limit,
                                       verbose=1)

        learning_rate = ReduceLROnPlateau(monitor=_monitor,
                                          mode='max',
                                          patience=5,
                                          factor=0.3,
                                          min_delta=0.00001)

        # board = tf.keras.callbacks.TensorBoard(self.p_project, update_freq=1)

        # history = History_Tensor(test)

        l_callbacks = [early_stopping]

        return l_callbacks

    def model_cv_contrastive(self):
        """
        function predict
        """
        height, width = 128, 128
        # height, width = 256, 256
        # height, width = 512, 512
        # dim = 3
        dim = 1
        b_contrastive = True

        p_y = self._model.validate_path(self.p_output, 'y', 'csv')
        y = pd.read_csv(p_y)

        x = self.process(height, width, dim)
        x = x['x_images']

        df_results, p_output_results = self.init_results()

        # height, width = 64, 64
        height, width = 128, 128
        # height, width = 256, 256
        # height, width = 512, 512
        # dim = 3
        dim = 1

        # learning_rate = 0.01
        learning_rate = 0.001

        temperature = 0.05
        # temperature = 0.1
        # temperature = 0.2

        # num_epochs = 50
        num_epochs = 200
        # num_epochs = 1

        # batch_size = 32
        batch_size = 64
        # batch_size = 512
        # batch_size = 256
        # batch_size = 128

        # self.init()

        # y = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2)
        # y = y.transpose(1, -1).squeeze(-1)

        # x = shuffle(x, random_state=42)
        # x = torch.tensor(x)
        # x = np.reshape(x, (x.shape[0], height, width, dim))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        # with validation set
        # i_split = int(len(x) * 0.9)
        # train_images, val_images = x[:i_split], x[i_split:]
        # train_set = DataLoader(train_images, batch_size=batch_size, shuffle=True,
        #                        num_workers=2, pin_memory=True, drop_last=True)
        # valid_set = DataLoader(val_images, batch_size=batch_size, shuffle=False,
        #                        num_workers=2, pin_memory=True, drop_last=False)

        l_callbacks = self.set_callbacks(b_contrastive, x_test)

        encoder = self.create_encoder(x_train, height, width, dim, b_contrastive)

        encoder.summary()

        print('Training contrastive encoder...')

        encoder = self.create_encoder(x_train, height, width, dim, b_contrastive)
        encoder_with_projection_head = self.add_projection_head(encoder, height, width, dim)
        encoder_with_projection_head.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=SupervisedContrastiveLoss(temperature),
        )
        encoder_with_projection_head.summary()

        history = encoder_with_projection_head.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

        print('Training projection network...')
        classifier = self.create_classifier(encoder, height, width, dim, trainable=False)
        s_model = 'Contrastive Model'

        # (1) without validation set
        history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, callbacks=l_callbacks)

        # (2) with validation set
        # curr_steps_per_epoch = x_train.shape[0] // batch_size
        # curr_validation_steps = x_val.shape[0] // batch_size
        # history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
        #                          validation_data=(x_val, y_val), callbacks=l_callbacks,
        #                          steps_per_epoch=curr_steps_per_epoch, validation_steps=curr_validation_steps,
        #                          shuffle=True)

        scores = classifier.evaluate(x_test, y_test)
        loss = round(scores[0] * 100, 3)
        accuracy = round(scores[1] * 100, 3)
        self.plot_acc_loss(history, s_model, b_contrastive)
        print(f'Model: {s_model}, Test Accuracy: {accuracy}%, Test Loss: {loss}%')

        df_results.loc[0, 'Model'] = s_model
        df_results.loc[0, 'Accuracy'] = accuracy
        df_results.loc[0, 'Loss'] = loss
        df_results.to_csv(path_or_buf=p_output_results, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')

        y_preds = classifier.predict(x_test).reshape(-1, 1)
        # y_probs = classifier.predict_proba(x_test)[:, 1]

        nii_pred_path = self.p_project+'/preds.csv'
        df_preds = pd.DataFrame.from_records(y_preds)
        df_preds.to_csv(path_or_buf=nii_pred_path, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
        p_preds = self.p_project + '/test/PRED.nii.gz'
        nii_preds_file = nib.Nifti1Image(y_preds, np.eye(4))
        nib.save(nii_preds_file, p_preds)

        nii_gt_path = self.p_project+'/gt.csv'
        df_gt = pd.DataFrame.from_records(y_test)
        df_gt.to_csv(path_or_buf=nii_gt_path, mode='w', index=False, na_rep='', header=True, encoding='utf-8-sig')
        p_gt = self.p_project + '/test/GT.nii.gz'
        nii_gt_file = nib.Nifti1Image(y_test, np.eye(4))
        nib.save(nii_gt_file, p_gt)

        # self.save_model(classifier, s_model)
        # classifier.save_weights(self.p_project + '/contrastive.h5')

        dice_sc, false_pos_vol, false_neg_vol = self.compute_metrics(nii_gt_path, nii_pred_path)
        csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
        csv_rows = ['y_true', dice_sc, false_pos_vol, false_neg_vol]
        with open('metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(csv_header)
            writer.writerows(csv_rows)

    def run(self):
        """
        main function: runs all experiments
        """
        self.model_cv()  # TTA Model (Novel)
        # self.model_cv_baseline()  # Baseline Model
        # self.model_cv_train()  # TTA-Trained-on-Augmentations Model
        # self.get_metrics_average()  # Displays Results
        # self.model_cv_contrastive()  # Contrastive Learning Model
        # self.model_cv_active()  # Active Learning Model
        # self.statistical_test()  # Friedman Statistical Test
        print(f'Done training and testing models.')
