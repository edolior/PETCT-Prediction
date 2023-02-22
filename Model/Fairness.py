import pandas as pd
import numpy as np
import os
import copy
import joblib
from io import StringIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.calibration import *
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import *


# ---------------------------------------------------------------------------------------
# Fairness Class:
#
# Plots sub-group calibrations
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Fairness:

    _model = None

    def __init__(self, r_model):
        """
        Fairness Constructor
        """
        self._model = r_model
        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_output = self.p_project + r'\output\output_parser'
        self.p_classifier = self.p_project + r'\output\output_classifier'
        self.p_models = self.p_classifier + r'\models'
        if self._model.b_vpn:
            self.p_project = self._model.set_vpn_dir(self.p_project)
            self.p_output = self._model.set_vpn_dir(self.p_output)
            self.p_classifier = self._model.set_vpn_dir(self.p_classifier)
            self.p_models = self._model.set_vpn_dir(self.p_models)
        self.l_targets_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']

    def load_model(self, curr_label):
        """
        function loads model from disk
        :param curr_label model name
        """
        # curr_type = '_diff_'
        curr_type = '_top_'
        curr_fold = '4'
        p_load = self._model.validate_path(self.p_models, curr_label+curr_type+curr_fold, 'pkl')
        o_model = joblib.load(p_load)
        filename = self._model.get_filename(p_load)
        print(f'Model {filename} has been loaded.')
        return o_model

    def load_data(self, curr_label):
        """
        function loads training and test sets from disk
        :param curr_label set name
        """
        # curr_type = '_diff_'
        curr_type = '_top_'
        curr_fold = '4'
        p_x_train = self._model.validate_path(self.p_models, curr_label+curr_type+curr_fold+'_x_train', 'csv')
        p_x_test = self._model.validate_path(self.p_models, curr_label+curr_type+curr_fold+'_x_test', 'csv')
        p_y_train = self._model.validate_path(self.p_models, curr_label+curr_type+curr_fold+'_y_train', 'csv')
        p_y_test = self._model.validate_path(self.p_models, curr_label+curr_type+curr_fold+'_y_test', 'csv')
        x_train = pd.read_csv(p_x_train)
        x_test = pd.read_csv(p_x_test)
        y_train = pd.read_csv(p_y_train)
        y_test = pd.read_csv(p_y_test)
        return x_train, x_test, y_train, y_test

    def histogram_plot(self, x_train, l_features, l_bins_age_1d=None, l_s_ages=None):
        for curr_feature in l_features:
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f'{curr_feature} Histogram Plot')
            x_train[curr_feature].hist()
            plt.show()
        if l_bins_age_1d is not None:
            # plt.hist(x_train['Age'], bins=l_bins_age_1d)
            # plt.hist(x_train['Age'], bins=len(l_bins_age_1d))
            # plt.hist(x_train['Age'], bins=np.arange(min(x_train['Age']), max(x_train['Age']) + len(l_bins_age_1d), len(l_bins_age_1d)))
            srs_groups_age = pd.cut(x_train['Age'], bins=l_bins_age_1d, labels=l_s_ages)

            plt.hist(srs_groups_age, color="steelblue", lw=0)
            # plt.hist(srs_groups_age, color="steelblue", ec="steelblue")

            # df_groups_age = pd.DataFrame(srs_groups_age, columns=['Group'])  # sort bars
            # df_groups_age.sort_values('Group')
            # plt.hist(df_groups_age['Group'])

            # df_groups_age = pd.concat((x_train['Age'], srs_groups_age), axis=1)
            # df_groups_age = df_groups_age.rename(columns={df_groups_age.columns[1]: 'Group'})
            # df_groups_age['Group'].hist()

            plt.grid()
            plt.show()

    def subset_evaluation(self, x_test, l_features):
        for curr_feature in l_features:
            l_subsets = list(x_test[curr_feature].unique())[:-1]
            for i in range(len(l_subsets)):
                subset = l_subsets[i]
                print(f'Subset #{i}: {subset}')

    def multi_roc(self, o_model, l_models, l_x_test, l_y_test, y_prob, y_test_curr):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 30)
        # fig.figure(figsize = (20,10))
        for s_curr_model, x_test, y_test in zip(l_models, l_x_test, l_y_test):
            try:
                # y_subset_preds = o_model.predict(x_test)
                y_subset_probs = o_model.predict_proba(x_test)[:, 1]
                auc_roc = roc_auc_score(y_test, y_subset_probs)
                fpr, tpr, thresholds = roc_curve(y_test, y_subset_probs)
                plt.plot(fpr, tpr, linewidth=1, label=f'{s_curr_model} (AUC:{auc_roc:.2f})')
            except Exception as e:
                print(f'Exception found {s_curr_model}: {str(e)}')
                continue

        fpr, tpr, thresholds = roc_curve(y_test_curr, y_prob)
        plt.plot(fpr, tpr, linewidth=1, label=f'Class Probability (AUC:{roc_auc_score(y_test_curr, y_prob):.2f})')

        fig.suptitle('ROC curve')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        plt.legend()
        plt.show()

    def calibration(self, x_train, x_test, y_train, y_test, l_features, o_model, s_model, l_models):
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(4, 2)
        colors = plt.cm.get_cmap("Dark2")

        ax_calibration_curve = fig.add_subplot(gs[:2, :2])
        calibration_displays = {}
        for i in range(len(l_models)):
            name = l_models[i]
            o_model.fit(x_train, y_train)
            display = CalibrationDisplay.from_estimator(
                o_model,
                x_test,
                y_test,
                n_bins=10,
                name=name,
                ax=ax_calibration_curve,
                color=colors(i),
            )
            calibration_displays[name] = display

        ax_calibration_curve.grid()
        ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

        # Add histogram
        grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
        for i in range(len(l_models)):
            name = l_models[i]
            row, col = grid_positions[i]
            ax = fig.add_subplot(gs[row, col])

            ax.hist(
                calibration_displays[name].y_prob,
                range=(0, 1),
                bins=10,
                label=name,
                color=colors(i),
            )
            ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

        plt.tight_layout()
        plt.show()

        # o_calibrated = CalibratedClassifierCV(o_model, cv=5, method='sigmoid')
        # o_calibrated.fit(x_train, y_train)
        # preds_calib = o_calibrated.predict(x_test)

    def feature_preprocess(self, x_train, x_test, y_train, y_test, l_features, o_model, curr_label, prob_diagnosis):
        l_x_test, l_y_test, l_models = [x_test], [y_test], ['LogisticRegression']
        for curr_feature in l_features:
            feature_test = x_test.query(f'Current Feature == "{curr_feature}"')
            x_test_feature, y_test_feature = feature_test[x_test], feature_test[curr_label]
            s_model = f'LogisticRegression-{curr_feature} Test Set'
            l_x_test.append(x_test_feature)
            l_y_test.append(y_test_feature)
            l_models.append(s_model)
            self.multi_roc(o_model, l_models, l_x_test, l_y_test, prob_diagnosis, y_test)

    def calibration_probability(self, o_model, s_model, x_train, x_val, x_test, y_train, y_val, y_test):
        model_to_probs = {}
        model_str_to_trained_model = {}

        if s_model == 'SVC':
            o_model = SVC(probability=True)
            o_model.fit(x_train, y_train)
        elif s_model == 'LR':
            o_model = LogisticRegression(solver='liblinear')
            o_model.fit(x_train, y_train)

        pred_probs_train = o_model.predict_proba(x_train)[:, 1]
        pred_probs_test = o_model.predict_proba(x_test)[:, 1]
        pred_probs_valid = o_model.predict_proba(x_val)[:, 1]

        model_to_probs[s_model] = {'train': pred_probs_train, 'test': pred_probs_test, 'valid': pred_probs_valid}

        plt.figure(figsize=(20, 4))

        plt.subplot(1, 2, 1)
        sns.distplot(pred_probs_train)
        plt.title(f"{s_model} - train", fontsize=20)

        plt.subplot(1, 2, 2)
        sns.distplot(pred_probs_test)
        plt.title(f"{s_model} - test", fontsize=20)

        model_str_to_trained_model[s_model] = o_model

        return model_to_probs, model_str_to_trained_model

    def plot_calibration_probability(self, model_to_probs, y_test):
        for model_str, pred_prob_dict in model_to_probs.items():
            pred_probs = pred_prob_dict['test']
            pred_probs_space = np.linspace(pred_probs.min(), pred_probs.max(), 10)
            empirical_probs = []
            pred_probs_midpoints = []

            for i in range(len(pred_probs_space) - 1):
                empirical_probs.append(
                    np.mean(y_test[(pred_probs > pred_probs_space[i]) & (pred_probs < pred_probs_space[i + 1])]))
                pred_probs_midpoints.append((pred_probs_space[i] + pred_probs_space[i + 1]) / 2)

            plt.figure(figsize=(10, 4))
            plt.plot(pred_probs_midpoints, empirical_probs, linewidth=2, marker='o')
            plt.title(f"{model_str}", fontsize=20)
            plt.xlabel('predicted prob', fontsize=14)
            plt.ylabel('empirical prob', fontsize=14)
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.legend(['original', 'ideal'], fontsize=20)

    def calibrate_model(self, model_to_probs, y_test):
        model_str_to_calibrator = {}

        for model_str, pred_prob_dict in model_to_probs.items():
            lr_model = LogisticRegression()  # train calibration model
            lr_model.fit(pred_prob_dict['test'].reshape(-1, 1), y_test)
            pred_probs = pred_prob_dict['valid']
            pred_probs_space = np.linspace(pred_probs.min(), pred_probs.max(), 10)
            empirical_probs = []
            pred_probs_midpoints = []

            for i in range(len(pred_probs_space) - 1):
                empirical_probs.append(
                    np.mean(valid_y[(pred_probs > pred_probs_space[i]) & (pred_probs < pred_probs_space[i + 1])]))
                pred_probs_midpoints.append((pred_probs_space[i] + pred_probs_space[i + 1]) / 2)

            calibrated_probs = lr_model.predict_proba(np.array([0.0] + pred_probs_midpoints + [1.0]).reshape(-1, 1))[:,
                               1]

            plt.figure(figsize=(10, 4))
            plt.plot(pred_probs_midpoints, empirical_probs, linewidth=2, marker='o')
            plt.title(f"{model_str}", fontsize=20)
            plt.xlabel('predicted prob', fontsize=14)
            plt.ylabel('empirical prob', fontsize=14)
            plt.plot([0.0] + pred_probs_midpoints + [1.0], calibrated_probs, linewidth=2, marker='o')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.legend(['original', 'calibrated', 'ideal'], fontsize=20)
            model_str_to_calibrator[model_str] = lr_model

    def run(self):
        l_targets = ['A+B']
        # l_features = ['Gender', 'Age', 'Ethnicity']
        l_features = ['Gender', 'Age']
        # l_features = ['Age']
        l_bins_age = [(0, 30), (30, 50), (50, 70), (70, 100)]
        l_s_ages = ['0-30', '30-50', '50-70', '70-100']
        l_bins_age_1d = [0, 30, 50, 70, 100]
        l_ethnicity_bins = ['Other/Unknown', 'Caucasian', 'African American', 'Hispanic', 'Native American', 'Asian']
        s_project = 'PET-CT'
        # s_model = 'LogisticRegression'
        # s_model = 'DecisionTree'
        s_model = 'XGBoost'

        for curr_label in l_targets:
            x_train, x_test, y_train, y_test = self.load_data(curr_label)

            o_model = self.load_model(curr_label)

            # self.histogram_plot(x_train, l_features, l_bins_age_1d, l_s_ages)

            for curr_feature in l_features:
                y_test_curr = copy.deepcopy(y_test)
                l_x_test, l_y_test, l_models = list(), list(), list()

                # o_model = LogisticRegressionCV().fit(x_train, y_train)
                # o_model = XGBClassifier(max_depth=6, eta=0.025, n_estimators=250, objective='binary:logistic',
                #                         random_state=9, verbosity=0, use_label_encoder=False).fit(x_train, y_train)

                y_probs = o_model.predict_proba(x_test)
                y_probs = y_probs[:, 1].reshape(-1, 1)[:, 0]

                if curr_feature == 'Age':
                    for i_bin in range(len(l_bins_age)):
                        x_train_feature, x_test_feature = copy.deepcopy(x_train), copy.deepcopy(x_test)
                        y_train_feature, y_test_feature = copy.deepcopy(y_train), copy.deepcopy(y_test)
                        age_group = l_bins_age[i_bin]

                        df_test_curr_age = x_test_feature.query(f'Age >= {age_group[0]} & Age < {age_group[1]}')
                        curr_indices = list(df_test_curr_age.index)
                        x_test_feature_sub, y_test_feature_sub = x_test_feature.reindex(curr_indices), y_test_feature.reindex(curr_indices)

                        s_curr_model = s_model+'_'+l_s_ages[i_bin]
                        x_test_feature_sub.reset_index(inplace=True, drop=True)
                        y_test_feature_sub.reset_index(inplace=True, drop=True)
                        l_x_test.append(x_test_feature_sub)
                        l_y_test.append(y_test_feature_sub)
                        l_models.append(s_curr_model)

                    self.multi_roc(o_model, l_models, l_x_test, l_y_test, y_probs, y_test_curr)
                    self.calibration(x_train, x_test, y_train, y_test, l_features, o_model, curr_label, l_models)
                    self.subset_evaluation(x_test, l_features)

                elif curr_feature == 'Gender':
                    for i_gender in range(len([0, 1])):
                        x_train_feature, x_test_feature = copy.deepcopy(x_train), copy.deepcopy(x_test)
                        y_train_feature, y_test_feature = copy.deepcopy(y_train), copy.deepcopy(y_test)

                        df_test_curr_gender = x_test_feature.query(f'Gender=={i_gender}')
                        curr_indices = list(df_test_curr_gender.index)
                        x_test_feature_sub, y_test_feature_sub = x_test_feature.reindex(
                            curr_indices), y_test_feature.reindex(curr_indices)

                        if i_gender == 0:
                            s_gender = 'Female'
                        else:
                            s_gender = 'Male'

                        s_curr_model = s_model + '_' + s_gender
                        x_test_feature_sub.reset_index(inplace=True, drop=True)
                        y_test_feature_sub.reset_index(inplace=True, drop=True)
                        l_x_test.append(x_test_feature_sub)
                        l_y_test.append(y_test_feature_sub)
                        l_models.append(s_curr_model)

                    self.multi_roc(o_model, l_models, l_x_test, l_y_test, y_probs, y_test_curr)
                    self.calibration(x_train, x_test, y_train, y_test, l_features, o_model, curr_label, l_models)

        model_to_probs, model_str_to_trained_model = self.calibration_probability(o_model, s_model, x_train, x_val,
                                                                                  x_test, y_train, y_val, y_test)
        self.plot_calibration_probability(model_to_probs, y_test)
        self.calibrate_model(model_to_probs, y_test)
        uncal_prob = o_model.predict_proba(x_test)[:, 1][0]
        print('Uncalibrated Prob:', uncal_prob)
        # m_calib_lr = model_str_to_calibrator['rf']  # calibration layer self-supervised
        # cal_prob = lr.predict_proba(np.array([[uncal_prob]]))[:, 1][0]
        # print('Calibrated Prob:', cal_prob)
