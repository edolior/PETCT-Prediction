import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------------------
# Report Class:
#
# Outputs different types of data analysis reports
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Report:

    _model = None

    def __init__(self, r_model):
        self._model = r_model

        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_output = self.p_project + r'\output\output_parser'

        if self._model.b_vpn:
            self.p_project = self._model.set_vpn_dir(self.p_project)
            self.p_output = self._model.set_vpn_dir(self.p_output)

        self.l_target_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.l_target_cols_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']

    def data_percentage(self, df_data, b_this_cols):
        if not b_this_cols:
            l_cols_demographic = ['Age', 'Gender', 'HealthCare', 'Unit', 'ServiceHistory', 'Service']
            l_cols_settings = ['Timestamp', 'VariableAmount', 'VariableLocation', 'VariableRange', 'GlucoseLevel',
                               'TestStartTime', 'TestSetting']
            l_cols_features = ['BreastArmPit', 'Chest', 'Lung', 'ChestLung', 'HeadNeck', 'SkeletonTissue', 'StomachPelvis',
                               'ArrivalReason', 'Summary']
            l_iterations = [l_cols_demographic, l_cols_settings, l_cols_features]
        else:
            l_iterations = list(df_data.columns)
            s_start, s_end = self.get_tfidf_cols(df_data)
            i_stop = df_data.columns.get_loc(s_end)
            l_iterations = l_iterations[:i_stop]
            l_iterations = [l_iterations]
        fig_name = 0
        for curr_list in l_iterations:
            d_percentages = {}
            for curr_col in df_data[curr_list]:
                i_null = df_data[curr_col].isnull()
                i_full = df_data[curr_col].shape[0]
                i_null_col = float(i_null.sum())
                i_exist = i_full - i_null_col
                perc = (i_exist / i_full) * 100
                formatted_perc = "{:.2f}".format(perc)
                d_percentages[curr_col] = float(formatted_perc)
                print('%s, Data Exists: %d (%.2f%%)' % (curr_col, i_exist, perc))
            fig = plt.figure()
            fig.suptitle('Existing Values Chart', fontsize=20)
            plt.ylabel('Percentage Existing', fontsize=16)
            if len(d_percentages) > 6:
                plt.xticks(rotation=45, ha='right')
            plt.bar(*zip(*d_percentages.items()))
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig_name += 1

            p_save = self._model.validate_path(self.p_output, str(fig_name), 'png')
            fig1.savefig(p_save, dpi=600)

    def unique_count(self, df_data, col):
        """
        function detects recurrent patients with matching timestamps (same date)
        :param df_data dataframe
        :param col label column
        :return removes duplicate patients and displays recurrent patients
        """
        s_bool = df_data.duplicated(subset=[col], keep=False)
        df_recurrent = df_data.iloc[s_bool[s_bool].index, ]
        uniques = np.unique(df_data[col])
        uniques_size = uniques.size
        length = df_data.shape[0]
        i_recurrents = length-uniques_size
        print(f'Number of Recurrent IDs: {i_recurrents}')
        self._model.set_df_to_csv(df_recurrent, 'df_recurrent', self.p_output, s_na='', b_append=False, b_header=True)
        df_uniques_only = df_data.drop_duplicates(subset=[col])
        df_uniques_only = df_uniques_only.reset_index()
        df_uniques_only.drop('index', inplace=True, axis=1)
        print(f'Dropped recurrent patients with same values of timestamps.')
        self._model.set_df_to_csv(df_uniques_only, 'df_features', self.p_output, s_na='', b_append=False, b_header=True)

    def class_count(self, df_data):
        d_percentages = {}
        df_classes = df_data[self.l_target_cols_merged]
        print('Classes Counts: ')
        for curr_col in df_classes[self.l_target_cols_merged]:
            srs_numeric = pd.to_numeric(df_data[curr_col], errors='coerce')
            value = srs_numeric.sum()
            d_percentages[curr_col] = int(value)
        fig = plt.figure()
        fig.suptitle('Existing Classes Chart', fontsize=20)
        plt.ylabel('Count', fontsize=16)
        plt.bar(*zip(*d_percentages.items()))
        fig1 = plt.gcf()
        plt.show()
        plt.draw()

        p_save = self.p_output + '\\' + 'classes_distribution' + '.png'
        if self._model.b_vpn:
            p_save = self.p_output + '/' + 'classes_distribution' + '.png'

        fig1.savefig(p_save, dpi=600)

    @staticmethod
    def existing_data(df_data):
        categories = list(df_data.columns.values)
        sns.set(font_scale=2)
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(categories, df_data.iloc[:, 2:].sum().values)
        plt.title("Existing Labels Chart", fontsize=24)
        plt.ylabel('Number of Samples ', fontsize=18)
        plt.xlabel('Class Type ', fontsize=18)
        rects = ax.patches  # adding the text labels
        labels = df_data.iloc[:, 2:].sum().values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom', fontsize=18)
        plt.show()

    @staticmethod
    def multi_label_count(df_data):
        i_rows = df_data.iloc[:, 2:].sum(axis=1)
        multiLabel_counts = i_rows.value_counts()
        multiLabel_counts = multiLabel_counts.iloc[1:]
        sns.set(font_scale=2)
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
        plt.title("Classes With Multiple Labels ")
        plt.ylabel('Number of Samples', fontsize=18)
        plt.xlabel('Number of Labels', fontsize=18)
        rects = ax.patches        # adding the text labels
        labels = multiLabel_counts.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
        plt.show()

    def values_count(self, df_data):
        l_categoricals = ['Gender', 'כללית', 'לא ידוע', 'לאומית', 'מאוחדת', 'מוסד רפאחר', 'מכבי', 'מסוק',
                                'פניה עצמית', 'צה"ל', 'שרות בתי הסוהר', 'מחלקה כירורגית',
                          'מחלקה כירורגית יחידה ארגונית', 'מחלקה פנימית א יחידה ארגונית',
                          'מכון איזוטופים יחידה ארגונית']
        d_percentages = {}
        srs_results = df_data[l_categoricals].sum()
        print('Values Counts: ')
        for index, value in srs_results.items():
            perc = value / df_data.shape[0] * 100
            formatted_perc = "{:.2f}".format(perc)
            curr_value = float(formatted_perc)
            d_percentages[index] = curr_value
            if curr_value > 50:
                print(f'Column: {index} has {curr_value}% data.')
        fig = plt.figure()
        fig.suptitle('Values Chart', fontsize=20)
        plt.ylabel('Percentage Values', fontsize=16)
        plt.bar(*zip(*d_percentages.items()))
        fig1 = plt.gcf()
        plt.show()
        plt.draw()

        p_save = self.p_output + '\\' + 'values_distribution' + '.png'
        if self._model.b_vpn:
            p_save = self.p_output + '/' + 'values_distribution' + '.png'

        fig1.savefig(p_save, dpi=600)

    def plot_curve(self, x_value, y_value, func_label, x_label, y_label, fig_name):
        plt.plot(x_value, y_value, marker='.', label=func_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

        p_save = self.p_output + '\\' + fig_name + '.png'
        if self._model.b_vpn:
            p_save = self.p_output + '/' + fig_name + '.png'

        plt.savefig(p_save, bbox_inches='tight')

    def plot_cm(self, y_test, y_pred, labels):
        cm = confusion_matrix(y_test, y_pred, labels)
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
        print('True Positive | False Negative | False Positive | True Negative' + '\n')
        print(tp, fn, fp, tn)

        matrix = classification_report(y_test, y_pred, labels=[1, 0], output_dict=True)
        df_cm = pd.DataFrame(matrix).transpose()
        print('Classification report : \n', matrix)

        p_save = self._model.p_output + r'\df_confusion_matrix.csv'
        if self._model.b_vpn:
            p_save = self._model.p_output + '/df_confusion_matrix.csv'

        df_cm.to_csv(p_save, index=False)

    def get_tfidf_cols(self, df_data):
        l_all_cols = ['CaseID', 'Age', 'Gender', 'כללית', 'לא ידוע', 'לאומית', 'מאוחדת', 'מוסד רפאחר', 'מכבי',
                      'מסוק',
                      'פניה עצמית', 'צה"ל', 'שרות בתי הסוהר', 'מחלקה כירורגית',
                      'מחלקה כירורגית יחידה ארגונית', 'מחלקה פנימית א יחידה ארגונית',
                      'מכון איזוטופים יחידה ארגונית', 'Timestamp', 'VariableAmount', 'VariableLocation',
                      'VariableRange', 'GlucoseLevel', 'TestStartTime', 'TestSetting', 'ServiceHistory', 'Service',
                      'A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']
        l_cols = list(df_data.columns)
        col = 0
        length = len(l_cols)
        for j in range(length):
            if l_cols[j] not in l_all_cols:
                col = j
                break
        if col == 0:
            for k in range(length, -1, -1):
                if l_cols[k - 1] not in l_all_cols:
                    col = k - 1
                    break
                if k == 0:
                    break
        s_end = l_cols[col]
        s_st1 = l_cols[length - 1]
        s_st2 = l_cols[0]
        if s_st2 in l_all_cols:
            s_start = s_st1
        else:
            s_start = s_st2
        return s_start, s_end
