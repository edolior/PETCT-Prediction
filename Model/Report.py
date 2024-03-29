import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
        """
        Report Constructor
        """
        self._model = r_model

        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_output = self.p_project + r'\output\output_parser'
        self.p_classifier = self.p_project + r'\output\output_classifier'

        if self._model.b_vpn:
            self.p_project = self._model.set_vpn_dir(self.p_project)
            self.p_output = self._model.set_vpn_dir(self.p_output)
            self.p_classifier = self._model.set_vpn_dir(self.p_classifier)

        self.l_target_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.l_target_cols_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']

        self.l_not_tfidf_cols = [
            'Age', 'Gender', 'Timestamp', 'VariableAmount', 'GlucoseLevel', 'TestStartTime', 'A+B', 'C+D', 'E+F', 'H+I',
            'G', 'J', 'K', 'L', 'M', 'N', 'כללית', 'לא ידוע', 'לאומית', 'מאוחדת', 'מוסד רפאחר', 'מכבי', 'מסוק',
            'עצמית פניה', 'צה"ל', 'שרות בתי הסוהר', 'מחלקה כירורגית', 'מחלקה כירורגית יחידה ארגונית',
            'מחלקה פנימית א יחידה ארגונית', 'מכון איזוטופים יחידה ארגונית', 'מיפוי FDG PETגלוקוז מסומן',
            'מיפוי FDG PETגלוקוז מסומןCTללא חיו ב'
        ]

    def data_percentage(self, df_data, b_this_cols):
        """
        function returns percentage of missing data
        :param df_data dataset
        :param b_this_cols boolean flag for a specific column
        """
        if not b_this_cols:
            l_cols_demographic = ['Age', 'Gender', 'HealthCare', 'Unit', 'ServiceHistory', 'Service']
            l_cols_settings = ['Timestamp', 'VariableAmount', 'VariableLocation', 'VariableRange', 'GlucoseLevel',
                               'TestStartTime', 'TestSetting']
            l_cols_features = ['BreastArmPit', 'Chest', 'Lung', 'ChestLung', 'HeadNeck', 'SkeletonTissue',
                               'StomachPelvis', 'ArrivalReason', 'Summary']
            l_iterations = [l_cols_demographic, l_cols_settings, l_cols_features]
        else:
            l_iterations = list(df_data.columns)
            s_start, s_end = self.get_tfidf_cols(df_data)
            i_stop = df_data.columns.get_loc(s_end)
            l_iterations = l_iterations[:i_stop]
            l_iterations = [l_iterations]
        fig_name = 0
        s_type = ''
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
            if fig_name == 0:
                s_type = 'Demographics'
            elif fig_name == 1:
                s_type = 'Examination Settings'
            elif fig_name == 2:
                s_type = 'Body Sectors'
            s_fig_title = 'Existing' + ' ' + s_type
            s_fig_y_title = 'Percentage Data'
            fig.suptitle(s_fig_title, fontsize=20)
            plt.ylabel(s_fig_y_title, fontsize=16)
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
        """
        function returns ratios of classes
        :param df_data dataset
        """
        s_pie = False
        # s_pie = True
        d_percentages = {}
        d_ratios = {'A+B': '1:0.5', 'C+D': '1:1', 'E+F': '1:31', 'G': '1:13', 'H+I': '1:6', 'J': '1:6', 'K': '1:20', 'L': '1:6', 'M': '1:131', 'N': '1:18'}
        l_names = ['1:0.5', '1:1', '1:31', '1:13', '1:6', '1:6', '1:20', '1:6', '1:131', '1:18']
        l_ratios = []
        l_percentages = []
        df_classes = df_data[self.l_target_cols_merged]
        length = df_data.shape[0]
        print('Classes Counts: ')
        for curr_col in df_classes[self.l_target_cols_merged]:
            srs_numeric = pd.to_numeric(df_data[curr_col], errors='coerce')
            value = srs_numeric.sum()
            i_count = int(value)
            i_neg = length - i_count
            d_percentages[curr_col] = i_count
            # d_percentages[curr_col] = [i_count, i_neg]
            l_percentages.append([i_count, i_neg])
            f_ratio = round(i_count/length, 2)
            l_ratios.append(f_ratio)
        fig = plt.figure()
        fig.suptitle('Existing Classes Chart', fontsize=20)
        plt.ylabel('Count', fontsize=16)

        if s_pie:
            palette_color = sns.color_palette('bright')
            plt.pie(l_ratios, labels=self.l_target_cols_merged, colors=palette_color, autopct='%.0f%%')
        else:
            # df_plot = pd.DataFrame.from_dict(d_percentages, orient='columns')

            # ax = sns.barplot(data=df_plot, errwidth=0)
            # ax = sns.barplot(x=df_plot.index, y=df_plot.columns, data=df_plot, errwidth=0)
            # for container in ax.containers:
            #     ax.bar_label(container, )

            # l_elements = list(range(1, len(l_names) + 1))
            # plt.bar(l_elements, df_plot, tick_label=l_names)
            # for i in range(len(df_plot)):
            #     df_plot[i] = float("{:.3f}".format(df_plot[i]))
            #     plt.annotate(str(df_plot[i]), xy=(l_elements[i], df_plot[i]), ha='center', va='bottom')

            plt.bar(*zip(*d_percentages.items()))

        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        p_save = self._model.validate_path(self.p_output, 'classes_distribution', '.png')
        fig1.savefig(p_save, dpi=600)

    def values_count(self, df_data):
        """
        function returns ratios of features
        :param df_data dataset
        """
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

    def get_ratios(self):
        """
        function returns ratios
        """
        y_curr = pd.read_csv(self.p_classifier + '/y.csv')
        df_majority, df_minority = None, None
        for curr_col in y_curr[self.l_target_cols_merged]:
            s_target = curr_col
            if isinstance(y_curr, pd.Series):
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

    def plot_curve(self, x_value, y_value, func_label, x_label, y_label, fig_name):
        """
        function displays and saves plot
        :param x_value
        :param y_value
        :param func_label label
        :param x_label x axis name
        :param y_label y axis name
        :param fig_name figure name
        """
        plt.plot(x_value, y_value, marker='.', label=func_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

        p_save = self.p_output + '\\' + fig_name + '.png'
        if self._model.b_vpn:
            p_save = self.p_output + '/' + fig_name + '.png'

        plt.savefig(p_save, bbox_inches='tight')

    def get_tfidf_cols(self, df_data):
        """
        function returns index ranges after performing TF-IDF to the assigned sub-model
        :param df_data
        """
        l_cols = list(df_data.columns)
        col = 0
        length = len(l_cols)
        for j in range(length):
            if l_cols[j] not in self.l_not_tfidf_cols:
                col = j
                break
        if col == 0:
            for k in range(length, -1, -1):
                if l_cols[k - 1] not in self.l_not_tfidf_cols:
                    col = k - 1
                    break
                if k == 0:
                    break
        s_end = l_cols[col]
        s_st1 = l_cols[length - 1]
        s_st2 = l_cols[0]
        if s_st2 in self.l_not_tfidf_cols:
            s_start = s_st1
        else:
            s_start = s_st2
        return s_start, s_end

    def analysis(self, df_data):
        """
        function performs statistical analysis on the dataset
        :param df_data
        """
        l_appendix = ['Age', 'Gender',  'GlucoseLevel', 'VariableAmount']
        d_calc = dict()
        for curr_col in l_appendix:
            if curr_col != 'Gender':
                mean_var = df_data[curr_col].mean()
                d_calc[curr_col] = round(mean_var, 3)
            if curr_col == 'Gender':
                count_m = df_data[curr_col].sum()
                d_calc['Male'] = int(count_m)
                d_calc['Female'] = int(df_data.shape[0] - count_m)
            if curr_col == 'Age' or curr_col == 'VariableAmount' or curr_col == 'GlucoseLevel':
                min = df_data[curr_col].min()
                max = df_data[curr_col].max()
                d_calc['Min'+curr_col] = int(min)
                d_calc['Max'+curr_col] = int(max)
        for key, value in d_calc.items():
            print(f'Feature: {key}: {value}.')
        self._model.set_dict_to_csv(d_calc, 'appendix', self.p_output)
