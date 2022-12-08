from Model.Parser import Parser
from Model.Report import Report
from Model.TextAug import TextAug
from Model.Classifier import Classifier

import pandas as pd
import numpy as np
import os
import csv
import time
import multiprocessing
import subprocess
import re
from tika import parser
from tqdm import tqdm

import pickle  # with vpn
# import pickle5 as pickle  # when no vpn


# ---------------------------------------------------------------------------------------
# Model Class:
#
# Primary class for running the entire flow of Cancer Prediction.
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Model:
    _controller = None
    _parser = None
    _report = None

    i_process_counter = 0
    i_files_counter = 0

    df_data = None

    l_files = []

    def __init__(self, r_controller, d_config):
        """
        Model Constructor
        """
        self._controller = r_controller
        self.d_config = d_config
        self.b_processing = self.d_config['b_processing']
        self.i_sample = self.d_config['i_sample']
        self.b_vpn = self.d_config['b_vpn']
        self.b_tta = self.d_config['b_tta']
        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_resource = self.set_dir_get_path(self.p_project, 'resources')
        self.p_output = self.set_dir_get_path(self.p_project, 'output')
        self.p_test = self.set_dir_get_path(self.p_project, 'Test')
        self.p_aug = self.set_dir_get_path(self.p_output, 'output_text_aug')
        self.p_classifier = self.set_dir_get_path(self.p_output, 'output_classifier')
        self.p_tta = self.set_dir_get_path(self.p_aug, 'TTA')

        self.p_onehot = self.p_aug + r'\df_onehot.csv'
        self.p_tfidf = self.p_aug + r'\x_sectors.csv'
        self.p_sectors = self.p_aug + r'\sectors.csv'
        self.p_imputed = self.p_classifier + r'\df_imputed.csv'
        self.p_standard = self.p_classifier + r'\df_standard.csv'
        self.p_data_final = self.p_classifier + r'\df_final.csv'
        self.p_filtered = self.p_classifier + r'\df_filtered.csv'
        self.p_data_pdf = self.p_resource + r'\allpdf'
        self.p_data_ct = self.p_resource + r'\allct'
        self.p_parser = self.p_output + r'\output_parser'
        self.p_features = self.p_parser + r'\df_features.csv'
        self.p_features_merged = self.p_parser + r'\df_features_merged.csv'
        self.p_text = self.p_output + r'\output_text_aug\df_text.csv'
        self.p_tta = self.p_aug + r'\TTA'

        if self.b_vpn:
            self.p_project = self.set_vpn_dir(self.p_project)
            self.p_resource = self.set_vpn_dir(self.p_resource)
            self.p_output = self.set_vpn_dir(self.p_output)
            self.p_aug = self.set_vpn_dir(self.p_aug)
            self.p_onehot = self.set_vpn_dir(self.p_onehot)
            self.p_tfidf = self.set_vpn_dir(self.p_tfidf)
            self.p_sectors = self.set_vpn_dir(self.p_sectors)
            self.p_imputed = self.set_vpn_dir(self.p_imputed)
            self.p_standard = self.set_vpn_dir(self.p_standard)
            self.p_data_final = self.set_vpn_dir(self.p_data_final)
            self.p_filtered = self.set_vpn_dir(self.p_filtered)
            self.p_data_pdf = self.set_vpn_dir(self.p_data_pdf)
            self.p_data_ct = self.set_vpn_dir(self.p_data_ct)
            self.p_parser = self.set_vpn_dir(self.p_parser)
            self.p_features = self.set_vpn_dir(self.p_features)
            self.p_features_merged = self.set_vpn_dir(self.p_features_merged)
            self.p_text = self.set_vpn_dir(self.p_text)
            self.p_tta = self.set_vpn_dir(self.p_tta)

        self.l_target_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.l_target_cols_merged = ['A+B', 'C+D', 'E+F', 'G', 'H+I', 'J', 'K', 'L', 'M', 'N']
        self.l_tfdf_cols_features = [
            'ArrivalReason',
            'BreastArmPit', 'Chest', 'Lung', 'ChestLung', 'HeadNeck', 'SkeletonTissue', 'StomachPelvis',
            'Summary'
        ]
        self.l_tta_types = ['backtrans', 'synonymheb', 'w2v']
        self._report = Report(self)

    def init_data_structures(self):
        """
        function inits data structures
        """
        self.df_data = pd.DataFrame(columns=self.l_target_cols).astype(int)

    def merge_targets(self):
        """
        function merges targets
        """
        if not self.check_file_exists(self.p_features):
            print('Features file not found.')
        else:
            df_data = pd.read_csv(self.p_features)
            df_data['A+B'] = df_data['A'] + df_data['B']
            df_data['C+D'] = df_data['C'] + df_data['D']
            df_data['E+F'] = df_data['E'] + df_data['F']
            df_data['H+I'] = df_data['H'] + df_data['I']

            l_drop = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I']
            df_data.drop(l_drop, axis=1, inplace=True)

            l_move = ['G', 'J', 'K', 'L', 'M', 'N']
            df_data = self.reorder_columns(df_data, l_move)

            for curr_target in self.l_target_cols_merged:
                found = df_data[df_data[curr_target] > 1]
                if found.shape[0] >= 1:
                    df_data[curr_target] = df_data[curr_target].replace(2, 1)

            df_data[self.l_target_cols_merged] = df_data[self.l_target_cols_merged].astype(int)  # format validation
            self.set_df_to_csv(df_data, 'df_features_merged', self.p_parser, s_na='NA', b_append=True, b_header=True)

    @staticmethod
    def reorder_columns(df_curr, l_move):
        """
        function reorders columns of a given dataframe
        :param df_curr input dataframe
        :param l_move list of column order
        """
        for col in l_move:
            df_pop = df_curr.pop(col)
            df_curr = pd.concat([df_curr, df_pop], 1)
        return df_curr

    @staticmethod
    def init_globals(process_counter):
        """
        function initializes shared variable for the process pool
        :param process_counter amount of processes being used
        """
        global i_process_counter
        global i_files_counter
        i_process_counter = process_counter
        i_files_counter = 0

    def set_file_list(self, b_parse, path=None):
        """
        function loads file paths to list
        :param b_parse boolean flag for Parser class usage
        :param path default or specific
        """
        l_files = []
        if path is None:
            path = self.p_data_pdf
        if 'output_text_aug' in path:
            b_done = False
            for root, dirs, files in os.walk(path):
                if not b_done:
                    for file in files:
                        curr_file_path = os.path.join(root, file)
                        l_files.append(curr_file_path)
                b_done = True
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    curr_file_path = os.path.join(root, file)
                    if b_parse:
                        self.l_files.append(curr_file_path)
                    else:
                        if 'df' not in curr_file_path:
                            l_files.append(curr_file_path)
        if b_parse:
            if self.i_sample is not None:
                self.l_files = self.l_files[:self.i_sample]
        else:
            return l_files

    @staticmethod
    def set_df_columns_order(this_df, this_list):
        """
        function sets column of order of a datafrane
        :param this_df input dataframe
        :param this_list order of columns by list
        """
        return this_df[this_list]

    def set_df_to_csv(self, df, filename, path, s_na='NA', b_append=True, b_header=True):
        """
        function appends new data onto a dataframe and saves on disk
        :param df dataframe
        :param filename of the child file
        :param path of parent directory
        :param s_na value to replace na's
        :param b_append boolean to append to dataframe
        :param b_header to write header or not
        :return updated dataframe saved in resources directory path
        """
        p_write = path + '\\' + filename + '.csv'
        if self.b_vpn:
            p_write = path + '/' + filename + '.csv'
        if b_append:
            df.to_csv(path_or_buf=p_write, mode='a', index=False, na_rep=s_na, header=b_header, encoding='utf-8-sig')
        else:
            df.to_csv(path_or_buf=p_write, mode='w', index=False, na_rep=s_na, header=b_header, encoding='utf-8-sig')

    def validate_path(self, parent_dir, child_filename, extension):
        """
        function configures path depending on VPN usage
        :param parent_dir parent directory path
        :param child_filename  child directory path
        :param extension of file
        """
        p_curr = parent_dir + '\\' + child_filename + '.' + extension
        if self.b_vpn:
            p_curr = parent_dir + '/' + child_filename + '.' + extension
        return p_curr

    def get_csv_to_df_header_only(self, curr_file, nrows=1):
        """
        function reads CSV header only
        :param curr_file CSV file
        :param nrows number of rows to view
        """
        l_headers = pd.read_csv(curr_file, nrows).columns.tolist()
        i_cols = len(l_headers)
        s_file = self.get_filename(curr_file)
        print(f'File: {s_file}, Columns: {i_cols}')
        return l_headers

    @staticmethod
    def get_csv_to_df(curr_file):
        """
        function transforms CSV to DataFrame
        :param curr_file CSV file
        """
        df = pd.read_csv(curr_file, skip_blank_lines=True)
        curr_df = df.copy()
        del df
        return curr_df

    @staticmethod
    def set_vpn_dir(p_curr):
        """
        function configures VPN paths
        :param p_curr path to convert to VPN
        """
        s_prefix = '/home/edoli'
        if s_prefix not in p_curr:
            s_parent_dir = '/PETCT-Prediction'
            p_curr = p_curr.replace('\\', '/')
            l_delimiter = p_curr.split('/')
            s_delimiter = l_delimiter[2]
            l_curr = p_curr.split(s_delimiter)
            if len(l_curr) > 1 and l_curr[1] != '':
                p_new = s_prefix + s_parent_dir + l_curr[1]
            else:
                p_new = s_prefix + s_parent_dir
            return p_new
        else:
            if '\\' in p_curr:
                p_curr = p_curr.replace('\\', '/')
            return p_curr

    def set_dict_to_csv(self, d, filename, path):
        """
        function transforms DICT to CSV
        :param d DICT
        :param filename
        :param path
        :return CSV
        """
        p_write = path + '\\' + filename + '.csv'
        if self.b_vpn:
            p_write = path + '/' + filename + '.csv'
        try:
            with open(p_write, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in d.items():
                    writer.writerow([key, value])
        except IOError:
            print('I/O error')

    @staticmethod
    def set_dict_of_dicts_to_csv(d, path):
        """
        function writes dict of dict to CSV
        :param d dict
        :param path to write
        """
        try:
            with open(path, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                for i_key, i_value in d.items():
                    writer.writerow([i_key])
                    for j_key, j_value in i_value.items():
                        writer.writerow([j_key, j_value])
        except IOError:
            print("I/O error")

    def set_pickle_to_csv(self, path, filename):
        """
        function transforms PICKLE to CSV
        :param path of pickle file
        :param filename
        """
        curr_path = path + '\\' + filename
        if self.b_vpn:
            curr_path = path + '/' + filename
        d_data = self.get_pickle(curr_path, filename)
        self.set_dict_to_csv(d_data, filename, self.p_output)

    def set_pickle(self, d, path, filename):
        """
        function writes pickle files
        :param path of pickle file
        :param filename
        """
        p_write = path + '\\' + filename + '.pkl'
        if self.b_vpn:
            p_write = path + '/' + filename + '.pkl'
            if '\\' in p_write:
                p_write = p_write.replace('\\', '/')
            if '//' in p_write:
                p_write = p_write.replace('//', '/')
        with open(p_write, 'wb') as output:
            pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)

    def get_pickle(self, path, filename):
        """
        function reads pickle files
        :param path of pickle file
        :param filename
        """
        p_read = path + '\\' + filename + '.pkl'
        if self.b_vpn:
            p_read = path + '/' + filename + '.pkl'
            if '\\' in p_read:
                p_read = p_read.replace('\\', '/')
            if '//' in p_read:
                p_read = p_read.replace('//', '/')
        with open(p_read, 'rb') as curr_input:
            return pickle.load(curr_input)

    @staticmethod
    def set_dir(path):
        """
        function creates a directory only
        :param path of directory
        """
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    def set_dir_get_path(self, path, folder_name):
        """
        function creates a directory and returns its path
        :param path of directory
        :param folder_name requested name
        """
        p_new_dir = path + '\\' + folder_name
        if self.b_vpn:
            p_new_dir = path + '/' + folder_name
        if not os.path.exists(p_new_dir):
            os.makedirs(p_new_dir)
            return p_new_dir
        return p_new_dir

    def get_filename(self, path):
        """
        function returns filename
        :param path path to file
        :return filename
        """
        l_substring = path.split('\\')
        if self.b_vpn:
            l_substring = path.split('/')
        l_subtitle = l_substring[len(l_substring) - 1].split('.')
        return l_subtitle[0]

    @staticmethod
    def check_file_exists(p_file):
        """
        function checks if a file exists
        :param p_file path to directory
        :return boolean
        """
        return os.path.exists(p_file)

    @staticmethod
    def get_csv_to_dict(path):
        """
        function transforms CSV to DICT
        :param path of CSV
        :return DICT
        """
        with open(path, mode='r') as infile:
            reader = csv.reader(infile)
            d_new = {rows[0]: rows[1] for rows in reader}
        return d_new

    @staticmethod
    def append_row_to_df(df, new_row):
        """
        function adds row in end of dataframe object
        :param df dataframe
        :param new_row new values to be inserted
        """
        value = new_row.values[0]
        cols = df.columns
        if len(cols) > 1:
            last_col = df.iloc[:, -1:].columns.values[0]
            null_rows = df[~df[last_col].isnull()]
            if null_rows.empty:
                df = df.drop([last_col], axis=1)
        df.loc[-1] = [value]
        df.index = df.index + 1
        df = df.sort_index()
        df = df.rename(columns={value: 'key'})
        first_value = df['key'].iloc[0]
        if '\'' in first_value:
            df['key'] = df['key'].str.replace('\'', '')
        return df

    def get_stopwords(self):
        """
        function reads stopwords file
        """
        return self._parser.get_stopwords()

    @staticmethod
    def crop(curr_file):
        """
        function crops PDF format to TXT format
        :param curr_file current file being handled
        """
        raw_text = parser.from_file(curr_file)
        return raw_text['content']

    def get_file_apply(self, curr_file, _parser):
        """
        function serves as mutex for multi-processing
        :param curr_file current file being handled
        :param _parser reference to Parser
        :return merged stopwords file
        """
        global i_files_counter
        with i_files_counter.get_lock():
            i_files_counter.value += 1
        self._parser.run(self.crop(curr_file))

    def run_parser_processing_apply(self, p_file):
        """
        function assigns multi-processes tasks to allocated files
        :param p_file curr file to be assigned
        """
        global i_process_counter
        _parser = None
        p_name = "#NUM_" + str(i_process_counter.value)
        with i_process_counter.get_lock():
            i_process_counter.value += 1
        f_start = time.time()
        _parser = Parser(self, self.df_data)
        self.get_file_apply(p_file, _parser)
        f_end = time.time()
        print(p_name + 'Time: ' + str(f_end - f_start))

    def run_parser_processing(self):
        """
        function runs Parser class pipeline in multi-process
        """
        global i_process_counter
        i_process_counter = multiprocessing.Value('i', 0)
        i_files_count = len(self.l_files)
        print('Beginning processing with ' + str(i_files_count) + ' files.')
        pool = multiprocessing.Pool(processes=2, initializer=self.init_globals, initargs=(i_process_counter,))
        i = pool.map_async(self.run_parser_processing_apply, self.l_files, chunksize=1)
        i.wait()

    def run_parser(self):
        """
        function runs Parser class pipeline
        """
        tqdm.pandas()
        self._parser = Parser(self, self.df_data, len(self.l_files))
        for i, curr_file in tqdm(enumerate(self.l_files)):
            self._parser.run(i, self.crop(curr_file))  # runs tokenizing
        self._parser.save_file()
        print('Parser Done.')

    def run_preprocess(self):
        """
        function runs pre-processing pipelines
        """
        if not self.check_file_exists(self.p_features):
            print('Features file not found.')
        else:
            df_data = pd.read_csv(self.p_features)
            self._report.unique_count(df_data, 'CaseID')  # removes duplicates with even timestamps
            self.merge_targets()  # merging classes

    def run_text_aug(self):
        """
        function runs TextAug class pipeline
        """

        # l_exp_sectors = [10, 20, 100, 500, 1000, 10000, None]
        # l_exp_settings = [5, 10, 15, 20, 25, 50, None]
        # for i in range(len(l_exp_sectors)):
        #     i_features_sectors = l_exp_sectors[i]
        #     i_features_settings = l_exp_settings[i]

        # i_features_sectors, i_features_settings = None, None
        i_features_sectors, i_features_settings = 50000, 20
        # i_features_sectors, i_features_settings = 10000, 50

        _aug = TextAug(self)

        if not self.check_file_exists(self.p_sectors):  # creates sector file
            df_onehot = pd.read_csv(self.p_features_merged)
            _aug.merge_to_one_col(df_onehot, 'sectors')

        if not self.check_file_exists(self.p_text):  # word2vec corpus of sectors
            df_sectors = pd.read_csv(self.p_sectors)
            _aug.merge_to_one_cell(df_sectors, 'df_text')

        if self.b_tta:
            # _aug.convert_bin_to_txt()  # converts binary files to text files
            # _aug.word2vec_train()  # trains and saves a word2vec model
            _aug.generate_aug()  # generates augmentations

        print('Augmentations Done.')

    def run_classifier(self):
        """
        function runs Classifier class pipeline
        """
        _classifier = Classifier(self, self.b_tta)
        _classifier.run()

    def get_dimensions(self):
        """
        function returns shapes of input files
        """
        l_files = ['x_sectors', 'x_synonymheb', 'x_backtrans', 'x_w2v']
        for p_file in l_files:
            p_curr = self.validate_path(self.p_aug, p_file, 'csv')
            filename = self.get_filename(p_curr)

            cmd = 'wc -l ' + p_curr  # (v1)
            rows = int(subprocess.check_output(cmd, shell=True).split()[0]) - 1
            l_headers = pd.read_csv(p_curr, nrows=1).columns.tolist()
            cols = len(l_headers)
            print(f'File: {filename}, \n Rows: {int(rows)}, \n Columns: {int(cols)}')

            # df = pd.read_csv(p_curr)  # (v2)
            # # df = pd.read_csv(p_curr, usecols=[i for i in range(5)])
            # rows = df.shape[0]
            # cols = df.shape[1]
            # del df
            # print(f'File: {filename}, \n Rows: {int(rows)}, \n Columns: {int(cols)}')

    def merge_stopwords(self, file1, file2):
        """
        function merges stopword files
        :param file1
        :param file2
        :return merged stopwords file
        """
        df_sw1 = self.get_csv_to_df(self.p_resource + file1)
        new_row1 = df_sw1.columns
        df_sw1 = self.append_row_to_df(df_sw1, new_row1)
        df_sw2 = self.get_csv_to_df(self.p_resource + file2)
        new_row2 = df_sw2.columns
        df_sw2 = self.append_row_to_df(df_sw2, new_row2)
        df_merged = df_sw1.merge(df_sw2, on='key').drop_duplicates()
        df_merged = df_merged.reset_index(drop=True)
        null_rows = df_merged[~df_merged['key'].isnull()]
        if ~null_rows.empty:
            null_value = null_rows.iloc[0]
            print(null_value)
        self.set_df_to_csv(df_merged, 'merged_stopwords', self.p_resource, s_na='NA', b_append=True, b_header=True)

    def run_report(self):
        """
        function runs Report class pipeline
        """
        if not self.check_file_exists(self.p_features_merged):
            print('Merged features file not found.')
        else:
            df_data = pd.read_csv(self.p_features_merged)
            self._report.data_percentage(df_data, False)
            self._report.class_count(df_data)
            self._report.get_ratios()
            self._report.analysis(df_data)
            self.get_dimensions()
  
    def run(self):  # MAIN FLOW
        """
        main function: runs all phases of pre-processing, training and testing of the models
        """
        b_parse = True
        i_run_start = time.time()
        self.set_file_list(b_parse)
        self.init_data_structures()
        print(f'Running on {len(self.l_files)} files...')
        if self.b_processing:
            self.run_parser_processing()
        else:
            # self.run_parser()
            # self.run_preprocess()
            # self.run_report()
            # self.run_text_aug()
            self.run_classifier()
        i_run_end = time.time()
        run_time = i_run_end - i_run_start
        print('Finished in: %.2f hours (%.2f minutes).' % (run_time/60/60, run_time/60))
