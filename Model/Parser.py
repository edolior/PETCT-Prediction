import numpy as np
import datetime as dt
import pandas as pd
import re


# ---------------------------------------------------------------------------------------
# Parser Class:
#
# Tokenizes Files
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Parser:

    _model = None

    def __init__(self, r_model, df_data, i_records):
        self._model = r_model
        self.df_data = df_data
        self.i_records = i_records

        self.p_stopwords = self._model.p_resource + r'\stopwords.csv'
        self.p_output = self._model.set_dir_get_path(self._model.p_output, 'output_parser')

        if self._model.b_vpn:
            self.p_stopwords = self._model.set_vpn_dir(self.p_stopwords)
            self.p_output = self._model.set_vpn_dir(self.p_output)

        l_punc = ["\"", '\"', ',', '"', '|', '?', '-', '_', '*', '`', '/', '@', ';', "'", '[', ']', '(', ')',
                       '{', '}', '<', '>', '~', '^', '?', '&', '!', '=', '+', '#', '$']
        l_punc_needed = ['%', ':', '.', ' ']
        self.period = '.'
        self.percentage = '%'
        self.space = ' '
        self.colon = ':'
        self.comma = ','
        self.hyphen = '-'
        self.d_stopwords = self.set_stopwords()
        self.d_punc = self.set_puncwords(l_punc)
        self.d_punc_needed = self.set_puncwords(l_punc_needed)
        self.d_map_format = {self.period: None, self.colon: None, self.comma: None, self.hyphen: None}

        self.d_features, self.d_features_key, self.d_features_histogram = {}, {}, {}
        self.s_text, self.s_case_id = '', ''
        self.s_target = 'CATDB'
        self.init_features_dict()
        self.init_features_histogram()

        l_feature_cols = ['CaseID', 'Age', 'Gender', 'HealthCare', 'Unit', 'Timestamp', 'ServiceHistory', 'Service',
                          'VariableAmount', 'VariableLocation', 'VariableRange', 'GlucoseLevel', 'TestStartTime']
        self.l_target_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.d_target_cols = {'A': '', 'B': '', 'C': '', 'D': '', 'E': '', 'F': '', 'G': '', 'H': '', 'I': '', 'J': '',
                              'K': '', 'L': '', 'M': '', 'N': ''}
        self.df_data = self.df_data.reindex(columns=self.df_data.columns.tolist() + l_feature_cols)
        self.d_fixed = dict()
        self.d_invalid = dict()

    def normalize_text(self, text):
        """
        Returns string without bad formats and punctuations.
            Args: text (str): string input of a sentence
        """
        normalized_text = text.translate(str.maketrans(self.d_punc))
        return normalized_text

    def get_stopwords(self):
        with open(self.p_stopwords, 'r', encoding='utf8') as file:
            data = file.read().replace('\n', self.space)
        return data

    def set_stopwords(self):
        d_stopwords = {}
        with open(self.p_stopwords, 'r', encoding='utf8') as file:
            data = file.read().replace('\n', self.space)
        l_stopwords = data.split()
        for word in l_stopwords:
            d_stopwords[word] = ''
        del l_stopwords
        return d_stopwords

    def init_features_dict(self):
        self.d_features_key = {
            'סיבת ההפניה': 'ArrivalReason',
            'ראש צוואר': 'HeadNeck',
            'שדיים שחי': 'BreastArmPit',
            'ריאות': 'Lung',
            'חזה': 'Chest',
            'חזה ריאות': 'ChestLung',
            'בטן ואגן': 'StomachPelvis',
            'שלד שריר ורקמה רכה': 'SkeletonTissue',
            'סיכום': 'Summary',
        }

    def init_features_histogram(self):
        self.d_features_histogram = {
            'ArrivalReason': 0,
            'HeadNeck': 0,
            'BreastArmPit': 0,
            'ChestLung': 0,
            'Lung': 0,
            'Chest': 0,
            'StomachPelvis': 0,
            'SkeletonTissue': 0,
            'Summary': 0,
        }

    def set_features_histogram(self, other_feature):
        try:
            self.d_features_histogram[other_feature] += 1
        except KeyError as e:
            print(f'Error adding feature: {other_feature}, details: {e}')

    def set_features_key(self, other_feature, b_search_only):
        # deals with invalid typing formats: e.g., head&neck, mCi8.64, random placement of symbols '#'
        if 'CT' in other_feature:
            other_feature = other_feature.replace('CT', '')
        other_feature = other_feature.strip()
        if 'חזה' in other_feature and 'ריאות' in other_feature:
            other_feature = 'חזה ריאות'
            return self.d_features_key['חזה ריאות'], True
        elif 'חזה' in other_feature and 'ריאות' not in other_feature:
            other_feature = 'חזה'
        elif 'ריאות' in other_feature and 'חזה' not in other_feature:
            other_feature = 'ריאות'
        elif 'שערי' in other_feature or 'ושערי' in other_feature:
            other_feature = 'חזה ריאות'
        elif 'סיבת' in other_feature:
            other_feature = 'סיבת'
        elif 'ראש' in other_feature:
            other_feature = 'ראש'
        elif 'שדיים' in other_feature and 'שחי' in other_feature:
            other_feature = 'שדיים שחי'
        elif 'בטן ואגן' in other_feature:
            other_feature = 'בטן ואגן'
        elif 'סיכום' in other_feature:
            other_feature = 'סיכום'
        elif 'שלד שריר' in other_feature:
            other_feature = 'שלד שריר'
        for key, value in self.d_features_key.items():
            if other_feature == 'חזה':
                return 'Chest', True
            if other_feature in key:
                return value, True
        if self.is_not_digit(other_feature):
            if not b_search_only:
                print(f'Found new feature named: {other_feature} in Case ID: {self.s_case_id}')
                self.d_features_key.update({other_feature: 'new feature'})
        return other_feature, False

    def set_puncwords(self, l_curr_puncs):
        d_puncwords = {}
        if l_curr_puncs is not None:
            for word in l_curr_puncs:
                # d_puncwords[word] = ''
                d_puncwords[word] = None
            del l_curr_puncs
        return d_puncwords

    def is_not_digit(self, value):
        # function returns true if value is a term
        if len(value) > 1:
            for element in value:
                if element in self.d_punc_needed:
                    continue
                flag = element.isdigit()
                if not flag:
                    return True
            return False
        else:
            flag = value.isdigit()
            if not flag:
                return True
            else:
                return False

    def is_not_digit_has_spaces(self, value):
        for char in value:
            if self.is_not_digit(char) and char != self.space:
                return False
        return True

    def is_digit(self, value):
        # function returns true if the input value is a digit
        i_correct = 0
        if not isinstance(value, str):
            value = str(value)
        for element in value:
            if element in self.d_punc_needed:
                continue
            if element.isdigit():
                i_correct += 1
            else:
                if i_correct == 0:  # some functions need cells with digits & letters and filters them afterwards
                    if not value[-1].isdigit():
                        return False
                    else:
                        return True
                else:
                    return True
        return True

    @staticmethod
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        return start

    def filter_to_numbers(self, l_other):
        return [x for x in l_other if x is not self.is_not_digit(x)]

    def find_next_digit_term(self, other_list):
        for i_term in range(len(other_list)):
            if not self.is_not_digit(other_list[i_term]):
                return i_term

    def remove_missing_data(self, df_data):
        # function finds missing labels and removes them
        length_before = df_data.shape[0]
        df_data.dropna(subset=self.l_target_cols, inplace=True)
        length_after = df_data.shape[0]
        print(f'Found {length_before - length_after} records of missing labels.')
        self._model.set_df_to_csv(df_data, 'df_features', self.p_output, s_na='NA', b_append=True, b_header=True)

    def get_element(self, s_other, b_str):
        s_other = s_other.strip()
        s_other = s_other.replace('  ', '')
        if self.colon in s_other:
            s_other = s_other.replace(self.colon, '')
        i_length_term = len(s_other)
        if 'Ga' in s_other:
            return s_other
        if s_other[0].isdigit():
            i_digit = i_length_term - 1
            while i_digit > 0:
                if s_other[i_digit].isdigit():
                    if b_str:
                        return s_other[i_digit+1:].strip()
                    else:
                        return s_other[:i_digit].strip()
                else:
                    i_digit -= 1
        else:
            i_digit = 0
            while i_digit < i_length_term:
                if s_other[i_digit].isdigit():
                    if b_str:
                        return s_other[:i_digit].strip()
                    else:
                        return s_other[i_digit:].strip()
                else:
                    i_digit += 1
        return np.nan

    def filter_term(self, other_term):
        if len(other_term) > 1 and 'ממצאי בדיקה' not in other_term and 'גרסה' not in other_term and 'CATDB' not in other_term and self.s_case_id not in other_term and ' דר ' not in other_term:
            if self.space not in other_term and other_term.isnumeric():
                return False
            if self.space not in other_term and 'CT' in other_term:
                return False
            if 'המשך' in other_term and len(other_term) == 4:
                return False
            elif 'עמוד' in other_term and 'שדרה' not in other_term:
                return False
            elif 'שעת' in other_term and 'פענוח' in other_term:
                return False
            elif 'בכבוד רב' in other_term:
                return False
            return True
        else:
            return False

    def split_by_paragraph(self, s_other):
        l_this = s_other.split('\n')
        l_other = []
        i = 2
        j = 0
        while i < len(l_this) - 3:
            curr_term = l_this[i].strip()
            if self.filter_term(curr_term):
                b_resume = True
                while j > 0 and self.colon in l_other[j - 1] and b_resume:
                    if self.colon not in curr_term:  # if there are more than 2 lines ahead but to merge
                        l_other[j - 1] = l_other[j - 1] + self.space + curr_term
                        i += 1
                        curr_term = l_this[i].strip()
                        while not self.filter_term(curr_term):
                            i += 1
                            curr_term = l_this[i].strip()
                    elif self.colon in curr_term and 'כמות הסמן' not in curr_term:  # if there's a ':' in the second page but is not a features, merge with prev
                        i_candidate_feature = curr_term.find(self.colon)
                        candidate_feature = curr_term[:i_candidate_feature]
                        _, flag_found = self.set_features_key(candidate_feature, True)
                        if not flag_found and self.is_not_digit(candidate_feature):  # merge with prev
                            l_other[j - 1] = l_other[j - 1] + self.space + curr_term
                            i += 1
                            curr_term = l_this[i].strip()
                            while not self.filter_term(curr_term):
                                i += 1
                                curr_term = l_this[i].strip()
                        else:
                            b_resume = False
                    else:
                        b_resume = False
                # do while
                b_resume = True
                next_term = l_this[i+1].strip()
                next_size = len(next_term)
                next_next_term = l_this[i+2].strip()
                next_next_size = len(next_next_term)  # each 2 empty lines is a new paragraph
                if (next_size > 2 or next_next_size > 2) and b_resume:  # if there aren't 2 empty lines ahead
                    while (next_size > 2 or next_next_size > 2) and b_resume:  # if there aren't 2 empty lines ahead
                        if next_size < 2:
                            del l_this[i+1]
                        else:
                            if self.filter_term(curr_term):
                                try:
                                    l_other[j] = l_other[j] + self.space + curr_term
                                except IndexError:
                                    l_other.insert(j, curr_term)
                            i += 1
                            curr_term = l_this[i].strip()
                        if i > len(l_this) - 3:
                            b_resume = False
                        else:
                            next_term = l_this[i + 1].strip()
                            next_size = len(next_term)
                            next_next_term = l_this[i + 2].strip()
                            next_next_size = len(next_next_term)
                try:
                    if self.filter_term(curr_term):
                        l_other[j] = l_other[j] + self.space + curr_term
                except IndexError:
                    l_other.insert(j, curr_term)
                j += 1
            i += 1
        if l_other[len(l_other)-2] == 'סיכום:':
            l_other[len(l_other)-2] += l_other[len(l_other)-1]
            del l_other[len(l_other)-1]
        if 'סיכום:' not in l_other[len(l_other)-1]:
            del l_other[len(l_other)-1]
        return l_other

    @staticmethod
    def split_text_by_keys(curr_txt, l_key, r_key):
        return curr_txt.split(l_key)[1].split(r_key)[0].strip()

    def get_digits(self, other_list):
        # function receives list of string (terms / digits)
        # function returns new list with the term's digits only
        this_list = []
        for term in other_list:
            if term != '':
                if self.is_digit(term):
                    this_list.append(term)
        return this_list

    def filter_characters(self, other_list):
        # function removes letters and returns the value with digits only
        if isinstance(other_list, str) and len(other_list) > 1:
            new_term = ''
            for i_char in range(len(other_list)):
                char = other_list[i_char]
                if char.isdigit() or char in self.d_punc_needed:
                    new_term += char
            return new_term
        elif len(other_list) > 1:
            for i_term in range(len(other_list)):
                s_term = other_list[i_term]
                if s_term != '':
                    for i_char in range(len(s_term)):
                        char = s_term[i_char]
                        if self.is_not_digit(char) and char not in self.d_punc_needed:
                            other_list[i_term] = other_list[i_term].replace(char, '')
            return other_list

    @staticmethod
    def index_next_empty(l_other, i):
        for j in range(i+1, len(l_other)):
            if len(l_other[j]) < 3:
                return j
        return -1

    def save_file(self):
        print(f'Number of valid files: {len(self.df_data)} out of {self.i_records}.')
        self.df_data[self._model.l_tfdf_cols_features] = self.df_data[self._model.l_tfdf_cols_features].replace(np.nan, '', regex=True)
        self._model.set_df_to_csv(self.df_data, 'df_features', self.p_output, s_na='NA', b_append=True, b_header=True)
        self._model.set_dict_to_csv(self.d_features_histogram, 'd_histogram', self.p_output)
        self._model.set_pickle(self.d_invalid, self.p_output, 'd_invalid')
        self._model.set_pickle(self.d_fixed, self.p_output, 'd_fixed')

    def get_case_id(self, s_demographics):
        i_case_id = np.nan
        try:
            self.s_case_id = ''.join(
                filter(str.isdigit, (s_demographics.split("מקרה", 1)[1]).split("גורם")[0].strip().split(self.space)[0]))
            i_case_id = int(self.s_case_id)
        except ValueError as ve:
            print(f'No CaseID found: {ve}')
            self.update_report('case id')
        return i_case_id

    def get_gender(self, s_demographics):
        i_gender = np.nan
        try:
            s_gender = s_demographics.split("כתובת", 1)[1].split("מין")[0].strip().replace(self.colon, '')
            if len(s_gender) > 1:
                s_gender = s_gender[-1]
            i_gender = 1 if s_gender == 'ז' else 0
        except ValueError as ve:
            print(f'No gender found: {ve} for Case ID: {self.s_case_id}')
            self.update_report('gender')
        return i_gender

    def get_age(self, s_demographics):
        i_age = np.nan
        try:
            i_age = (s_demographics.split("גיל", 1)[1]).split("מקרה")[0].strip().split(self.space)[0].replace(self.colon, '')
            i_age = int(i_age)
        except ValueError:
            try:
                s_original = (s_demographics.split("גיל", 1)[1]).split("מקרה")[0]
                i_find = s_original.find('גיל')
                i_age = s_original[i_find+4:i_find+7].strip()
                i_age = int(i_age)
            except ValueError as ve:
                print(f'No age found: {ve} for Case ID: {self.s_case_id}')
                self.update_report('age')
        if not self.is_digit(i_age):
            i_age = np.nan
        if self.s_case_id == '31287822':
            i_age = 69
            self.validate_value('age')
        if self.s_case_id == '31222606':
            i_age = 68
            self.validate_value('age')
        return i_age

    def get_health_care(self, s_demographics):
        s_health_care = ''
        try:
            s_health_care = self.get_element((s_demographics.split("מקרה", 1)[1]).split("גורם")[0], True)
        except IndexError as ie:
            print(f'No health care found: {ie} for Case ID: {self.s_case_id}')
            self.update_report('health care')
        if 'תולדות מחלה' in s_health_care:
            s_health_care = ''
        return s_health_care

    def get_unit(self, s_demographics):
        s_unit = ''
        try:
            s_unit = (s_demographics.split("מפנה", 1)[1]).split("מזמינה")[0].strip().replace(self.colon, '')
        except IndexError as ie:
            try:
                s_unit = self.get_element((s_demographics.split("מקרה", 1)[1]).split("מזמינה")[0].strip().replace(self.colon, ''), True)
                if 'יחידה ארגונית' in s_unit:
                    s_unit = s_unit.replace('יחידה ארגונית', '').strip()
            except IndexError as ie:
                print(f'No unit found: {ie} for Case ID: {self.s_case_id}')
                self.update_report('unit')
        return s_unit

    def get_service(self, s_demographics, s_health_care):
        s_service = ''
        try:
            if s_health_care == '':
                s_service = (s_demographics.split("ביצוע", 2))[2]
            else:
                s_service = self.get_element((s_demographics.split("ביצוע", 2))[2], True)
        except IndexError:
            try:
                s_service = (s_demographics.split("תולדות מחלה", 1)[1])
            except IndexError as ie:
                print(f'No service found: {ie} for Case ID: {self.s_case_id}')
                self.update_report('service')
        if 'תיאור הבדיקה' in s_service:
            s_service = ''
            # print(f'תיאור הבדיקה found in {self.s_case_id}')
        if 'שם' in s_service:
            s_service = ''
        return s_service

    def get_timestamp(self, s_demographics, s_health_care):
        o_date_time = np.nan
        try:
            if s_health_care == '':
                l_raw_timestamp_original = (s_demographics.split("תאריך אישור ביצוע", 1)[1]).split("תולדות מחלה")[0].strip().split(self.space)
            else:
                l_raw_timestamp_original = (s_demographics.split("תאריך אישור ביצוע", 1)[1]).split("תיאור")[0].strip().split(self.space)

            l_raw_timestamp_filtered = l_raw_timestamp_original[-4:]

            if self.is_not_digit(l_raw_timestamp_filtered[0]):
                l_raw_timestamp_filtered = self.get_digits(l_raw_timestamp_original)
                l_raw_timestamp_filtered = self.filter_characters(l_raw_timestamp_filtered)

            try:
                curr_time = l_raw_timestamp_filtered[0]
                curr_date = l_raw_timestamp_filtered[1]
                day = curr_date[:2]
                month = curr_date[2:4]
                year = curr_date[4:]
                s_raw_timestamp = year + '-' + month + '-' + day + self.space + curr_time
                o_date_time = dt.datetime.strptime(s_raw_timestamp, '%Y-%m-%d %H:%M')
            except ValueError:
                try:
                    curr_time = l_raw_timestamp_filtered[-2]
                    curr_date = l_raw_timestamp_filtered[-1]
                    day = curr_date[:2]
                    month = curr_date[2:4]
                    year = curr_date[4:]
                    s_raw_timestamp = year + '-' + month + '-' + day + self.space + curr_time
                    o_date_time = dt.datetime.strptime(s_raw_timestamp, '%Y-%m-%d %H:%M')
                except IndexError:
                    try:
                        s_raw_timestamp = l_raw_timestamp_filtered[4] + '-' + l_raw_timestamp_filtered[3] + '-' + \
                                          l_raw_timestamp_filtered[2] + self.space + \
                                          l_raw_timestamp_filtered[1]
                        o_date_time = dt.datetime.strptime(s_raw_timestamp, '%Y-%m-%d %H:%M')
                    except IndexError as ie:
                        o_date_time = np.nan
                        print(f'Timestamp Error: {ie} for Case ID: {self.s_case_id}')
                        self.update_report('timestamp')
        except ValueError as ve:
            print(f'No timestamp found: {ve} for Case ID: {self.s_case_id}')
            self.update_report('timestamp')
        return o_date_time

    def check_delimiter(self, value):
        # function removes invalid formats from value
        if not pd.isna(value):
            value = value.strip()
            if len(value) > 0:
                if value[-1] in self.d_punc_needed:
                    value = value[:-1]
                if value[0] in self.d_punc_needed:
                    value = value[1:]
            return value

    def get_indicator_quantity(self, curr_txt):
        f_indicator_quantity = np.nan  # mCi - Milli Curie unit of radioactivity
        s_indicator_quantity = self.get_element(curr_txt.split('כמות הסמן')[1].split("אזור הזרקה")[0], False)
        if not self.is_digit(s_indicator_quantity):
            try:
                l_str = list(s_indicator_quantity.split(self.space))
                l_indicator_filtered = self.get_digits(l_str)
                s_indicator_quantity = l_indicator_filtered[0]
            except AttributeError:
                if self.s_case_id == '31230305':
                    s_indicator_quantity = '7.0'
                    self.validate_value('indicator quantity')
            try:
                s_indicator_quantity = self.check_delimiter(s_indicator_quantity)
            except TypeError:
                if self.s_case_id == '31214989':
                    f_indicator_quantity = np.nan
                    self.validate_value('indicator quantity')
        try:
            s_indicator_quantity = self.check_delimiter(s_indicator_quantity)
            if not pd.isna(s_indicator_quantity):
                f_indicator_quantity = float(s_indicator_quantity)
        except ValueError:
            try:
                s_indicator_quantity = self.filter_characters(s_indicator_quantity)
                s_indicator_quantity = self.check_delimiter(s_indicator_quantity)
                if len(s_indicator_quantity) > 10 or s_indicator_quantity.count(self.period) > 2:
                    l_values = s_indicator_quantity.split(self.space)
                    s_indicator_quantity = l_values[0]
                    s_indicator_quantity = self.check_delimiter(s_indicator_quantity)
                f_indicator_quantity = float(s_indicator_quantity)
            except (ValueError, IndexError) as ve:
                print(f'No indicator quantity found: {ve} for Case ID: {self.s_case_id}')
                self.update_report('indicator quantity')
        return f_indicator_quantity

    def get_injection_area(self, curr_txt):
        s_injection_area = ''
        try:
            s_injection_area = curr_txt.split("אזור הזרקה")[1].split('טווח הסריקה')[0].strip()
        except Exception:
            try:
                s_injection_area = curr_txt.split("אזור ההזרקה")[1].split('טווח הסריקה')[0].strip()
            except IndexError:
                try:
                    s_injection_area = curr_txt.split("אזור")[1].split('טווח הסריקה')[0].strip()
                except IndexError as ie:
                    print(f'No injection area found: {ie} for Case ID: {self.s_case_id}')
                    self.update_report('injection area')
        if s_injection_area != '' and len(s_injection_area) > 1:
            s_injection_area = self.check_delimiter(s_injection_area)
        if len(s_injection_area) == 1:
            s_injection_area = ''
        if self.s_case_id == '31220153':
            s_injection_area = 'מרפק ימין'
            self.validate_value('injection area')
        if self.s_case_id == '31221251':
            s_injection_area = 'יד שמאל'
            self.validate_value('injection area')
        if self.s_case_id == '31215838':
            s_injection_area = 'אמה ימין'
            self.validate_value('injection area')
        return s_injection_area

    def get_injection_range(self, curr_txt):
        s_range = ''
        if 'רמת גלוקוז בדם' in curr_txt:
            try:
                s_range = curr_txt.split('טווח הסריקה')[1].split('רמת גלוקוז בדם לפני ההזרקה')[0].strip()
            except IndexError:
                try:
                    s_range = curr_txt.split('טוח הסריקה')[1].split('רמת גלוקוז בדם לפני ההזרקה')[0].strip()
                except IndexError:
                    try:
                        s_range = curr_txt.split('הסריקה')[1].split('רמת גלוקוז בדם לפני ההזרקה')[0].strip()
                    except IndexError:
                        s_range = ''
        else:
            try:
                s_range = curr_txt.split('טווח הסריקה')[1].split('הבדיקה החלה')[0].strip()
            except Exception as e:
                print(f'No injection range found: {e} for Case ID: {self.s_case_id}')
                self.update_report('injection range')
        s_range = self.check_delimiter(s_range)
        if self.is_digit(s_range):
            s_range = s_range.split(self.period)[0]
        return s_range

    def get_test_settings(self, curr_txt):
        s_test_setting = ''
        try:
            s_test_setting = curr_txt.split('הזרקת הסמן')[1].split('ממצאי הבדיקה')[0].strip()
            if len(s_test_setting) == 1:
                s_test_setting = ''
        except (ValueError, IndexError) as error_missing:
            print(f'No test settings found: {error_missing} for Case ID: {self.s_case_id}')
            self.update_report('test settings')
        if s_test_setting == '':
            try:
                s_test_setting = 'ניתן' + self.space + curr_txt.split('ניתן')[1].split('הבדיקה')[0].strip()
            except (ValueError, IndexError) as error_missing:
                print(f'No test settings found: {error_missing} for Case ID: {self.s_case_id}')
                self.update_report('test settings')
        return s_test_setting

    def filter_list(self, other_value, other_list):
        l_new_values = other_list[1: -1]
        i = 0
        b_flag = False
        this_value = None
        while i < len(l_new_values) and not b_flag:
            curr_element = l_new_values[i]
            if curr_element != '' and self.is_digit(curr_element) and len(curr_element) > 1:
                this_value = curr_element
                b_flag = True
            i += 1
        if this_value is not None:
            return self.check_delimiter(this_value)
        else:
            return other_value

    def get_glucose_levels(self, curr_txt):
        i_glucose_level = np.nan  # mg%
        s_glucose_level_end = ''
        try:
            s_glucose_level = self.get_element(curr_txt.split('רמת גלוקוז בדם לפני ההזרקה')[1].split('הבדיקה החלה')[0],
                                               False)

            if pd.isna(s_glucose_level):
                s_glucose_level = self.get_element(curr_txt.split('mCi')[1].split('כמות הסמן')[0], False)

            if isinstance(s_glucose_level, str) and self.percentage in s_glucose_level:
                s_glucose_level = s_glucose_level.replace(self.percentage, '')

            if isinstance(s_glucose_level, str):
                if self.is_digit(s_glucose_level):
                    s_glucose_level = self.check_delimiter(s_glucose_level)
                    try:
                        i_glucose_level = int(s_glucose_level)
                    except ValueError:
                        try:
                            b_flag = False
                            s_glucose_level_end = self.get_element(curr_txt.split('רמת גלוקוז בדם לפני ההזרקה')[1].split('ניתן חומר')[0], False)
                            # s_glucose_level_end = self.get_element(curr_txt.split('ניתן חומר')[1].split('רמת גלוקוז בדם לפני ההזרקה')[0], False)
                            if not pd.isna(s_glucose_level_end) and s_glucose_level_end != '':
                                if 'דקות' in s_glucose_level_end and 'mg' not in s_glucose_level_end:
                                    s_glucose_level_end = ''
                                    b_flag = True
                                if not b_flag:
                                    s_glucose_level_end = self.filter_characters(s_glucose_level_end)
                                    s_glucose_level_end = self.check_delimiter(s_glucose_level_end).strip()
                                    if self.period in s_glucose_level_end:
                                        s_glucose_level_end = s_glucose_level_end.split(self.period)[0]
                                    if isinstance(s_glucose_level_end, str) and self.percentage in s_glucose_level_end:
                                        s_glucose_level_end = s_glucose_level_end.replace(self.percentage, '')
                                    i_glucose_level = int(s_glucose_level_end)
                            else:
                                s_glucose_level = self.filter_characters(s_glucose_level)
                                s_glucose_level = self.check_delimiter(s_glucose_level)
                                if len(s_glucose_level) > 10 or s_glucose_level.count(self.period) > 2:
                                    l_values = s_glucose_level.split(self.space)
                                    s_glucose_level = l_values[0]
                                    if self.period in s_glucose_level:
                                        s_glucose_level = l_values[-1]
                                    if len(s_glucose_level) == 1 or len(s_glucose_level) > 3:
                                        s_glucose_level = self.filter_list(s_glucose_level, l_values)
                        except IndexError:
                            s_glucose_level = self.filter_characters(s_glucose_level)
                            s_glucose_level = self.check_delimiter(s_glucose_level)
                            if len(s_glucose_level) > 10 or s_glucose_level.count(self.period) > 2:
                                l_values = s_glucose_level.split(self.space)
                                s_glucose_level = l_values[0]
                                if self.period in s_glucose_level:
                                    s_glucose_level = l_values[-1]
                                if len(s_glucose_level) == 1 or len(s_glucose_level) > 3:
                                    s_glucose_level = self.filter_list(s_glucose_level, l_values)
                        try:
                            if s_glucose_level_end == '':
                                i_glucose_level = int(s_glucose_level)
                        except ValueError:
                            l_values = s_glucose_level.split(self.space)
                            s_glucose_level = l_values[0]
                            if len(s_glucose_level) > 1:
                                s_glucose_level = self.check_delimiter(s_glucose_level)
                                i_glucose_level = int(s_glucose_level)
                                if self.s_case_id == '31249729':
                                    i_glucose_level = 218
                                    self.validate_value('glucose')
        except (IndexError, ValueError) as e:
            if self.s_case_id == '31292428':
                i_glucose_level = 101
                self.validate_value('glucose')
            if self.s_case_id == '31292445':
                i_glucose_level = 150
                self.validate_value('glucose')
            if self.s_case_id == '31291866':
                i_glucose_level = 125
                self.validate_value('glucose')
            if self.s_case_id == '31263515':
                i_glucose_level = 92
                self.validate_value('glucose')
            if self.s_case_id == '31262174':
                i_glucose_level = 91
                self.validate_value('glucose')
            if self.s_case_id == '31292121':
                i_glucose_level = 333
                self.validate_value('glucose')
            if self.s_case_id == '31292560':
                i_glucose_level = 106
                self.validate_value('glucose')
            if self.s_case_id == '31292175':
                i_glucose_level = 90
                self.validate_value('glucose')
            else:
                print(f'No glucose level found: {e} for Case ID: {self.s_case_id}')
                self.validate_value('glucose')
        return i_glucose_level

    def get_start_time(self, curr_txt):
        # function returns examination start time after injection in minutes
        i_start_time = np.nan
        s_start_time = ''
        try:
            s_start_time = curr_txt.split('הבדיקה החלה')[1].split('דקות')[0].strip()
            s_start_time = self.check_delimiter(s_start_time)
            if not pd.isna(s_start_time):
                if self.s_case_id == '31278972':
                    s_start_time = 60
                    self.validate_value('start time')
                i_start_time = int(s_start_time)
        except (IndexError, ValueError) as error_missing:
            if self.s_case_id == '31259896':
                i_start_time = 58
                self.validate_value('start time')
            l_start_time = s_start_time.split(self.space)
            if len(l_start_time) > 1:
                b_flag = False
                i = 0
                while i < len(l_start_time) and not b_flag:
                    element = l_start_time[i]
                    if self.is_digit(element):
                        s_start_time = element
                        i_start_time = int(s_start_time)
                        b_flag = True
                    else:
                        i += 1
                if self.s_case_id == '31220153':
                    i_start_time = np.nan
            else:
                print(f'No start time found: {error_missing} for Case ID: {self.s_case_id}')
                self.update_report('start time')
        if self.s_case_id == '31230965':
            i_start_time = 70
            self.validate_value('start time')
        if self.s_case_id == '31275777':
            i_start_time = 63
            self.validate_value('start time')
        if self.s_case_id == '31242230':
            i_start_time = 78
            self.validate_value('start time')
        if self.s_case_id == '31268621':
            i_start_time = 80
            self.validate_value('start time')
        return i_start_time

    def validate_value(self, curr_segment):
        if self.s_case_id not in self.d_fixed or len(self.d_fixed[self.s_case_id]) == 0:
            self.d_fixed[self.s_case_id] = list()
        l_curr = self.d_fixed[self.s_case_id]
        b_resume = True
        if len(l_curr) > 0:
            for curr_error in l_curr:
                if curr_error == curr_segment:
                    b_resume = False
        if b_resume:
            self.d_fixed[self.s_case_id].append(curr_segment + ' fix')

    def update_report(self, curr_segment):
        if self.s_case_id not in self.d_invalid or len(self.d_invalid[self.s_case_id]) == 0:
            self.d_invalid[self.s_case_id] = list()
        self.d_invalid[self.s_case_id].append(curr_segment)

    def filter_empty_classes(self):
        b_valid = True
        counter = 0
        for key, value in self.d_features.items():
            if key in self.d_target_cols:
                if value == 0:
                    counter += 1
        if counter == 14 or len(self.d_features) < 20:
            b_valid = False
            print(f'Invalid labels for: {self.s_case_id}')
            self.update_report('features')
        return b_valid

    def get_labels(self):
        if self.s_target in self.s_text:
            self.d_features.update(dict.fromkeys(self.l_target_cols, int(0)))
            self.s_text = self.s_text.strip()
            for match in re.finditer(self.s_target, self.s_text):
                i_start = 5
                i_end = 6
                label = self.s_text[match.end()+i_start:match.end()+i_end]
                if self.s_target in self.s_text[match.end()+i_start:match.end()+i_end+5].strip():
                    label = ''
                while label not in self.l_target_cols and i_start > 0:
                    i_start -= 1
                    i_end -= 1
                    label = self.s_text[match.end() + i_start:match.end() + i_end]
                if label in self.l_target_cols:
                    self.d_features[label] = int(1)
                next_label = self.s_text[match.end() + i_start + 1:match.end() + i_end + 1]
                b_flag = False  # deals with cases: label#1, label#2 in same line
                while next_label not in self.l_target_cols and i_start > 0 and not b_flag:
                    if next_label == self.colon:
                        b_flag = True
                    i_start -= 1
                    i_end -= 1
                    next_label = self.s_text[match.end() + i_start:match.end() + i_end]
                    if self.s_target in self.s_text[match.end() + i_start - 5:match.end() + i_end]:
                        b_flag = True
                if not b_flag and next_label in self.l_target_cols:
                    self.d_features[next_label] = int(1)
        else:  # else no label found
            if self.s_case_id == '31208205':
                self.d_features['A'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31208818':
                self.d_features['D'] = int(1)
                self.d_features['E'] = int(1)
                self.d_features['H'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31269580':
                self.d_features['A'] = int(1)
                self.d_features['K'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31269993':
                self.d_features['A'] = int(1)
                self.d_features['J'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31269980':
                self.d_features['D'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31265139':
                self.d_features['A'] = int(1)
                self.d_features['H'] = int(1)
                self.validate_value('label')
            if self.s_case_id == '31269985':
                self.d_features.update(dict.fromkeys(self.l_target_cols, int(0)))
                self.d_features['A'] = int(1)
                self.d_features['J'] = int(1)
                self.validate_value('label')
            else:
                print(f'No label found for Case ID: {self.s_case_id}')
                self.update_report('label')
                return False
        return self.filter_empty_classes()

    def set_features_demographics(self):
        s_demographics = (self.s_text.split("ת.ז.", 1))[1].split("ממצאים")[0].replace('\n', '')
        i_case_id = self.get_case_id(s_demographics)
        i_gender = self.get_gender(s_demographics)
        i_age = self.get_age(s_demographics)
        s_health_care = self.get_health_care(s_demographics)
        s_unit = self.get_unit(s_demographics)
        s_service = self.get_service(s_demographics, s_health_care)
        o_date_time = self.get_timestamp(s_demographics, s_health_care)
        s_health_care = s_health_care.translate(str.maketrans(self.d_map_format)).strip()
        s_unit = s_unit.translate(str.maketrans(self.d_map_format)).strip()
        s_service = s_service.translate(str.maketrans(self.d_map_format)).strip()
        self.d_features = {'CaseID': i_case_id, 'Age': i_age, 'Gender': i_gender, 'HealthCare': s_health_care,
                           'Unit': s_unit, 'Timestamp': o_date_time, 'ServiceHistory': s_service}

    def set_features_evaluations(self):
        s_descriptions = (self.s_text.split("תיאור הבדיקה", 1))[1].split("תאריך הקלדה")[0]
        l_paragraphs_filter = list(filter(lambda x: x != '', s_descriptions.split('\n')))
        l_paragraphs = self.split_by_paragraph(s_descriptions)
        b_resume = True
        i = 0
        while i < len(l_paragraphs) and b_resume:
            curr_txt = l_paragraphs[i]
            if i == 0:
                s_test_description = curr_txt
                if 'שם' in s_test_description or 'ת.ז.' in s_test_description:
                    s_test_description = ''
                s_test_description = s_test_description.translate(str.maketrans(self.d_map_format)).strip()
                self.d_features.update({'Service': s_test_description})
            if curr_txt != self.space and self.colon in curr_txt and self.s_target not in curr_txt:
                if 'כמות הסמן' in curr_txt:
                    curr_txt = curr_txt.replace(self.colon, '')
                    f_indicator_quantity = self.get_indicator_quantity(curr_txt)
                    s_injection_area = self.get_injection_area(curr_txt)
                    s_range = self.get_injection_range(curr_txt)
                    s_test_settings = self.get_test_settings(curr_txt)
                    i_glucose_level = self.get_glucose_levels(curr_txt)
                    i_start_time = self.get_start_time(curr_txt)
                    s_injection_area = s_injection_area.translate(str.maketrans(self.d_map_format)).strip()
                    s_range = s_range.translate(str.maketrans(self.d_map_format)).strip()
                    s_test_settings = s_test_settings.translate(str.maketrans(self.d_map_format)).strip()
                    self.d_features.update({'VariableAmount': f_indicator_quantity, 'VariableLocation': s_injection_area,
                                            'VariableRange': s_range, 'GlucoseLevel': i_glucose_level,
                                            'TestStartTime': i_start_time, 'TestSetting': s_test_settings})
                else:
                    feature_key = l_paragraphs[i].split(self.colon, 1)[0]
                    feature_value = l_paragraphs[i].split(self.colon, 1)[1]
                    if 'סיבת ההפניה' in feature_value:
                        feature_value = feature_key
                        feature_key = 'סיבת ההפניה'
                    curr_feature, flag_found = self.set_features_key(feature_key, False)
                    if flag_found:
                        feature_value = feature_value.translate(str.maketrans(self.d_map_format)).strip()
                        self.d_features[curr_feature] = feature_value
                    else:
                        print('feature not found: ' + feature_key)
                    if 'לא ספציפיות' in curr_feature:
                        print(f'לא ספציפיות {self.s_case_id}')
                    self.set_features_histogram(curr_feature)
            i += 1
        return self.get_labels()

    def run(self, i, s_content):
        self.s_text = self.normalize_text(s_content)  # Preprocess
        self.d_features = dict()  # Features Dictionary
        self.set_features_demographics()  # Extract Dynamic Demographic Features
        b_valid = self.set_features_evaluations()  # Extract Dynamic Evaluation Features
        if b_valid:  # Saves data if valid
            self.df_data.index += 1
            self.df_data = self.df_data.append(self.d_features, ignore_index=True)
        if i == self.i_records - 1:  # 1779
            print('Done Parsing.')
