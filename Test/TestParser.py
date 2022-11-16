import os
import csv
import unittest

import pickle  # with vpn
# import pickle5 as pickle  # when no vpn

# ---------------------------------------------------------------------------------------
# Test Class:
#
# Tests functions of the Parser phase.
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class TestParser(unittest.TestCase):

    def __init__(self, b_vpn):
        super().__init__()
        self.b_vpn = b_vpn
        self.p_project = os.path.dirname(os.path.dirname(__file__))
        self.p_input = self.p_project + r'\output\output_parser'
        if self.b_vpn:
            self.p_input = self.set_vpn_dir(self.p_input)

    def set_pickle(self, d, path, filename):
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
    def set_vpn_dir(p_curr):
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

    def validate_path(self, parent_dir, child_filename, extension):
        p_curr = parent_dir + '\\' + child_filename + '.' + extension
        if self.b_vpn:
            p_curr = parent_dir + '/' + child_filename + '.' + extension
        return p_curr

    def test_fixed_input_formats(self):
        d_fixed_true = self.load_dict_fix_true()
        p_to_check = self.validate_path(self.p_input, 'd_fixed', 'pkl')
        if not os.path.exists(p_to_check):
            print('No Fixed Dictionary Found.')
        else:
            d_missing_pred = self.get_pickle(self.p_input, 'd_fixed')
            self.assertDictEqual(d_missing_pred, d_fixed_true)

    def test_invalid_input_formats(self):
        d_missing_true = self.load_dict_missing_true()
        p_missing_pred = self.validate_path(self.p_input, 'd_invalid', 'pkl')
        if not os.path.exists(p_missing_pred):
            print('No Invalid Dictionary Found.')
        else:
            d_missing_pred = self.get_pickle(self.p_input, 'd_invalid')
            self.assertDictEqual(d_missing_pred, d_missing_true)

    def load_dict_missing_true(self):
        p_true = self.validate_path(self.p_input, 'd_invalid_true', 'pkl')
        d_invalid = dict()
        d_invalid = self.get_pickle(self.p_input, 'd_invalid_true')
        return d_invalid

    def load_dict_fix_true(self):
        p_true = self.validate_path(self.p_input, 'd_fixed_true', 'pkl')
        d_fixed = dict()
        d_fixed = self.get_pickle(self.p_input, 'd_fixed_true')
        return d_fixed


# b_vpn = True
b_vpn = False
test = TestParser(b_vpn)
test.test_invalid_input_formats()
test.test_fixed_input_formats()
