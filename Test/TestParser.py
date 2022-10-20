import os
import csv
import unittest

# import pickle  # with vpn
import pickle5 as pickle  # when no vpn

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
        if os.path.exists(p_true):
            d_invalid = self.get_pickle(self.p_input, 'd_invalid_true')
        else:
            d_invalid['31220153'] = ['glucose', 'start time', 'injection area']
            d_invalid['31118619'] = ['label']
            d_invalid['31211155'] = ['glucose', 'start time']
            d_invalid['31239819'] = ['start time']
            d_invalid['31069503'] = ['label']
            d_invalid['31267572'] = ['start time']
            d_invalid['31240862'] = ['start time']
            d_invalid['31254164'] = ['start time']
            d_invalid['31208822'] = ['label']
            d_invalid['31167164'] = ['label']
            d_invalid['31254391'] = ['glucose']
            d_invalid['31248395'] = ['glucose']
            d_invalid['31277579'] = ['label']
            d_invalid['31233222'] = ['start time']
            d_invalid['31251433'] = ['glucose']
            d_invalid['31245436'] = ['glucose']
            d_invalid['31274468'] = ['glucose']
            d_invalid['31267382'] = ['start time']
            d_invalid['31269483'] = ['glucose']
            d_invalid['31209327'] = ['glucose']
            d_invalid['31274617'] = ['glucose']
            d_invalid['31275737'] = ['label']
            d_invalid['31267171'] = ['glucose']
            d_invalid['31261266'] = ['start time']
            d_invalid['31287498'] = ['glucose']
            d_invalid['31258514'] = ['glucose']
            d_invalid['31292445'] = ['glucose']
            d_invalid['31250574'] = ['glucose']
            d_invalid['31227787'] = ['start time']
            d_invalid['31185269'] = ['label']
            d_invalid['31225270'] = ['glucose']
            d_invalid['31260428'] = ['glucose']
            d_invalid['31260513'] = ['glucose']
            d_invalid['31266893'] = ['glucose']
            d_invalid['31284113'] = ['glucose']
            d_invalid['31218363'] = ['label']
            d_invalid['31260692'] = ['start time']
            d_invalid['31247681'] = ['glucose']
            d_invalid['31283872'] = ['injection area']
            d_invalid['31264780'] = ['glucose']
            d_invalid['31214989'] = ['indicator quantity', 'injection area', 'glucose', 'start time']
            d_invalid['31245406'] = ['glucose']
            d_invalid['31218189'] = ['start time']
            d_invalid['31288037'] = ['glucose']
            d_invalid['31258866'] = ['label']
            d_invalid['31287567'] = ['glucose']
            d_invalid['31263770'] = ['glucose']
            d_invalid['31260543'] = ['glucose']
            d_invalid['31293343'] = ['start time']
            d_invalid['31269580'] = ['glucose']
            d_invalid['31217175'] = ['label']
            d_invalid['31288050'] = ['glucose']
            d_invalid['31278896'] = ['label']
            d_invalid['31216052'] = ['label']
            d_invalid['31267100'] = ['glucose']
            d_invalid['31257469'] = ['glucose']
            d_invalid['31254363'] = ['glucose']
            d_invalid['31274734'] = ['glucose']
            d_invalid['31277022'] = ['start time']
            d_invalid['31259946'] = ['label']
            d_invalid['10727102'] = ['start time']
            d_invalid['31251476'] = ['glucose']
            d_invalid['31268006'] = ['start time']
            d_invalid['31283636'] = ['label']
            d_invalid['31290825'] = ['label']
            d_invalid['31294415'] = ['start time']
            d_invalid['31293651'] = ['start time']
            d_invalid['31215898'] = ['label']
            d_invalid['31240712'] = ['start time']
            d_invalid['31258466'] = ['label']
            d_invalid['31233098'] = ['label']
            d_invalid['31290386'] = ['glucose']
            d_invalid['31280519'] = ['glucose']
            d_invalid['31281414'] = ['label']
            d_invalid['31272833'] = ['start time']
            d_invalid['31254498'] = ['glucose']
            d_invalid['31245272'] = ['label']
            d_invalid['31216017'] = ['label']
            d_invalid['31208564'] = ['label']
            d_invalid['31276386'] = ['glucose']
            d_invalid['31244717'] = ['glucose']
            d_invalid['31237095'] = ['glucose']
            d_invalid['31227560'] = ['start time']
            d_invalid['31267026'] = ['glucose']
            d_invalid['31268220'] = ['label']
            d_invalid['31214572'] = ['glucose', 'start time']
            d_invalid['31243272'] = ['label']
            d_invalid['31274666'] = ['glucose', 'start time']
            d_invalid['31215838'] = ['label']
            d_invalid['31260598'] = ['glucose']
            d_invalid['31227781'] = ['start time']
            d_invalid['31235351'] = ['label']
            d_invalid['31290405'] = ['glucose']
            d_invalid['31231207'] = ['glucose', 'start time']
            d_invalid['31242767'] = ['glucose']
            d_invalid['31254332'] = ['glucose']
            d_invalid['31248328'] = ['glucose']
            d_invalid['31232061'] = ['label']
            d_invalid['31242226'] = ['label']
            d_invalid['31271631'] = ['start time']
            d_invalid['31262897'] = ['glucose, start time']
            d_invalid['31268990'] = ['indicator quantity', 'injection area', 'glucose', 'start time']
            d_invalid['31257917'] = ['glucose']
            d_invalid['31280642'] = ['glucose']
            d_invalid['31275777'] = ['label']
            d_invalid['31262174'] = ['indicator quantity']
            d_invalid['31216093'] = ['label']
            d_invalid['31241629'] = ['glucose']
            d_invalid['31260358'] = ['glucose']
            d_invalid['31215080'] = ['glucose']
            d_invalid['31287387'] = ['glucose']
            d_invalid['31284136'] = ['glucose']
            d_invalid['31290704'] = ['label']
            d_invalid['31233793'] = ['label']
            d_invalid['31260380'] = ['glucose']
            d_invalid['31279059'] = ['label']
            d_invalid['31245492'] = ['glucose']
            d_invalid['31258373'] = ['label']
            d_invalid['31249410'] = ['glucose']
            d_invalid['31234548'] = ['indicator quantity']
            d_invalid['31075483'] = ['label']
            d_invalid['31269442'] = ['glucose']
            d_invalid['31259680'] = ['glucose']
            d_invalid['31266938'] = ['glucose']
            d_invalid['31210379'] = ['glucose']
            d_invalid['31266993'] = ['glucose']
            d_invalid['31263795'] = ['glucose']
            d_invalid['31270027'] = ['glucose']
            d_invalid['31277116'] = ['glucose']
            d_invalid['31277153'] = ['glucose']
            d_invalid['31260151'] = ['glucose']
            d_invalid['31290557'] = ['label']
            d_invalid['31233295'] = ['label']
            d_invalid['31246490'] = ['label']
            d_invalid['31255376'] = ['start time']
            d_invalid['31268429'] = ['start time']
            d_invalid['31224023'] = ['start time']
            d_invalid['31268621'] = ['label']
            d_invalid['31266808'] = ['indicator quantity', 'injection area', 'glucose', 'start time']
            self.set_pickle(d_invalid, self.p_input, 'd_invalid_true')
        return d_invalid

    def load_dict_fix_true(self):
        p_true = self.validate_path(self.p_input, 'd_fixed_true', 'pkl')
        d_fixed = dict()
        if os.path.exists(p_true):
            d_fixed = self.get_pickle(self.p_input, 'd_fixed_true')
        else:
            d_fixed['31208205'] = ['label fix']
            d_fixed['31208818'] = ['label fix']
            d_fixed['31269580'] = ['label fix']
            d_fixed['31269993'] = ['label fix']
            d_fixed['31269980'] = ['label fix']
            d_fixed['31265139'] = ['label fix']
            d_fixed['31269985'] = ['label fix']
            d_fixed['31287822'] = ['age fix']
            d_fixed['31222606'] = ['age fix']
            d_fixed['31230305'] = ['indicator quantity fix']
            d_fixed['31214989'] = ['indicator quantity fix']
            d_fixed['31220153'] = ['injection area fix']
            d_fixed['31221251'] = ['injection area fix']
            d_fixed['31215838'] = ['injection area fix']
            d_fixed['31249729'] = ['glucose fix']
            d_fixed['31292428'] = ['glucose fix']
            d_fixed['31292445'] = ['glucose fix']
            d_fixed['31291866'] = ['glucose fix']
            d_fixed['31263515'] = ['glucose fix']
            d_fixed['31262174'] = ['glucose fix']
            d_fixed['31292121'] = ['glucose fix']
            d_fixed['31292560'] = ['glucose fix']
            d_fixed['31292175'] = ['glucose fix']
            d_fixed['31278972'] = ['start time fix']
            d_fixed['31259896'] = ['start time fix']
            d_fixed['31220153'] = ['start time fix']
            d_fixed['31230965'] = ['start time fix']
            d_fixed['31275777'] = ['start time fix']
            d_fixed['31242230'] = ['start time fix']
            d_fixed['31268621'] = ['start time fix']
            self.set_pickle(d_fixed, self.p_input, 'd_fixed_true')
        return d_fixed


# b_vpn = True
b_vpn = False
test = TestParser(b_vpn)
test.test_invalid_input_formats()
test.test_fixed_input_formats()
