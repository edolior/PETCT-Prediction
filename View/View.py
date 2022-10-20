

# ---------------------------------------------------------------------------------------
# View Class:
#
# User Interface for input of all the settings.
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class View:

    _controller = None

    def __init__(self, ref_controller):
        self._controller = ref_controller

    def run(self):

        d_config_input = {
                            'b_processing': False,
                            # 'b_processing': True,
                            # 'i_sample': 1000,
                            'i_sample': None,
                            'b_vpn': True,
                            # 'b_vpn': False,
                            'b_tta': True
                            # 'b_tta': False
                         }

        self._controller.run(d_config_input)
