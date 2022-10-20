from Model import Model
import os


# ---------------------------------------------------------------------------------------
# Controller Class:
#
# Mediator Class between View and Model.
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------


class Controller:

    _model = None

    def __init__(self):
        self.p_project = os.path.dirname(os.path.dirname(__file__))

    def run(self, d_config):
        _model = Model.Model(self, d_config)
        _model.run()
