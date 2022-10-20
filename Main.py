from Controller import Controller
from View import View

# ---------------------------------------------------------------------------------------
# Main Class:
#
# Runs the Model class.
#
#
# Edo Lior
# PET/CT Prediction
# BGU ISE
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    _controller = Controller.Controller()
    _view = View.View(_controller)
    _view.run()
