__author__ = 'nelson'

# Define the basic template of a layer
class PlainLayer():
    def __init__(self, name):
        self.name = name
        self.W = None
        self.b = None
        self.input_data = None
        self.output_data = None
        self.input_shape = None
        self.output_shape = None
        self.params = []
        self.gparams = []       # Not being used
        self.opt = True         # Needs optimization
        return
