import matplotlib.pyplot as plt
from colours import plot_colours
ocols = plot_colours()

def params():
    plotpar = {'axes.labelsize': 18,
               'text.fontsize': 10,
               'legend.fontsize': 15,
               'xtick.labelsize': 18,
               'ytick.labelsize': 18,
               'text.usetex': True}
    plt.rcParams.update(plotpar)
    return {'capsize':0, 'fmt':'k.', 'ecolor':'.8'}

class colours(object):
    def __init__(self):

        self.orange = '#FF9933'
        self.lightblue = '#66CCCC'
        self.blue = '#0066CC'
        self.pink = '#FF33CC'
        self.turquoise = '#3399FF'
        self.lightgreen = '#99CC99'
        self.green = '#009933'
        self.maroon = '#CC0066'
        self.purple = '#9933FF'
        self.red = '#CC0000'
        self.lilac = '#CC99FF'
