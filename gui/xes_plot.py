import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
#import NanoIndent_Analysis
import xes_analysis2

class Analysis_plot:
    def __init__(self, frame):
        self.frame = frame
        self.fig = Figure(figsize=(7,3.3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        # Create initial figure canvas
        self.canvas.get_tk_widget().grid(column=2, row=1, rowspan=13, columnspan=5, sticky="nsew",
                                         padx=5, pady=5)
        self.ax = self.fig.add_subplot(111)
        # create toolbar
        self.toolbarFrame = tk.Frame(master=self.frame)
        self.toolbarFrame.grid(column=2, row=0, rowspan=1, columnspan=5, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.params = {}

    def initial_parameters(self,dir,params,title):
        dir = str(dir.get())
        self.nano_analysis = xes_analysis2.xes_analysis(dir,params)
        self.nano_analysis.extract_data(plot_err=False)
        self.nano_analysis.score()

        self.nano_analysis.analyze()
        self.fig.clf()
        self.nano_analysis.plot_data(title=title,fig_gui = self.fig)
        self.canvas.draw()
        self.title = 'Fit'
        self.xlabel = 'Energy (eV)'
        self.ylabel= 'Counts/s'
        return self.nano_analysis.get_params()
        '''
        self.nano_analysis.score()
        self.nano_analysis.calculate_parameters(verbose=False)
        self.fig.clf()
        self.nano_analysis.plot_data(title=title,fig_gui=self.fig)
        self.canvas.draw()
        return self.nano_analysis.get_params()
        '''
class Data_plot:
    def __init__(self, frame):
        self.frame = frame
        self.fig = Figure(figsize=(5.5, 2.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        # Create initial figure canvas
        self.canvas.get_tk_widget().grid(column=0, row=1, rowspan=4, columnspan=8, sticky="nsew",
                                         padx=5, pady=5)
        self.ax = self.fig.add_subplot(111)
        # create toolbar
        self.toolbarFrame = tk.Frame(master=self.frame)
        self.toolbarFrame.grid(column=3, row=0, rowspan=1, columnspan=5, sticky="w")
        toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.params = {}

    def initial_parameters(self,data_obj):
        """
        fileraw
        """
        self.data_obj = data_obj
        self.title = 0
        self.xlabel = 'Energy (eV)'
        self.ylabel= 'Counts/s'
    #def plot written by evan 
    def plot(self,x_data_array,y_data_array, param_label,title):
        x = self.data_obj.get_x()
        y = self.data_obj.get_y()
        
        self.ax.clear()
        for i in range(len(x_data_array)):
            self.ax.plot(x_data_array[i],y_data_array[i], 'b.', label = param_label)
        #self.ax.invert_xaxis()
        self.ax.legend()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(title)
        self.fig.tight_layout()

        self.canvas.draw()

    