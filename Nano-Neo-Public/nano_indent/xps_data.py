import numpy as np

class xps_data():
    def __init__(self,fileName,skipLn):
        self.fileName = fileName
        self.skipLn = skipLn #int represnting files to skip
        self.read_file_txt(fileName)

    #not yet complete i was just throwing down some starter code
    def read_file_txt(self,fileName):
        self.x,self.y = np.loadtxt(self.fileName,skiprows=self.skipLn,unpack=True)
        self.numP = len(self.x)
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
        