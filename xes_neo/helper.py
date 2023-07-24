import time
from __init__  import __version__

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def timecall():
    return time.time()

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def str_to_list(s):
    arr = [float(i) for i in list(s.split(","))]
    return arr

def norm(val):
    return np.linalg.norm(val)


def banner():
    """
    Tim Stack:
    https://patorjk.com/software/taag/#p=display&h=1&v=1&f=Univers&t=XES%20NEO
    colossal?
    """
    banner_str = ('''
                    XES_NEO ver %s  
                                                                                       
8b        d8  88888888888  ad88888ba       888b      88  88888888888  ,ad8888ba,    
 Y8,    ,8P   88          d8"     "8b      8888b     88  88          d8"'    `"8b   
  `8b  d8'    88          Y8,              88 `8b    88  88         d8'        `8b  
    Y88P      88aaaaa     `Y8aaaaa,        88  `8b   88  88aaaaa    88          88  
    d88b      88"""""       `"""""8b,      88   `8b  88  88"""""    88          88  
  ,8P  Y8,    88                  `8b      88    `8b 88  88         Y8,        ,8P  
 d8'    `8b   88          Y8a     a8P      88     `8888  88          Y8a.    .a8P   
8P        Y8  88888888888  "Y88888P"       88      `888  88888888888  `"Y8888Y"'                                                                              

    '''% __version__)

    return banner_str
