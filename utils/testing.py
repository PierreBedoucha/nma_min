# functions or decorators for performace testing

import time
import platform

REPORT = './performance_report.txt'

def timing(func):
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - t0
        name, module = func.__name__, func.__module__
        arg_str = ', '.join(str(arg)[:50] for arg in args) # print out only [:50]
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        suffix = ' [at %s, %s]' % (platform.node(), date)
        
        with open(REPORT, 'a') as f:
            f.write('[%0.4fs] %s.%s \nParameters: %s \n%s \n\n' %
                    (elapsed, module, name, arg_str, suffix))
        return result    
    return clocked
