import numpy as np

def clip_img(x):
    return np.float32(-1 if x<-1 else (1 if x > 1 else x))
