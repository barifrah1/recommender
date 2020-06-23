import pandas as pd
import numpy as np
def get_data():
    train=pd.read_csv(TRAIN_PATH)
    train['User_ID_Alias']=train['User_ID_Alias']-1
    train['Movie_ID_Alias']=train['Movie_ID_Alias']-1
    validation=pd.read_csv(VALIDATION_PATH)
    return train,validation
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    # return train, validation
    raise NotImplementedError


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
