import pandas as pd
import numpy as np
from config import TRAIN_PATH, VALIDATION_PATH, USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET
def get_data():
    train=pd.read_csv(TRAIN_PATH)
    train[USER_COL_NAME_IN_DATAEST]=train[USER_COL_NAME_IN_DATAEST]-1
    train[ITEM_COL_NAME_IN_DATASET]=train[ITEM_COL_NAME_IN_DATASET]-1
    validation=pd.read_csv(VALIDATION_PATH)
    validation[USER_COL_NAME_IN_DATAEST]=validation[USER_COL_NAME_IN_DATAEST]-1
    validation[ITEM_COL_NAME_IN_DATASET]=validation[ITEM_COL_NAME_IN_DATASET]-1
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
