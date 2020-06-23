from typing import Dict
import pickle
import numpy as np
from numpy import square, sqrt
from pandas import array, unique

from interface import Regressor
from utils import Config, get_data
from config import *

class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr #learning rate parameter
        self.gamma = config.gamma # regulization parameter
        self.train_epochs = config.epochs #numer of iterations for sgd algorithm
        self.n_users = None # number of users
        self.n_items = None # number of items
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None 
        self.global_mean=None # mean rating of all records

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

# calculating regulization part in objective function
    def calc_regularization(self):
        return self.gamma*(np.sum(self.user_biases**2)+np.sum(self.item_biases**2))

    def fit(self, X):
        self.n_users =X.max()[USERS_COL_INDEX] #calculating size of bu array
        self.n_items =X.max()[ITEMS_COL_INDEX] #calculating size of bi array
        self.users = set(unique(X[USER_COL_NAME_IN_DATAEST].values))
        self.items = set(unique(X[ITEM_COL_NAME_IN_DATASET].values))
        #Define user_biases as vector with normal distribution around 0 with very low variance
        self.user_biases = np.random.normal(0,0.0000000001,self.n_users+1)
        #Define item_biases as vector with normal distribution around 0 with very low variance.
        #randomness around zero helps us find better results.
        self.item_biases = np.random.normal(0,0.0000000001,self.n_items+1)
        self.global_mean=X[RATING_COL_NAME_IN_DATASET].mean()
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X.values)
            print("rmse ",self.calculate_rmse(X))
            train_mse = np.square(self.calculate_rmse(X))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                    "train_mse": train_mse}
            print("train_mse:", train_mse)                        
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()



    def run_epoch(self, data):
        for x in data:
            user_index = x[USERS_COL_INDEX]
            item_index = x[ITEMS_COL_INDEX]
            self.user_biases[user_index]=self.user_biases[user_index]+self.lr*(x[RATINGS_COL_INDEX]-self.global_mean-self.user_biases[user_index]-self.item_biases[item_index]-self.gamma*self.user_biases[user_index])
            self.item_biases[item_index]=self.item_biases[item_index]+self.lr*(x[RATINGS_COL_INDEX]-self.global_mean-self.user_biases[user_index]-self.item_biases[item_index]-self.gamma*self.item_biases[item_index])        
    
    def predict_on_pair(self, user: int, item: int):
        if(user not in self.users or item not in self.items):
            return self.global_mean
        else:
            x=self.global_mean+self.user_biases[user]+self.item_biases[item]
            return x
    
    # defaule rmse , only change is that data is a dataframe 
    def calculate_rmse(self, data):
        e = 0
        data = data.values
        for row in data:
            user, item, rating = row
            e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])
    
    # save global mean and bu and bi arrays to pickle file
    def save_params(self):
        lis =[self.global_mean,self.user_biases,self.item_biases]
        pickle.dump(lis, open(BASELINE_PARAMS_FILE_PATH, "wb"))



if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
