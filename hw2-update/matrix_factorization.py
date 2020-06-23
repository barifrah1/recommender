from interface import Regressor
from typing import Dict
from utils import Config, get_data
from pandas import array
import numpy as np
from numpy import square, sqrt
from config import USER_COL_NAME_IN_DATAEST,RATING_COL_NAME_IN_DATASET,RATINGS_COL_INDEX,USERS_COL_INDEX,ITEMS_COL_INDEX

class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.k=config.k
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.p=None
        self.q=None
        self.current_epoch = 0
        self.global_mean=None  
        self.R=None     

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        return self.gamma*(np.sum(self.user_biases**2)+np.sum(self.item_biases**2)+np.sum(self.p**2)+np.sum(self.q**2))

    def fit(self, X):
        self.n_users =X.max()[USERS_COL_INDEX]
        self.n_items =X.max()[ITEMS_COL_INDEX]
        #Define user_biases as vector with normal distribution around 0 with very low variance
        self.user_biases = np.random.normal(0,0.01,self.n_users+1)
        #Define item_biases as vector with normal distribution around 0 with very low probability.
        #set it with numbers close to zero helps us finds better results.
        self.item_biases = np.random.normal(0,0.01,self.n_items+1)
        self.p=np.random.normal(0,0.01, size=(self.n_users+1, self.k))
        self.q=np.random.normal(0,0.01, size=(self.n_items+1, self.k))
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
            print("validation_rmse:",self.calculate_rmse(validation))

    def run_epoch(self, data: np.array):
        i=0
        np.random.shuffle(data)
        for x in data:
            i+=1
            # check the tuple is valid(users and movies from the list)
            if(x[USERS_COL_INDEX] >=0 and x[USERS_COL_INDEX] <=self.n_users and x[ITEMS_COL_INDEX] >=0 and x[ITEMS_COL_INDEX] <=self.n_items): 
                pu=self.p[x[USERS_COL_INDEX]]# take the u-th row from latent matrix P
                qi=self.q[x[ITEMS_COL_INDEX]]# take the i-th row from latent matrix Q
                qit=np.transpose(qi) #change qi from row to column.
                r_est=self.global_mean+self.user_biases[x[USERS_COL_INDEX]]+self.item_biases[x[ITEMS_COL_INDEX]]+np.matmul(pu,qit) # calculate rui hat.
                error=x[RATINGS_COL_INDEX]-r_est #calculate the error with the real rating
                self.user_biases[x[USERS_COL_INDEX]]=self.user_biases[x[USERS_COL_INDEX]]+self.lr*(error-self.gamma*self.user_biases[x[USERS_COL_INDEX]]) # update bu
                self.item_biases[x[ITEMS_COL_INDEX]]=self.item_biases[x[ITEMS_COL_INDEX]]+self.lr*(error-self.gamma*self.item_biases[x[ITEMS_COL_INDEX]]) #update bi
                if(i>1): # if its not the first ore the second iteration update also pu an qi vectors
                    pu=pu+self.lr*(error*qi-self.gamma*pu) #update pu
                    qi=qi+self.lr*(error*pu-self.gamma*qi) #update qi
                    self.p[x[USERS_COL_INDEX]]=pu #update matrix P in the u-th row with pu
                    self.q[x[ITEMS_COL_INDEX]]=qi  #update matrix Q in the i-th row with qi            
        self.R=np.matmul(self.p,np.transpose(self.q))
        
        

    def predict_on_pair(self, user: int, item: int):
        x=self.global_mean+self.user_biases[user]+self.item_biases[item]+self.R[user][item]
        return x
    
    
    def calculate_rmse(self, data: array):
        e = 0
        data = data.values
        for row in data:
            user, item, rating = row
            e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.009030143369021236,
        gamma=0.0528855224903822365,
        k=44,
        epochs=50)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
