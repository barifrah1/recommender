from interface import Regressor
from utils import get_data
from config import *
from numpy import square, sqrt
from pandas import array

class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}
        
        
    def fit(self, X):
        # using X as dataframe , group by records of user and then calculating mean of ratings for each user
        #Saving a dictionary that its key is user id and the value is the mean rating of the user.
        tuples = X[[USER_COL_NAME_IN_DATAEST,RATING_COL_NAME_IN_DATASET]].groupby(USER_COL_NAME_IN_DATAEST, as_index=False).mean().values
        for row in tuples:
            self.user_means[row[0]] = row[1]
        self.global_mean = X.mean()
        return
            

    def predict_on_pair(self, user: int, item: int):
        # if user exists in training set, return his mean score , else return all ratings mean score
        if(user in self.user_means.keys()):
            return self.user_means[user]
        else:
            return self.global_mean
  
# default rmse calculating with one change - change data from dataframe to np.array with 2D
    def calculate_rmse(self, data):
        e = 0
        data = data.values
        for row in data:
            user, item, rating = row
            e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])  


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
