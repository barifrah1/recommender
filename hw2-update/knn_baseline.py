import numpy as np
import pickle
from interface import Regressor
from utils import get_data, Config
from config import *


class KnnBaseline(Regressor):
    def __init__(self, config):
        self.K = config.k
        self.n_users = None
        self.n_items = None
        self.items_dict = {}
        self.users_dict = {}
        self.items_mean_rating = {}
        self.items_corr_dict = {}
        self.bu = None
        self.bi = None
        self.global_mean = None


    def fit(self, X):
        X_rows = X.values # converts X from datafram to np.array
        self.n_users =X_rows[:,USERS_COL_INDEX].max()
        self.n_items =X_rows[:,ITEMS_COL_INDEX].max()
        #item and user means for prediction pahse(users or items with no data about them)
        self.items_mean_rating = dict(X[[ITEM_COL_NAME_IN_DATASET,RATING_COL_NAME_IN_DATASET]].groupby(ITEM_COL_NAME_IN_DATASET, as_index=False).mean().values)
        self.user_mean=X[[USER_COL_NAME_IN_DATAEST,RATING_COL_NAME_IN_DATASET]].groupby(USER_COL_NAME_IN_DATAEST, as_index=False).mean().values
        for row in X_rows:
            if(row[ITEMS_COL_INDEX] not in self.items_dict.keys()): #building user and item dicts
                self.items_dict[np.int16(row[ITEMS_COL_INDEX])]  = {}
            if(row[USERS_COL_INDEX] not in self.users_dict.keys()):
                self.users_dict[np.int16(row[USERS_COL_INDEX])]  = set()
            self.items_dict[np.int16(row[ITEMS_COL_INDEX])][np.int16(row[USERS_COL_INDEX])] = np.int16(row[RATINGS_COL_INDEX])
            self.users_dict[np.int16(row[USERS_COL_INDEX])].add(np.int16(row[ITEMS_COL_INDEX])) 
        self.upload_params() #upload correlations and biases

    def predict_on_pair(self, user: int, item: int):
        #case of user that does not appear in the train
        if( user not in self.users_dict.keys()):
            pre=self.items_mean_rating[item]
            return pre
        #case of item that does not appear in the train   
        if (item not in self.items_corr_dict.keys()):
            pre=self.user_mean[user][1]
            return pre
        #calculating set of itesm that user i rated and item j has positive correlation with
        intersection_group = self.users_dict[user].intersection(set(self.items_corr_dict[item].keys()))
        #case of strangeness
        if(len(intersection_group)==0 ): # if no such of those, return baseline regression result
            return self.global_mean + self.bu[user] + self.bi[item]
        only_relevent_elements_user_u = { key_item:self.items_corr_dict[item][key_item] for key_item in intersection_group }
        sorted_dict = sorted(only_relevent_elements_user_u.items(), key=lambda x: x[1], reverse=True)[:self.K]
        bui = self.global_mean + self.bu[user] +self.bi[item]
        return bui+(sum(map(lambda x: x[1]*(self.items_dict[x[0]][user]-(self.global_mean + self.bu[user] +self.bi[x[0]])),sorted_dict))/sum(map(lambda x: x[1],sorted_dict)))
        


    def upload_params(self): # upload corr_matrix to items_corr_dict instead of calculatng it again
        self.corr = pd.read_csv(CORRELATION_PARAMS_FILE_PATH).values
        for row in self.corr:
            if(row[0] not in self.items_corr_dict.keys()):
                self.items_corr_dict[row[0]] = {}
            self.items_corr_dict[np.int16(row[0])][np.int16(row[1])] = np.float32(row[2])
        self.symmetric_corr_matrix()
        # open a file, where you stored the pickled data
        file = open(BASELINE_PARAMS_FILE_PATH, 'rb')
        # dump information to that file
        data = pickle.load(file)
        # close the file
        file.close()
        self.global_mean = data[0]
        self.bu = data[1]
        self.bi = data[2]

    def save_params(self):
        raise NotImplementedError
        
    def symmetric_corr_matrix(self):
        for i in range(self.n_items +1):
            if(i in self.items_corr_dict.keys()):
                for key,value in self.items_corr_dict[i].items():
                    if(key not in self.items_corr_dict.keys()):
                        self.items_corr_dict[key] = {}
                    self.items_corr_dict[key][i] = value
                    
    def calculate_rmse(self, data):
        data = data.values
        e = 0
        for row in data:
            #print(row)
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e / data.shape[0])


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
