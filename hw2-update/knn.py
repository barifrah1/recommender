import numpy as np
import pandas as pd
import os
import csv
from interface import Regressor
from utils import get_data, Config
from tqdm import tqdm
from config import *
class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.K = config.k
        self.n_users = None
        self.n_items = None
        self.items_dict = {}
        self.users_dict = {}
        self.items_mean_rating = {}
        self.items_corr_dict = {}
        self.global_mean = None
        #raise NotImplementedError

    def fit(self, X):
        self.global_mean = X[RATING_COL_NAME_IN_DATASET].mean()
        X_rows = X.values # converts X from datafram to np.array
        if os.path.isfile(CORRELATION_PARAMS_FILE_PATH): #if correlations have already been calculated, upload them and build user and item dicts and item mean rating. 
            self.n_users =X_rows[:,USERS_COL_INDEX].max()
            self.n_items =X_rows[:,ITEMS_COL_INDEX].max()
            for row in X_rows:
                if(row[ITEMS_COL_INDEX] not in self.items_dict.keys()): # build item dictionary
                    self.items_dict[np.int16(row[ITEMS_COL_INDEX])]  = {}
                if(row[USERS_COL_INDEX] not in self.users_dict.keys()): # build user dictionary
                    self.users_dict[np.int16(row[USERS_COL_INDEX])]  = set()
                self.items_dict[np.int16(row[ITEMS_COL_INDEX])][np.int16(row[USERS_COL_INDEX])] = np.int16(row[RATINGS_COL_INDEX])
                self.users_dict[np.int16(row[USERS_COL_INDEX])].add(np.int16(row[ITEMS_COL_INDEX])) 
            self.upload_params() #upload correlations
            return
        else: # in case corelations haven't been calculated yet
            #build  items dictionary - for each item, who watched it.
            for row in X_rows:
                if(row[ITEMS_COL_INDEX] not in self.items_dict.keys()):
                    self.items_dict[np.int16(row[ITEMS_COL_INDEX])]  = {}
                if(row[USERS_COL_INDEX] not in self.users_dict.keys()):
                    self.users_dict[np.int16(row[USERS_COL_INDEX])]  = set()
                self.items_dict[np.int16(row[ITEMS_COL_INDEX])][np.int16(row[USERS_COL_INDEX])] = np.int16(row[RATINGS_COL_INDEX])
                self.users_dict[np.int16(row[USERS_COL_INDEX])].add(np.int16(row[ITEMS_COL_INDEX]))
            self.items_mean_rating = dict(X[[ITEM_COL_NAME_IN_DATASET,RATING_COL_NAME_IN_DATASET]].groupby(ITEM_COL_NAME_IN_DATASET, as_index=False).mean().values)
            self.n_users =X.max()[USERS_COL_INDEX]
            self.n_items =X.max()[ITEMS_COL_INDEX]
            self.build_item_to_itm_corr_dict(X)
            self.save_params()# save correlations to csv
            self.symmetric_corr_matrix() # create the symmetric values in the dictionary - important for prediction ponly
            
    def build_item_to_itm_corr_dict(self, data):
        items =  self.items_dict.keys() 
        for i in tqdm(range(self.n_items+1)): #run over item i
            self.items_corr_dict[i] = {}
            if( i in items):
                item_i_keys = set(self.items_dict[i].keys())
            for j in range(i+1,self.n_items+1): #calculate correlation between item i and item j, j start from i+1 because it's symmetric
                if(i in items and j in items):
                    intersection_group = item_i_keys.intersection(set(self.items_dict[j].keys())) # all the users who rated item i and item j
                    if(len(intersection_group)>1):# if item i and item j has more than 1 mutual user who rate them both
                        intersection_group = list(intersection_group)
                        mechane = 0
                        #calculating correlation 
                        new_i_dict = {key:self.items_dict[i][key]-self.items_mean_rating[i] for key in intersection_group}
                        new_j_dict = {key:self.items_dict[j][key]-self.items_mean_rating[j] for key in intersection_group}
                        mone = sum( new_i_dict[key]*new_j_dict[key] for key in intersection_group ) # mone in pearsom equation
                        mechane = np.sqrt(sum(map(lambda x: x*x ,new_i_dict.values())))*np.sqrt(sum(map(lambda x: x*x ,new_j_dict.values())))
                        if(mechane==0):
                            continue
                        else:
                            sim_i_j = mone/(mechane) #similarity calculation between item i and item j
                            if(sim_i_j>0): # save it only if corelation is positive
                                self.items_corr_dict[np.int16(i)][np.int16(j)] = np.float32(sim_i_j)
                      


    def predict_on_pair(self, user, item): #predict rating for user and item
        if( user not in self.users_dict.keys() or item not in self.items_corr_dict.keys()):
            return self.global_mean

        intersection_group = self.users_dict[user].intersection(set(self.items_corr_dict[item].keys()))
        if(len(intersection_group)==0 ):
            return self.global_mean
        only_relevent_elements_user_u = { key_item:self.items_corr_dict[item][key_item] for key_item in intersection_group }
        sorted_dict = sorted(only_relevent_elements_user_u.items(), key=lambda x: x[1], reverse=True)[:self.K]
        return sum(map(lambda x: x[1]*self.items_dict[x[0]][user],sorted_dict))/sum(map(lambda x: x[1],sorted_dict))

    def upload_params(self): # upload corr_matrix to items_corr_dict instead of calculatng it again
        self.corr = pd.read_csv(CORRELATION_PARAMS_FILE_PATH).values
        for row in self.corr:
            if(row[0] not in self.items_corr_dict.keys()):
                self.items_corr_dict[row[0]] = {}
            self.items_corr_dict[np.int16(row[0])][np.int16(row[1])] = np.float32(row[2])
        self.symmetric_corr_matrix()
            
    def save_params(self): # save corr_matrix to csv file 
        with open(CORRELATION_PARAMS_FILE_PATH, 'w') as csv_file:  
            writer = csv.writer(csv_file,lineterminator='\n')
            writer.writerow(CSV_COLUMN_NAMES)
            for key, value in self.items_corr_dict.items():
                for k2,v2 in value.items():
                    writer.writerow([np.int16(key),np.int16(k2),np.float32(v2)])
    
    def calculate_rmse(self, data):
        data = data.values
        e = 0
        for row in data:
            #print(row)
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e / data.shape[0])
    
    def symmetric_corr_matrix(self):
        for i in range(self.n_items +1):
            if(i in self.items_corr_dict.keys()):
                for key,value in self.items_corr_dict[i].items():
                    if(key not in self.items_corr_dict.keys()):
                        self.items_corr_dict[key] = {}
                    self.items_corr_dict[key][i] = value
                


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
