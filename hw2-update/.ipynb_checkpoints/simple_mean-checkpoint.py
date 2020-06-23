from interface import Regressor
from utils import get_data


class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

        
    def fit(self, X):
        x=X.groupby('User_ID_Alias')['Ratings_Rating'].mean()
        for y in range(len(x)):
            self.user_means[y]=x[y]
            

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user+1]

if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
