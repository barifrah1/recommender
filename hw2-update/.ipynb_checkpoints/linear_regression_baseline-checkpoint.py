from typing import Dict

import numpy as np

from interface import Regressor
from utils import Config, get_data


class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

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
        raise NotImplementedError

    def fit(self, X):
        """
        pseudo code:

            self.n_users =
            self.n_items =
            self.user_biases = np.zeros(self.n_users)
            self.item_biases = np.zeros(self.n_items)
            self.global_bias =
            while self.current_epoch < self.train_epochs:
                self.run_epoch(X)
                train_mse = np.square(self.calculate_rmse(X))
                train_objective = train_mse * X.shape[0] + self.calc_regularization()
                epoch_convergence = {"train_objective": train_objective,
                                     "train_mse": train_mse}
                self.record(epoch_convergence)
                self.current_epoch += 1
            self.save_params()


        """

        raise NotImplementedError

    def run_epoch(self, data: np.array):
        raise NotImplementedError

    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

    def save_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
