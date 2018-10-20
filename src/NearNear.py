import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class NearNear(object):

    def __init__(self, k=3, imprating_rad = 1, locrating_rad = 1, distance_rad = 1, distance_rad_increment = 1):
        self.k = k
        self.imprating_rad = imprating_rad # units of improvement rating
        self.locrating_rad = locrating_rad # units of location rating
        self.distance_rad = distance_rad # miles
        self.distance_rad_increment = distance_rad_increment # miles

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        '''Compute estimated housing price for a X_test'''

        # Use haversine formula to compute physical distances between each property in train set and each property in test set
        latitude_train = self.X_train['property_latitude'].values*np.pi / 180
        longitude_train = self.X_train['property_longitude'].values*np.pi / 180
        latitude_test = X_test['property_latitude'].values*np.pi / 180
        longitude_test = X_test['property_longitude'].values*np.pi / 180
        # reshape to column vectors so that subtraction results in a matrix
        # rows = each test example
        # columns = distance from entire train set
        latitude_test = latitude_test.reshape(len(latitude_test),1)
        longitude_test = longitude_test.reshape(len(longitude_test),1)

        d_lat = latitude_test - latitude_train
        d_long = longitude_test - longitude_train
        a = (np.sin(d_lat/2))**2 * np.cos(latitude_test) * np.cos(latitude_train) * (np.sin(d_long/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d_miles_mat = 3961 * c # (where 3961 is the radius of the Earth in miles)

        # compute distances in 'imprating' and 'locrating'
        d_imprating_mat = np.abs(self.X_train['property_imprating'].values - X_test['property_imprating'].values.reshape(1,len(X_test)).T)
        d_locrating_mat = np.abs(self.X_train['property_locrating'].values - X_test['property_locrating'].values.reshape(1,len(X_test)).T)

        # filter out properties outside the specified radii
        imprating_rad = self.imprating_rad # units of improvement rating
        locrating_rad = self.locrating_rad # units of location rating
        distance_rad = self.distance_rad # miles
        distance_rad_increment = self.distance_rad_increment # miles

        d_mask = d_miles_mat - distance_rad <= 0
        imprating_mask = d_imprating_mat - imprating_rad <= 0
        locrating_mask = d_locrating_mat - locrating_rad <= 0
        comparables_mask = d_mask & imprating_mask & locrating_mask

        estimates = [self.y_train.values[comparables_mask[i]] for i in range(len(X_test))]
        estimates = [np.mean(estimate) if any(estimate) else 0 for estimate in estimates]

        while not all(estimates) and (distance_rad < 3961):
            # ^continue until all estimates made or the earth is scoured
            distance_rad += distance_rad_increment # miles
            d_mask = d_miles_mat - distance_rad <= 0
            estimate_mask = np.array([[estimate]*len(self.y_train) for estimate in estimates]) <= 0
            comparables_mask = d_mask & imprating_mask & locrating_mask & estimate_mask
            estimates_new = [self.y_train.values[comparables_mask[i]] for i in range(len(X_test))]
            estimates = [np.mean(estimate_new) if any(estimate_new) else estimate
                        for estimate_new, estimate
                        in zip(estimates_new, estimates)]
        return np.array(estimates)

    def score(y_true, y_pred):
        return r2_score(y_true, y_pred)
