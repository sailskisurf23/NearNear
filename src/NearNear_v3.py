import pandas as pd
import numpy as np
#from sklearn.metrics import r2_score

class NearNear(object):
    '''
    This regression model averages 'comparable' items within an expanding
    physical search radius using Haversine Formula

    __init__
    lat - Name of latitude column. First column assumed if None. (str)
    long - Name of long column. Second column assumed if None. (str)
    comp_cat=None - Name of categorical comparables column (str)
    comp_cont=None - Name of continuous comparables column (str)
    distance_rad_increment = 1


    '''


    def __init__(self, lat=None, lon=None, comp_cat=None, comp_cont=None, distance_rad_increment=1):
        # self.k = k #TODO add kNN funcionality
        self.lat = lat
        self.lon = lon
        self.comp_cat = comp_cat # categorical comparison column
        self.comp_cont = comp_cont # continuous comparable column
        self.distance_rad = 1 # miles
        self.distance_rad_increment = distance_rad_increment # miles

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Grab Latitude and Longitude columns and compute distance matrix
        if self.lat == None:
            lat_train = self.X_train.iloc[:,0].values
            lat_test = X_test.iloc[:,0].valuse
        else:
            lat_train = self.X_train[self.lat].values
            lat_test = X_test[self.lat].values
        if self.lon == None:
            lon_train = self.X_train.iloc[:,1].values
            lon_test = X_test.iloc[:,1].values
        else:
            lon_train = self.X_train[self.lon].values
            lon_test = X_test[self.lon].values
        d_miles_mat = compute_distances(lat_train, lon_train, lat_test, lon_test)

        # TODO add continuous comparables funcionality
        # d_imprating_mat = np.abs(self.X_train['property_imprating'].values - X_test['property_imprating'].values.reshape(1,len(X_test)).T)
        # d_locrating_mat = np.abs(self.X_train['property_locrating'].values - X_test['property_locrating'].values.reshape(1,len(X_test)).T)
        #
        # filter out properties outside the specified radii
        # imprating_rad = self.imprating_rad # units of improvement rating
        # locrating_rad = self.locrating_rad # units of location rating
        #
        # imprating_mask = d_imprating_mat - imprating_rad <= 0
        # locrating_mask = d_locrating_mat - locrating_rad <= 0
        # comparables_mask = d_mask & imprating_mask & locrating_mask

        # Apply mask based on search radius and comparable criteria
        distance_rad = self.distance_rad # miles
        distance_rad_increment = self.distance_rad_increment # miles
        d_mask = d_miles_mat - distance_rad <= 0

        if self.comp_cat != None:
            cat_mask = X_train[self.comp_cat] == X_test[self.comp_cat].reshape(1,len(np_cat)).T
            comparables_mask = d_mask & cat_mask
        else:
            comparables_mask = d_mask

        # select target values for comparables for each property and average
        comparables = [self.y_train.values[comparables_mask[i]] for i in range(len(X_test))]
        estimates = [np.mean(estimate) if any(estimate) else 0 for estimate in comparables]

        # expand search radius for properties with no comparables in initial radius
        while not all(estimates) and (distance_rad < 3961):
            distance_rad += distance_rad_increment # miles
            d_mask = d_miles_mat - distance_rad <= 0
            estimate_mask = np.array([[estimate]*len(self.y_train) for estimate in estimates]) <= 0
            if self.comp_cat != None:
                comparables_mask = d_mask & cat_mask & estimate_mask
            else:
                comparables_mask = d_mask & estimate_mask
            estimates_new = [self.y_train.values[comparables_mask[i]] for i in range(len(X_test))]
            estimates = [np.mean(estimate_new) if any(estimate_new) else estimate
                        for estimate_new, estimate
                        in zip(estimates_new, estimates)]
        return np.array(estimates)

    def score(y_true, y_pred):
        return r2_score(y_true, y_pred)


def compute_distances(lat_train, lon_train, lat_test, lon_test):
    '''
    Use Haversine formula to compute distance from each point in train set to each point in test set
    INPUT
    lat_train (np.ndarray)
    lon_train (np.ndarray)
    lat_test (np.ndarray)
    lon_test (np.ndarray)

    OUTPUT
    d_miles_mat (np.ndarray)
    '''
    #convert to radians
    lat_train = lat_train * np.pi / 180
    lon_train = lon_train * np.pi / 180
    lat_test  = lat_test  * np.pi / 180
    lon_test  = lon_test  * np.pi / 180
    # reshape to column vectors so that subtraction results in a matrix of 'distances'
    # rows = each test example
    # columns = each train example
    lat_test = lat_test.reshape(len(lat_test),1)
    lon_test = lon_test.reshape(len(lon_test),1)

    # plug into Haversine Formula
    d_lon = lon_test - lon_train
    d_lat = lat_test - lat_train
    a = (np.sin(d_lat/2))**2 + np.cos(lat_train) * np.cos(lat_test) * (np.sin(d_lon/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d_miles_mat = 3961 * c # (where 3961 is the radius of the Earth in miles)
    return d_miles_mat
