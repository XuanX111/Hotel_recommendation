import data_io
import random
import pandas as pd
import ml_metrics as metrics
import numpy as np
from sklearn.decomposition import PCA

from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import datetime
from scipy.sparse import csr_matrix, hstack
from sklearn.naive_bayes import BernoulliNB


class recom_nb():

    def __init__(self):
        self.train = data_io.read_train()
        self.test = data_io.read_test()
        self.destin = data_io.read_desin()

        # pca analysis on the destination
        pca = PCA(n_components=3)
        self.dest_pca = pca.fit_transform(self.destin[["d{0}".format(i + 1) for i in range(149)]])
        self.dest_pca = pd.DataFrame(self.dest_pca)
        self.dest_pca["srch_destination_id"] = self.destin["srch_destination_id"]


    def downs_sample(self,size=10000):

        ## adding time feature
        self.train["date_time"] = pd.to_datetime(self.train["date_time"])
        self.train["year"] = self.train["date_time"].dt.year
        self.train["month"] = self.train["date_time"].dt.month

        # test_ids = set(self.test.user_id.unique())
        # train_ids = set(self.train.user_id.unique())
        # intersection_count = len(test_ids & train_ids)

        unique_users = self.train.user_id.unique()
        down_user_id = random.sample(list(unique_users),size)
        down_train = self.train[self.train.user_id.isin(down_user_id)]

        t_before_train = down_train[((down_train.year == 2013) | ((down_train.year == 2014) & (down_train.month < 8)))]
        t_after_train = down_train[((down_train.year == 2014) & (down_train.month >= 8))]
        # t_after_train = t_after_train[t_after_train.is_booking==True]

        return t_after_train

    def simple_model(self,X_train):
        most_common = list(self.train.hotel_cluster.value.counts().head(5).index)
        y_pred = [most_common for i in range(X_train.shape[0])]
        return y_pred

    def evaluate(self,y_pred,y_true):

        return metrics.mapk(y_true,y_pred,k=5)

    def features_engineer(self,df):
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
        df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")

        props = {}
        for prop in ["month", "day", "dayofweek", "quarter"]:
            props[prop] = getattr(df["date_time"].dt, prop)

        carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
        for prop in carryover:
            props[prop] = df[prop]

        date_props = ["month", "day", "dayofweek", "quarter"]
        for prop in date_props:
            props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
            props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
        props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

        ret = pd.DataFrame(props)

        ret = ret.join(self.dest_pca, on="srch_destination_id", how='left', rsuffix="dest")
        ret = ret.drop("srch_destination_iddest", axis=1)
        ret.fillna(-1, inplace=True)   #replace with the missing value...

        return ret

    def train_nb(self,X,y):

        def map5eval(actual, preds):

            predicted = preds.argsort(axis=1)[:, -np.arange(5)]
            # print(predicted)
            metric = 0.
            for i in range(5):
                metric += np.sum(actual == predicted[:, i]) / (i + 1)
            metric /= actual.shape[0]
            return metric

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, stratify=y, test_size=0.2)
        map5 = make_scorer(map5eval,greater_is_better = True,needs_proba=True)

        clf = BernoulliNB(alpha=1.0)

        sw = 1+4*X_train.is_booking
        clf.partial_fit(X_train, y_train,classes = np.arange(100),sample_weight = sw)
        score = cross_val_score(clf,X_train,y_train,cv=5,scoring = map5)
        print(score)
        return clf, X_test,y_test

    def predict_nb(self, clf, X_test):
        y_pred = clf.predict_proba(X_test)
        preds = y_pred

        # df_pred = pd.DataFrame(y_pred)
        # preds = []
        # for index, row in df_pred.iterrows():
        #     preds.append(list(row.nlargest(5).index))

        return preds

    def write_output(self,model,file):
        data_io.save_model(model,file)

def main():
    df_nb = recom_nb()
    print('start')
    down_train = df_nb.downs_sample()
    print('Done downsizing')
    feature_train = df_nb.features_engineer(down_train)
    print('Done feature')
    clf, X_test, y_test = df_nb.train_nb(feature_train,feature_train['hotel_cluster'])
    # print(y_pred)
    y_pred = df_nb.predict_nb(clf, X_test)
    df_nb.write_output(y_pred, 'nb')
    print('Done training')




if __name__ == "__main__":
    main()
