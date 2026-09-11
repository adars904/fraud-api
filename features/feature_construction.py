import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def transaction_amt(df):
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df


def TransactionAmt_decimal(df4):
    df4 = df4.copy()
    df4["TransactionAmt_decimal"] = ((df4["TransactionAmt"] - df4["TransactionAmt"].astype(int)) * 100).round(2)
    return df4


def uid(df5):
    df5 = df5.copy()
    df5["uid"] = (df5["card1"].astype(str) + '_' +
                  df5["card2"].astype(str) + '_' +
                  df5["card3"].astype(str) + '_' +
                  df5["card5"].astype(str) + '_' +
                  df5["addr1"].astype(str) + '_' +
                  df5["addr2"].astype(str)
                  )
    return df5


def construct_features(x_train, x_test):
    x_train = transaction_amt(x_train)
    x_test = transaction_amt(x_test)

    x_train = TransactionAmt_decimal(x_train)
    x_test = TransactionAmt_decimal(x_test)

    x_train = uid(x_train)
    x_test = uid(x_test)

    return x_train, x_test


class UIDFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.uid_mean_ = X.groupby("uid")["TransactionAmt"].mean()
        self.uid_std_ = X.groupby("uid")["TransactionAmt"].std()
        return self

    def transform(self, X):
        X = X.copy()
        X["uid_TransactionAmt_mean"] = X["uid"].map(self.uid_mean_)
        X["uid_TransactionAmt_std"] = X["uid"].map(self.uid_std_)

        mean_amt = X["uid_TransactionAmt_mean"]

        X["Amt_to_mean_ratio"] = np.where(
            mean_amt > 0,
            X["TransactionAmt"] / mean_amt,
            np.nan
        )

        X = X.drop(columns=["uid"])

        return X
