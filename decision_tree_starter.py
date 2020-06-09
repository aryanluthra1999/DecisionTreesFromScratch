# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from tqdm import tqdm_notebook

import pydot_ng as pydot
import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    def str_root(self):
        if self.split_idx:
            if self.features:
                feat_name = str(self.features[self.split_idx])
            else:
                feat_name = str(self.split_idx)

            return "Pred " + str(self.pred) + ". Splitting on " + feat_name + " with thresh " + str(self.thresh)
        else:
            return "Final Pred : " + str(self.pred)

    @staticmethod
    def gini_impurity(labels):
        # TODO implement gini_impurity function
        unique, counts = np.unique(labels, return_counts=True)
        counts = np.linalg.norm(counts)
        counts = counts ** 2
        return 1 - np.sum(counts)

    @staticmethod
    def purification(X, y, thresh):
        G_i = DecisionTree.gini_impurity(y)
        if len(X.shape) < 2:
            shape_X = np.array([X]).T
        else:
            shape_X = X

        X0, y0, X1, y1 = DecisionTree.split(shape_X, y, 0, thresh)
        G_f = DecisionTree.gini_impurity(y0) * len(y0) + DecisionTree.gini_impurity(y1) * len(y1)
        G_f /= len(y0) +len(y1)

        return G_f - G_i

    @staticmethod
    def entropy(labels):
        if len(labels) == 0:
            return 0
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) == 1:
            return 0
        assert len(counts) == 2, 'Counts are... elems:' + str(unique) + "   counts:" + str(counts) + "\n labels :" + str(labels)
        return scipy.stats.entropy(counts, base=2)

    @staticmethod
    def information_gain(X, y, thresh, gini=False):
        # TODO implement information gain function
        if gini:
            result = DecisionTree.purification(X, y, thresh)
            # print(result)
            return result

        S_i = DecisionTree.entropy(y)

        if len(X.shape) < 2:
            shape_X = np.array([X]).T
        else:
            shape_X = X
        X0, y0, X1, y1 = DecisionTree.split(shape_X, y, 0, thresh)

        S_f = DecisionTree.entropy(y0) * len(y0) + DecisionTree.entropy(y1) * len(y1)
        S_f /= len(y)

        result = S_i - S_f
        # print(result)
        return result

    @staticmethod
    def split(X, y, idx, thresh):
        X0, idx0, X1, idx1 = DecisionTree.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    @staticmethod
    def split_test(X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y, num_feat=None, bootstrap=False):
        if not num_feat:
            num_feat = X.shape[1]
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            valid_feats = np.random.choice(np.arange(X.shape[1]), size=num_feat, replace=False)

            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                if (i in valid_feats) else np.zeros(10) for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                if bootstrap:
                    X0, y0 = resample(X0, y0)
                    X1, y1 = resample(X1, y1)
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


def print_tree(t, indent=0):
    """Print a representation of this tree in which each node is
    indented by two spaces times its depth from the root.
    """
    print('  ' * indent + t.str_root())
    branches = []
    if t.left != None:
        print_tree(t.left, indent + 1)
    if t.right != None:
        print_tree(t.right, indent + 1)


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, max_depth=10, features=None, n=20):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [DecisionTree(max_depth, features) for i in range(self.n)]

    def fit(self, X, y):
        # TODO implement function
        print("Training")
        for dt in tqdm(self.decision_trees):
            Xr, yr = resample(X, y)
            dt.fit(Xr, yr)
        return

    def predict(self, X, proba=False, notebook = True):
        if notebook:
            tqdm = tqdm_notebook
        # TODO implement function
        print("Predicting")
        result = np.mean([t.predict(X) for t in tqdm(self.decision_trees)], axis=0)
        if proba:
            return result
        else:
            return np.round(result)


class RandomForest(BaggedTrees):
    def __init__(self, params=None, max_depth=10, features=None, n=20, m=None):
        BaggedTrees.__init__(self, params, max_depth, features, n)
        self.m = m
    
    def fit(self, X, y, notebook = True):
        if notebook:
            tqdm = tqdm_notebook
        if not self.m:
            self.m = int(np.ceil(np.sqrt(X.shape[1])))
        print("Training")
        for dt in tqdm(self.decision_trees):
            dt.fit(X, y, num_feat=self.m, bootstrap=True)
        return


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf, X, y):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y, cv=5))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)
#
# if __name__ == '__main__':
#     dataset = "titanic"
#     params = {
#         "max_depth": 5,
#         # "random_state": 6,
#         "min_samples_leaf": 10,
#     }
#     N = 100
#     path_train = 'titanic_training.csv'
#     data = genfromtxt(path_train, delimiter=',', dtype=None)
#     path_test = 'titanic_testing_data.csv'
#     test_data = genfromtxt(path_test, delimiter=',', dtype=None)
#     y = data[1:, 0]  # label = survived
#     class_names = ["Died", "Survived"]
#
#     labeled_idx = np.where(y != b'')[0]
#     y = np.array(y[labeled_idx], dtype=np.int)
#     print("\n\nPart (b): preprocessing the titanic dataset")
#     X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
#     X = X[labeled_idx, :]
#     Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
#     assert X.shape[1] == Z.shape[1]
#     features = list(data[0, 1:]) + onehot_features
#
#     print("Features:", features)
#     print("Train/test size:", X.shape, Z.shape)
#
#     print("\n\nPart 0: constant classifier")
#     print("Accuracy", 1 - np.sum(y) / y.size)
#
#     # Basic decision tree
#     print("\n\nPart (a-b): simplified decision tree")
#     dt = DecisionTree(max_depth=10, feature_labels=features)
#     dt.fit(X, y)
#     # print_tree(dt)
#     print(sklearn.metrics.accuracy_score(y, dt.predict(X)))
#     print("Predictions", dt.predict(Z)[:100])
#
#     print("\n\nPart (c): sklearn's decision tree")
#     clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
#     clf.fit(X, y)
#
#     print(sklearn.metrics.accuracy_score(y, clf.predict(X)))
#     print("Predictions", clf.predict(Z)[:100])
#
#     evaluate(clf, X, y)
#     out = io.StringIO()
#     sklearn.tree.export_graphviz(
#         clf, out_file=out, feature_names=features, class_names=class_names)
#     graph = pydot.graph_from_dot_data(out.getvalue())
#     pydot.graph_from_dot_data(out.getvalue()).write_pdf(f'''{dataset}-tree.pdf''')
