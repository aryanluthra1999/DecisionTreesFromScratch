import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample

import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, kind="Classifier", max_depth=3, feature_ids=None):
        self.max_depth = max_depth
        self.type = kind
        self.features = feature_ids
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        print("Decision Tree Classifier")

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

    @staticmethod
    def entropy(labels):
        unique, counts = np.unique(labels, return_counts=True)
        counts = np.linalg.norm(counts)
        counts = counts * np.log2(counts)
        return -1 * np.sum(counts)

    @staticmethod
    def gini_impurity(labels):
        # TODO implement gini_impurity function
        unique, counts = np.unique(labels, return_counts=True)
        counts = np.linalg.norm(counts)
        counts = counts ** 2
        return 1 - np.sum(counts)

    @staticmethod
    def purification(X, y, idx, thresh):
        G_i = DecisionTree.gini_impurity(y)

        X0, y0, X1, y1 = DecisionTree.split(X, y, idx, thresh)
        G_f = DecisionTree.gini_impurity(y0) * max(y0.shape) + DecisionTree.gini_impurity(y1) * max(y1.shape)
        G_f /= max(y.shape)

        return G_i - G_f

    @staticmethod
    def information_gain(X, y, idx, thresh):
        # TODO implement information gain function
        S_i = DecisionTree.entropy(y)

        X0, y0, X1, y1 = DecisionTree.split(X, y, idx, thresh)

        S_f = DecisionTree.entropy(y0) * max(y0.shape) + DecisionTree.entropy(y1) * max(y1.shape)
        S_f /= max(y.shape)

        return S_i - S_f

    @staticmethod
    def get_best_thresh(X, y, split_idx, method="entropy"):
        assert method in ["entropy", "gini"]
        if method == "entropy":
            gain = DecisionTree.information_gain
        else:
            gain = DecisionTree.purification

        feat = np.sort(np.unique(X[:, split_idx].flatten()))
        # print(feat)
        if len(feat) > 1:
            possible_thresh = [sum([feat[i], feat[i + 1]]) / 2 for i in range(len(feat) - 1)]
            gains = [gain(X, y, split_idx, thresh) for thresh in possible_thresh]

            return possible_thresh[np.argmax(gains)]
        else:
            return 0

    def get_best_split(self, X, y, method="entropy"):
        assert method in ["entropy", "gini"]

        best_threshs = [DecisionTree.get_best_thresh(X, y, idx, method) for idx in self.features]
        best_idx = np.argmax(best_threshs)

        return best_idx, best_threshs[best_idx]

    def segmenter(self, X, y, method="gini"):
        assert method in ["entropy", "gini"]

        if self.max_depth == 0 or len(np.unique(y.flatten())) == 1:
            self.data, self.pred = y, np.mean(y)
            return
        else:
            best_idx, best_thresh = self.get_best_split(X, y, method)
            print("best idx: ", best_idx)
            print("best thresh: ", best_thresh)
            self.split_idx, self.thresh = best_idx, best_thresh

    def is_leaf(self):
        return self.left is None and self.right is None

    def split_rule(self):
        return self.split_idx, self.thresh

    def train(self, X, y, method="gini"):
        if self.features is None:
            self.features = np.arange(X.shape[1])

        if self.max_depth <= 0 or len(np.unique(y.flatten())) == 1:
            self.data, self.pred = y, np.mean(y)
        else:
            self.segmenter(X, y, method)
            X0, y0, X1, y1 = DecisionTree.split(X, y, self.split_idx, self.thresh)
            self.left = DecisionTree(kind=self.type, max_depth=self.max_depth - 1, feature_ids=self.features)
            self.right = DecisionTree(kind=self.type, max_depth=self.max_depth - 1, feature_ids=self.features)
            self.left.train(X0, y0, method)
            self.left.train(X1, y1, method)

    def fit(self, X, y, method="gini"):
        self.train(X, y, method)

    def predict_once(self, Xi, proba=False):
        if self.is_leaf():
            if proba:
                return self.pred
            else:
                return int(np.round(self.pred))
        else:
            # print(self.split_rule())
            if self.thresh is None or Xi[self.split_idx] < self.thresh:
                return self.left.predict_once(Xi, proba)
            else:
                return self.right.predict_once(Xi, proba)

    def predict(self, X, proba=False):
        return np.array([self.predict_once(Xi, proba) for Xi in X])


def DecisionTreeTest():
    X = [1, 4, 3, 2, 7, 8, 5, 3, 9, 7]
    y = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    thresh = 5
    # pdb.set_trace()
    result = DecisionTree.information_gain(X, y, thresh)
    assert result == 1, "Got {} but should be 1".format(result)


class RandomForest:

    def __init__(self, n=200, max_depth=3, num_features=None):
        if params is None:
            params = {}
        # TODO implement function
        # self.gain_thresh = min_info_or_purity_gain
        self.num_features = num_features
        self.decision_trees = [DecisionTree(max_depth) for i in range(n)]

    def fit(self, X, y):
        feature_columns = np.arange(X.shape[1])

        if not self.num_features:
            self.num_features = np.ceil(np.sqrt(len(feature_columns)))

        for tree in self.decision_trees:
            strapped_features = np.random.sample(feature_columns, size=self.num_features)
            tree.features = strapped_features
            strapped_X, strapped_y = resample(X, y, replace=True)
            tree.train(strapped_X, strapped_y)

    def predict_once(self, Xi, proba=False):
        results = [tree.predict_once(Xi, True) for tree in self.decision_trees]
        result = np.mean(results)

        if proba:
            return result
        else:
            return int(np.round(result))

    def predict(self, X, proba):
        return np.array([self.predict_once(Xi, proba) for Xi in X])


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
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":

    DecisionTreeTest()
    dataset = "titanic"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z)[:100])

    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO implement and evaluate parts c-h
