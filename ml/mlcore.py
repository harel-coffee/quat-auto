#!/usr/bin/env python3
"""
mlcore lib for machine learning experiments

Steve Göring 2017
"""
import sys
import os
import argparse
import json

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import itertools

import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
import itertools

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier

import pandas as pd

try:
    from sklearn.model_selection import cross_val_predict
except:
    from sklearn.cross_validation import cross_val_predict
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.tree import export_graphviz


def print_trees(pipeline, feature_columns, name="trees"):
    os.makedirs(name, exist_ok=True)
    i = 0
    feature_names = []
    for k, fname in enumerate(feature_columns):
        if pipeline.named_steps["feature_selection"].get_support()[k]:
            feature_names.append(fname)

    feature_columns = [x for x in feature_columns]
    for tree in pipeline.named_steps["regressor"].estimators_:
        treefilename = "{}/tree_".format(name) + str(i) + ".dot"
        export_graphviz(tree,
                    out_file=treefilename,
                    feature_names=list(feature_names),
                    filled=True,
                    rounded=True)
        i += 1
        os.system("dot -Tpdf {} -o {}.pdf".format(treefilename, treefilename))


def train_dummy_class(x, y):
    X = x.copy()
    X = X.values
    Y = y.values.astype(np.int32)

    dummy_clf = DummyClassifier()
    predicted = cross_val_predict(dummy_clf, X, Y, cv=10)
    dummy_clf.fit(X, Y)
    crossval_result_table = pd.DataFrame({"predicted": predicted,
                                          "truth": Y})
    return {
        "dummyclassifier": dummy_clf,
        "crossval": crossval_result_table
    }


def train_gradboost_class(x, y, num_trees=10, threshold="0.001*mean"):
    X = x.copy()
    columns = X.columns
    X = X.values
    Y = y.values

    pipeline = Pipeline([('feature_selection', SelectFromModel(ExtraTreesClassifier(n_jobs=-1),
                                                               threshold=threshold)),
                         ('regressor', GradientBoostingClassifier(n_estimators=num_trees
                                                                 ))])
    predicted = cross_val_predict(pipeline, X, Y, cv=10)
    pipeline.fit(X, Y)

    crossval_result_table = pd.DataFrame({"predicted": predicted, "truth": Y})

    return {
        "gradboost": pipeline,
        "crossval": crossval_result_table
    }


def train_rf_class(x, y, num_trees=10, threshold="0.001*mean", n_splits=10):
    X = x.copy()
    X = X.values
    Y = y.values  # .astype(np.int32)

    pipeline = Pipeline([('feature_selection', SelectFromModel(ExtraTreesClassifier(n_jobs=-1),
                                                               threshold=threshold)),
                         ('classifier', RandomForestClassifier(n_estimators=num_trees, n_jobs=-1,
                                                               criterion="entropy",
                                                               #class_weight="balanced_subsample",
                                                               #max_features=None
                                                               ))])
    predicted = cross_val_predict(pipeline, X, Y, cv=n_splits)
    pipeline.fit(X, Y)
    crossval_result_table = pd.DataFrame({"predicted": predicted,
                                          "truth": Y})
    return {
        "randomforest": pipeline,
        "crossval": crossval_result_table
    }


def train_knn_class(x, y):
    X = x.copy()

    X = X.values

    Y = y.values  # .astype(np.int32)

    pipeline = Pipeline([('feature_selection', SelectFromModel(ExtraTreesClassifier(n_jobs=-1),
                                                               threshold="0.001*mean")),
                         ('classifier', KNeighborsClassifier(n_jobs=-1))
                        ])
    predicted = cross_val_predict(pipeline, X, Y, cv=10)
    pipeline.fit(X, Y)
    crossval_result_table = pd.DataFrame({"predicted": predicted, "truth": Y})

    return {
        "knn": pipeline,
        "crossval": crossval_result_table
    }


def train_rf_regression(x, y, num_trees=10, threshold="0.001*mean"):
    X = x.replace([np.inf, -np.inf], np.nan).fillna(0).values
    Y = y.values
    columns = x.columns

    pipeline = Pipeline([('feature_selection', SelectFromModel(ExtraTreesRegressor(n_jobs=-1),
                                                               threshold=threshold)),
                         ('regressor', RandomForestRegressor(n_estimators=num_trees, n_jobs=-1,
                                                             criterion="mse"
                                                             ))])
    predicted = cross_val_predict(pipeline, X, Y, cv=10)
    pipeline.fit(X, Y)

    feature_importance_threshold = pipeline.named_steps["feature_selection"].threshold_
    feature_selected_supp = pipeline.named_steps["feature_selection"].get_support()

    feature_importances_values_ = pipeline.named_steps['regressor'].feature_importances_

    feature_importance_values = []
    i = 0
    for x in feature_selected_supp:
        if x:
            feature_importance_values.append(feature_importances_values_[i])
            i += 1
        else:
            feature_importance_values.append(0)

    feature_importance = pd.DataFrame({"feature": list(columns),
                                       "importance":  feature_importance_values
                                       })

    crossval_result_table = pd.DataFrame({"predicted": predicted, "truth": Y})
    return {
        "randomforest": pipeline,
        "crossval": crossval_result_table,
        "feature_importance_threshold": feature_importance_threshold,
        "feature_importance": feature_importance,
        "used_features": float(feature_selected_supp.sum()),
        "number_features": len(columns),
    }


def train_gradboost_regression(x, y, num_trees=10, threshold="0.001*mean"):
    X = x.copy()
    columns = X.columns
    print(columns)
    X = X.values
    Y = y.values

    pipeline = Pipeline([('feature_selection', SelectFromModel(ExtraTreesRegressor(n_jobs=-1),
                                                               threshold=threshold)),
                         ('regressor', GradientBoostingRegressor(n_estimators=num_trees,
                                                                 criterion="friedman_mse"
                                                                 ))])
    predicted = cross_val_predict(pipeline, X, Y, cv=10)
    pipeline.fit(X, Y)

    crossval_result_table = pd.DataFrame({"predicted": predicted, "truth": Y})

    return {
        "gradboost": pipeline,
        "crossval": crossval_result_table
    }


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plotname = "figures/confusion_{}.png".format(title.lower().replace(" ", "_"))
    plt.savefig(plotname)
    plt.savefig(plotname.replace(".png", ".pdf"))


def eval_plots_class(truth, pred, title=""):
    rmse = np.sqrt(mean_squared_error(truth, pred))
    r2 = r2_score(truth, pred)
    acc = accuracy_score(truth, pred)

    print(title)
    res = classification_report(truth, pred)
    print("classification report")
    print(res)

    cm = confusion_matrix(truth, pred)

    plt.figure()
    classes = sorted(list(set(truth) | set(pred)))
    plot_confusion_matrix(cm,
                          classes=classes,
                          normalize=True,
                          title='Confusion matrix, {}, with normalization'.format(title))
    metrics = {
        "title": title,
        "rmse": rmse,
        "r2": r2,
        "acc": acc
    }
    print(json.dumps(metrics, indent=4, sort_keys=True))

    #print_roc(truth, pred, title)
    return metrics


def eval_plots_regression(truth, pred, title="", folder="", plotname=""):
    df = pd.DataFrame({"truth": truth, "predicted": pred})
    bounds = (min(df["truth"].min(), df["predicted"].min()),
              max(df["truth"].max(), df["predicted"].max()))

    ax = df.plot(x="predicted",
                 y="truth",
                 kind="scatter",
                 xlim=bounds,
                 ylim=bounds,
                 figsize=(6, 6),
                 title=title)

    ax.plot(bounds, bounds, 'k--', lw=2, color="gray")
    ax.text(-0.5, -0.5, "opt", horizontalalignment='center',  bbox=dict(facecolor='lightgray', alpha=0.5))
    os.makedirs(folder, exist_ok=True)
    ax.get_figure().savefig(folder + "/scatter_{}.png".format(plotname))
    ax.get_figure().savefig(folder + "/scatter_{}.pdf".format(plotname))


def load_serialized(filename_with_path):
    """ load a serialized model """
    if not os.path.isfile(filename_with_path):
        print("{} is not a valid file, please check".format(filename_with_path))
        return
    return joblib.load(filename_with_path)


def save_serialized(clf, filename_with_path):
    """ save model to a file """
    storage_dir = os.path.dirname(filename_with_path)
    if storage_dir != "":
        os.makedirs(storage_dir, exist_ok=True)
    joblib.dump(clf, filename_with_path)