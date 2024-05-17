import numpy as np
import pandas as pd
from lib import encoding as en
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)
df = pd.read_csv('dataset/bank_dataset.csv')

df.head()

df.loc[df['products_number'] == 4, 'products_number'] = 3

df['credit_score_seg'] = pd.cut(df['credit_score'], bins=[349, 500, 590, 620, 660, 690, 720, np.inf],
                                labels=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

df['balance_seg'] = pd.cut(df['balance'], bins=[-1, 50000, 90000, 127000, np.inf],
                           labels=['A', 'B', 'C', 'D'])

df['age_seg'] = pd.cut(df['age'], bins=[17, 36, 55, np.inf],
                       labels=['A', 'B', 'C'])

df['tenure_seg'] = pd.cut(df['tenure'], bins=[-1, 3, 5, 7, np.inf],
                          labels=['A', 'B', 'C', 'D'])

df_final = df.drop('customer_id', axis=1)

df_final = en.one_hot_encoder(df_final, ['country', 'gender', 'age_seg'], drop_first=True)

en.label_encoder(df_final, 'credit_score_seg')
en.label_encoder(df_final, 'balance_seg')
en.label_encoder(df_final, 'tenure_seg')

y = df_final['churn']
X = df_final.drop(columns=['churn'], axis=1)

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)

model = LGBMClassifier(verbose=-1).fit(X, y)
cv_results = cross_validate(model, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
f1 = cv_results['test_f1'].mean()
auc = cv_results['test_roc_auc'].mean()
accuracy = cv_results['test_accuracy'].mean()
print(f'f1: {f1:.2f}')
print(f'auc: {auc:.2f}')
print(f'accuracy: {accuracy:.2f}')

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

gs_best = GridSearchCV(model, lightgbm_params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
final_model = model.set_params(**gs_best.best_params_)

cv_results = cross_validate(final_model, X, y, cv=3, scoring=['accuracy', 'f1', 'roc_auc'])
f1 = cv_results['test_f1'].mean()
auc = cv_results['test_roc_auc'].mean()
accuracy = cv_results['test_accuracy'].mean()

print(f'f1: {f1:.2f}')
print(f'auc: {auc:.2f}')
print(f'accuracy: {accuracy:.2f}')


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(model, X)


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=3):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


val_curve_params(model, X, y, "max_depth", range(1, 11), scoring="f1")


def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
    print(score.T)


base_models(X_smote, y_smote)

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
        best_models[name] = final_model
    print(score.T)
    return best_models


hyperparameter_optimization(X_smote, y_smote)

#           accuracy        f1   roc_auc
# LR        0.687430  0.688229  0.744370
# KNN       0.678199  0.704607  0.736213
# SVC       0.571581  0.636674  0.592588
# CART      0.791350  0.781778  0.791363
# RF        0.846481  0.831376  0.931166
# Adaboost  0.806169  0.786922  0.895204
# GBM       0.827141  0.810116  0.914503
# XGBoost   0.852572  0.832040  0.936142 || 0.854079  0.834652  0.936105
# LightGBM  0.843279  0.821761  0.929809 || 0.860734  0.841322  0.939579
# CatBoost  0.852446  0.827656  0.934845
