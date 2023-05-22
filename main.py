# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import ComplementNB
# from sklearn.naive_bayes import BernoulliNB
# from sklearn import tree
# from random import randint
#
# #загрузка данных
# data = pd.read_csv('iris.data', header=None)
# #3
# X = data.iloc[:, :4].to_numpy()
# labels = data.iloc[:, 4].to_numpy()
# #4
# le = preprocessing.LabelEncoder()
# Y = le.fit_transform(labels)
# #5
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
# #Байесовские методы
# #1
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print(np.count_nonzero(y_test != y_pred))
# #2 точность классификации
# print(gnb.fit(X_train, y_train).score(X_test, y_test) * 100)
# #3
# size = 0
# list_test_size = []
# percentage_misclassified_observations = []
# classification_accuracy = []
# while size <= 0.95:
#     size += 0.05
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
#     gnb = GaussianNB()
#     y_pred = gnb.fit(X_train, y_train).predict(X_test)
#     list_test_size.append(size)
#     percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
#     classification_accuracy.append(gnb.fit(X_train, y_train).score(X_test, y_test))
# fig, ax = plt.subplots()
# ax.bar(list_test_size, classification_accuracy, width=0.03)
# ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
# ax.set_facecolor('seashell')
# fig.set_figwidth(17)
# fig.set_figheight(8)
# fig.set_facecolor('floralwhite')
# plt.show()
# #Классифицирующие деревья
# #1
# clf = MultinomialNB(force_alpha=True)
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print((y_test != y_pred).sum())
#
# print(np.count_nonzero(y_test != y_pred))
# print(clf.fit(X_train, y_train).score(X_test, y_test) * 100)
#
# clf = ComplementNB(force_alpha=True)
# y_pred = clf.fit(X_train, y_train).predict(X_test)
#
# print(f'Количество наблюдений, который были неправильно определены {np.count_nonzero(y_test != y_pred)}')
# print(f'Точность классификации {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
#
# clf = BernoulliNB(force_alpha=True)
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(f'Количество наблюдений, который были неправильно определены {np.count_nonzero(y_test != y_pred)}')
# print(f'Точность классификации {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
# clf = tree.DecisionTreeClassifier()
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(np.count_nonzero(y_test != y_pred))
# print(f'Точность классификации {clf.fit(X_train, y_train).score(X_test, y_test) * 100}%')
# #3
# print(f'Количество листьев: {clf.get_n_leaves()}')
# print(f'Глубина: {clf.get_depth()}')
# #4
# plt.subplots(1, 1, figsize=(10, 10))
# tree.plot_tree(clf, filled=True)
# plt.show()
# #5
# size = 0
# list_test_size = []
# percentage_misclassified_observations = []
# classification_accuracy = []
# while size <= 0.95:
#     size += 0.05
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size)
#     gnb = tree.DecisionTreeClassifier()
#     y_pred = gnb.fit(X_train, y_train).predict(X_test)
#     list_test_size.append(size)
#     percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
#     classification_accuracy.append(gnb.fit(X_train, y_train).score(X_test, y_test))
# fig, ax = plt.subplots()
# ax.bar(list_test_size, classification_accuracy, width=0.03)
# ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
# ax.set_facecolor('seashell')
# fig.set_figwidth(17)
# fig.set_figheight(8)
# fig.set_facecolor('floralwhite')
# plt.show()
# #6
# criterion_parameters = ('gini', 'entropy', 'log_loss')
# splitter_parameter = ('best', 'random')
# for parameter in criterion_parameters:
#     sp_par_random = splitter_parameter[randint(0, 1)]
#     max_dp_random = randint(5, 40)
#     min_samples_split_random = randint(5, 40)
#     min_samples_leaf_random = randint(5, 40)
#     gnb = tree.DecisionTreeClassifier(criterion=parameter, splitter=sp_par_random, max_depth=max_dp_random,
#                                       min_samples_split=min_samples_split_random,
#                                       min_samples_leaf=min_samples_leaf_random)
#     y_pred = gnb.fit(X_train, y_train).predict(X_test)
#     print(
#         f'При criterion: {parameter}, splitter: {sp_par_random}, max_depth: {max_dp_random}, min_samples_split: {min_samples_split_random}, min_samples_leaf: {min_samples_leaf_random} \n точность классификации {gnb.fit(X_train, y_train).score(X_test, y_test) * 100}%, количество листьев: {gnb.get_n_leaves()}, глубина: {gnb.get_depth()}\n')
