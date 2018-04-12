#-*- coding: utf-8 -*-
from __future__ import unicode_literals
from multi_label_surveys import multi_label_surveys
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV

#1. training and testing
training_test = multi_label_surveys(r'data\Consumer_Complaints_Train_Test.csv')
lemma = False
training_test.clean_df(lemmatize=lemma)
training_test.build_test_train_df(under_sample=False)
#models are commented to reduce runtimes. Uncomment for improved results
single_models = [
    # MLPClassifier(verbose=False,hidden_layer_sizes=(5,),max_iter=200),
    # MLPClassifier(verbose=False,hidden_layer_sizes=(5,),max_iter=80),
    # MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #               beta_1=0.9, beta_2=0.999, early_stopping=False,
    #               epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
    #               learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
    #               warm_start=False),
    # LogisticRegression(),
    LinearSVC(),
    # SVC(kernel='rbf'),
    # SVC(kernel='poly'),
    # SVC(kernel='sigmoid'),
    # DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=50, max_depth=2,
                                      random_state=0, max_features=None)
    ]
multi_models = [
    #ridge has a long run time
    #RidgeClassifierCV(),
    # DecisionTreeClassifier(),
    # LogisticRegression(random_state=0, multi_class='ovr'),
    LinearSVC(multi_class='ovr'),
    Perceptron(verbose=False, max_iter=200),
    # Perceptron(verbose=False, max_iter=2000),
    # MultinomialNB(),
    #RandomForestClassifier()
]
for single_model in single_models:
    for multi_model in multi_models:
        training_test.transform_df(single_model=single_model, multi_model=multi_model)
        print('single model is: ' + str(single_model))
        print('multi model is: ' + str(multi_model))
        training_test.multi_label_metric(verbose=False, best_model_metric='labels')
        training_test.print_write(r'data\complaints_classified.csv', verbose=False)

#Now use the best models for training
training_test.transform_df(single_model=training_test.most_accurate_single_model,
                           multi_model=training_test.most_accurate_multi_model)
training_test.multi_label_metric(verbose=True)
training_test.print_write(r'data\complaints_classified.csv', verbose=False)
best_single_model = training_test.most_accurate_single_model

#2. deploy trained models for production use
#   a. extract models from training
production = multi_label_surveys(r'data\Consumer_Complaints_Production.csv')
production.clean_df(production=True, lemmatize=lemma)
production.set_count_vect(training_test.get_count_vect())
production.set_single_classifier(training_test.get_single_classifier())
production.set_tfidf_transformer(training_test.get_tfidf_transformer())
production.set_multi_classifier(training_test.get_multi_classifier())
production.set_multi_class_predicted_i_t(training_test.get_multi_class_predicted_i_t())
production.set_mlb(training_test.get_mlb())
#   b. label production instances
production.label_for_production()
production.production_write_df(r'data\entire_df.csv')


#related documentation/tutorials
#http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
#https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb