from __future__ import unicode_literals
import pandas as pd
import numpy as np
import editdistance
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

import spacy
nlp = spacy.load('en_core_web_sm')

plt.subplots_adjust(top=0.98, bottom=0.47, left=0.27, right=0.99, hspace=0.20,
                    wspace=0.20)

class multi_label_surveys:

    def __init__(self, training_test_file_path):
        self.df = pd.read_csv(training_test_file_path, encoding = 'utf-8')
        #df_train,df_test are numpy arrays. Unknown parameterless constructor for np array
        self.df_train = None
        self.df_test = None
        self.X_train = []
        self.X_test = []
        self.y_traintext = []
        self.predicted_labels = []
        #class_labels is a pandas series
        self.class_labels = None
        self.y_actual = []
        self.y_actual=[]

        self.X_train_tfidf = None
        self.y_ravel = None
        self.count_vect = None
        self.tfidf_transformer = TfidfTransformer()

        self.single_classifier = None
        self.mlb = None
        self.multi_classifier = None
        self.multi_class_predicted_i_t = None
        self.production_surveys = []

        self.current_single_label_model = None
        self.current_multi_label_model = None

        self.most_accurate_single_model = None
        self.most_accurate_multi_model = None
        self.best_accuracy = 0.0
        self.local_multiple_labels = 0
        self.best_multi_labels= 0

    def get_df(self):
        return self.df

    def set_df(self, csv_file_path):
        self.df = None
        self.df = pd.read_csv(csv_file_path, encoding = 'utf-8')

    def set_test(self, df):
        self.df_test = df

    def lemmatize(self,text):
        try:
            doc = nlp(str(text).decode('utf-8'))
        except:
            doc = nlp(u''.join(text))
        sent = []
        for word in doc:
            sent.append(word.lemma_)
        return ' '.join(sent)

    def spell_check(self,word1,word2):
        #TODO
        ed = editdistance.eval(word1,word2)
        return ed

    def clean_df(self, production=False, lemmatize = False):
        #make copy of comments column
        self.df['Original_Comments'] = self.df['Comments']
        if production == False:
            self.df.groupby('Reason').Comments.count().plot.bar(ylim=0)
            plt.suptitle('Class frequency in Data')
            plt.show()
            # shuffles the rows so that they're randomized
            self.df = self.df.sample(frac=1)
            self.df['Reason'] = self.df['Reason'].astype(str)
            # Uppercase class labels. There are duplicate class labels in different cases
            self.df['Reason'] = map(lambda x: x.upper(), self.df['Reason'])
            # remove surveys with 'NAN' label
            self.df = self.df[self.df['Reason'] != 'NAN']
        #lemmatize
        if lemmatize == True:
            self.df['Comments'] = self.df.apply(lambda x: self.lemmatize(x['Comments']), axis=1)

        #tell python that 'NAN' values are strings not float
        print('df shape is ',self.df.shape)
        #replace punctuation with ' '
        self.df['Comments'] = self.df['Comments'].str.replace('[.!?,;\-:()"]', ' ')
        #replace multiple white spaces with one space
        self.df['Comments'] = self.df['Comments'].str.replace('\s+', ' ')
        #replace numbers with 'X' in text, i.e 2016 = X, 9 = X
        self.df['Comments'] = self.df['Comments'].str.replace('\d+', 'X')
        #remove comments with 2 or less words
        self.df = self.df[self.df['Comments'].str.split().str.len() >= 2]
        print(self.df.head())
        self.df = self.df[pd.notnull(self.df['Comments'])]
        self.df.info()
        if production == False:
            col = ['Reason', 'Comments']
            self.df = self.df[col]

    def build_test_train_df(self, over_sample=False,
                            over_sample_rate=10, under_sample=False):
        #self.clean_df(training=True)
        #get last ~300 rows of dataframe for testing later
        self.y_traintext = []
        self.df_test = self.df.drop(self.df.index[0:4000])
        print(len(self.df_test.index))

        #building test text list
        self.X_test = np.asarray(self.df_test['Comments'])
        print(len(self.df.index))
        #df = first 4,000 rows of df
        self.df_train = self.df.drop(self.df.index[4000:len(self.df.index)])

        print('df shape is ',self.df_train.shape)

        labels = ['MONEY TRANSFERS', 'OTHER FINANCIAL SERVICE', 'VIRTUAL CURRENCY']
        labels1 = ['CONSUMER LOAN','PAYDAY LOAN','PREPAID CARD', 'STUDENT LOAN']

        if over_sample == True:
            self.label_count_visualize('Class Frequency Before Oversampling')
            self.over_sample(labels, over_sample_rate)
            self.over_sample(labels1, over_sample_rate / 10)
            self.label_count_visualize('Class Frequency After Oversampling')

        if under_sample == True:
            self.label_count_visualize('Class Frequency Before Undersampling')
            self.under_sample(labels,10)
            labels.extend(labels1)
            self.under_sample(labels, 2)
            self.label_count_visualize('Class Frequency After Undersampling')
        self.X_train = []
        self.X_train = np.asarray(self.df_train['Comments'])
        product = self.df_train['Reason'].tolist()
        print(self.X_train)
        print(product)
        for i in product:

            label = [i]
            self.y_traintext.append(label)

    def under_sample(self, labels,factor):
        print(len(self.df_train))
        min_df = self.df_train[self.df_train['Reason'].isin(labels)]
        maj_df = self.df_train[~self.df_train['Reason'].isin(labels)]
        start = len(maj_df)/factor
        stop = len(maj_df)
        print('stop:', stop)
        maj_df = maj_df.drop(maj_df.index[start:stop])
        print(len(maj_df))
        print(maj_df)
        #append both dfs
        self.df_train = min_df.append(maj_df)
        #shuffles the rows so that they're randomized, frac=1 means that you return 100% of df
        self.df_train = self.df_train.sample(frac=1)
        print(len(self.df_train))

    def over_sample(self, labels, over_sample_rate):
        over_sample = self.df_train[self.df_train['Reason'].isin(labels)]
        k = 0
        while k < over_sample_rate-1:
            self.df_train = self.df_train.append(over_sample)
            k+=1
        print(over_sample)

    def label_count_visualize(self, title):
        #visualize  label metrics in training set
        self.df_train.groupby('Reason').Comments.count().plot.bar(ylim=0)
        plt.suptitle(title)
        plt.show()

    def feature_extraction(self, verbose=False):
        self.tfidf_transformer = TfidfTransformer()
        self.count_vect = CountVectorizer(ngram_range=(1, 3), stop_words='english',
                                          min_df=2)
        X_train_counts = self.count_vect.fit_transform(self.X_train)
        #print(X_train_counts.shape)
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        #print(self.X_train_tfidf.shape)
        self.y_ravel = np.asarray(self.y_traintext).ravel()
        # print extracted features in training set
        if verbose == True:
            self.df_train['category_id'] = self.df_train['Reason'].factorize()[0]
            category_id_df = self.df_train[['Reason', 'category_id']].drop_duplicates().sort_values('category_id')
            category_to_id = dict(category_id_df.values)
            print('starting feature extraction')
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2,
                                   norm='l2',encoding='latin-1', ngram_range=(1, 3))
            features = tfidf.fit_transform(self.df_train.Comments).toarray()
            feature_labels = self.df_train.category_id
            print('features shape is ', features.shape)
            N = 10
            for Reason, category_id in sorted(category_to_id.items()):
                features_chi2 = chi2(features, feature_labels == category_id)
                indices = np.argsort(features_chi2[0])
                indices1 = np.argsort(features_chi2[0 + 1])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                feature_names1 = np.array(tfidf.get_feature_names())[indices1]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
                print("# '{}':".format(Reason))
                print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
                print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
                print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))

    def get_tfidf_transformer(self):
        return self.tfidf_transformer
    def set_tfidf_transformer(self,tfidf_transformer):
        self.tfidf_transformer = tfidf_transformer

    def transform_df(self,multi_model = LinearSVC(),
                     single_model=RandomForestClassifier(n_estimators=50, max_depth=2,
                     random_state=0, max_features=None)):
        self.feature_extraction()
        self.current_single_label_model = single_model
        self.current_multi_label_model = multi_model
        clf = single_model
        self.single_classifier = clf.fit(self.X_train_tfidf, self.y_ravel)
        X_new_counts = self.count_vect.transform(self.X_test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        single_class_predicted = self.single_classifier.predict(X_new_tfidf)

        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform(self.y_traintext)

        #multi-label classifier
        self.multi_classifier = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1,2), stop_words='english',
                                           min_df = 2)),
            ('tfidf', TfidfTransformer()),
            ('clf',OneVsRestClassifier(multi_model) )])

        # print multi classification labels
        self.multi_classifier.fit(self.X_train, Y)
        multi_class_predicted = self.multi_classifier.predict(self.X_test)
        self.multi_class_predicted_i_t = self.mlb.inverse_transform(multi_class_predicted)

        self.y_actual = self.df_test['Reason'].tolist()

        #resetting predicted_labels
        self.predicted_labels = []
        self.local_multiple_labels = 0
        for multi_labels, single_label in zip(self.multi_class_predicted_i_t,single_class_predicted):
            #if multi-classifier labels length is greater than 0, than use the multi-label
            #multi-classifier doesn't always return a label
            #else use the single classifier label
            if(len(multi_labels) > 0):
                label = '{0}'.format('=>'.join(multi_labels))
                self.predicted_labels.append(label)
                if(len(multi_labels) > 1):
                    self.local_multiple_labels+=1
            else:
                label = single_label
                self.predicted_labels.append(label)

    def get_count_vect(self):
        return self.count_vect
    def set_count_vect(self,count_vect):
        self.count_vect = count_vect

    def get_single_classifier(self):
        return self.single_classifier
    def set_single_classifier(self,single_classifier):
        self.single_classifier=single_classifier

    def get_multi_classifier(self):
        return self.multi_classifier
    def set_multi_classifier(self,multi_classifier):
        self.multi_classifier = multi_classifier

    def get_multi_class_predicted_i_t(self):
        return self.multi_class_predicted_i_t
    def set_multi_class_predicted_i_t(self,multi_class_predicted_i_t):
        self.multi_class_predicted_i_t=multi_class_predicted_i_t

    def get_mlb(self):
        return self.mlb
    def set_mlb(self,mlb):
        self.mlb = mlb

    def avg_recall(self, recall_list,support_list):
        sum=0.0
        total_cases = 0
        for recall,support in zip( recall_list,support_list):
            sum+=recall*support
            total_cases+=support
        avg = 100*(sum/total_cases)
        return avg

    def confusion_matrix(self, verbose = True):
        self.class_labels = pd.Series(np.unique(self.y_actual))
        #print metrics report
        cr = metrics.classification_report(self.y_actual, self.predicted_labels,labels = self.class_labels,
                                            #target_names=df_test['Reason'].unique()))
                                            target_names=list(self.class_labels))
        print(cr)
        print("multiple_labels: "+str(self.local_multiple_labels))
        if verbose:
            cm = confusion_matrix(self.y_actual, self.predicted_labels, labels=self.class_labels)
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            plt.subplots_adjust(top=0.98, bottom=0.47, left=0.27, right=0.99, hspace=0.20,
                                wspace=0.20)
            plt.suptitle('Confusion Matrix')
            plt.show()

    def multi_label_metric(self, verbose=True,best_model_metric='accuracy'):
        self.confusion_matrix(verbose=verbose)
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.y_actual,
                                                                                     self.predicted_labels,
                                                                                     labels=self.class_labels)
        local_avg_recall = self.avg_recall(recall, support)
        print('local accuracy: ' + str(local_avg_recall))
        if best_model_metric == 'accuracy':
            if self.best_accuracy < local_avg_recall:
                self.best_accuracy = local_avg_recall
                self.most_accurate_single_model = self.current_single_label_model
                self.most_accurate_multi_model = self.current_multi_label_model
            print('Most accurate single model: ' + str(self.most_accurate_single_model))
            print('Most accurate multi model: ' + str(self.most_accurate_multi_model))
            print('Best accuracy: ' + str(self.best_accuracy))
        if best_model_metric == 'labels':
            if self.best_multi_labels < self.local_multiple_labels:
                self.best_multi_labels = self.local_multiple_labels
                self.most_accurate_single_model = self.current_single_label_model
                self.most_accurate_multi_model = self.current_multi_label_model
            print('Most label single model: ' + str(self.most_accurate_single_model))
            print('Most label multi model: ' + str(self.most_accurate_multi_model))
            print('Most labels: ' + str(self.best_multi_labels))

    def print_write(self,out_file_path,verbose=True):
        #zip both classifiers and print to file
        f=open(out_file_path,'w')
        for  actual_label, predicted_label,  item in zip(self.y_actual,self.predicted_labels,self.X_test):
            #the console won't recognize special utf-8 characters without encoding
            actual_label = actual_label.encode('utf-8')
            #replace commas in text with semicolon. writing to csv file later so we don't want extra commas in text
            item = item.replace(",", ";")
            predicted_label = predicted_label.encode('utf-8')
            try:
                line = actual_label+','+predicted_label+','+item
            except:
                line = actual_label + ',' + predicted_label + ',' + u''.join(item)
            if verbose:
                print(line)
            try:
                f.write(line+ '\n')
            except:
                f.write((line + '\n').encode('ascii', 'ignore'))
        f.close()

    def label_for_production(self):
        production_df = self.get_df()
        production_df = production_df[['Comments']]
        #Drop the rows where all of the elements are nan
        production_df = production_df.dropna(axis=0, how='all')
        print(production_df)
        self.production_surveys = np.asarray(production_df['Comments'])
        X_new_counts = self.count_vect.transform(self.production_surveys)

        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        single_class_predicted = self.single_classifier.predict(X_new_tfidf)

        multi_class_predicted = self.multi_classifier.predict(self.production_surveys)
        multi_class_predicted_i_t = self.mlb.inverse_transform(multi_class_predicted)

        for multi_labels, single_label in zip(multi_class_predicted_i_t, single_class_predicted):
            # if multi-classifier labels length is greater than 0, than use the multi-label
            # multi-classifier doesn't always return a label
            # else use the single classifier label
            #self.predicted_labels = []
            #self.local_multiple_labels = 0
            if (len(multi_labels) > 0):
                label = list(multi_labels)
                self.predicted_labels.append(label)
                if (len(multi_labels) > 1):
                    self.local_multiple_labels += 1
            else:
                label = [single_label]
                self.predicted_labels.append(label)

    def production_write(self,out_file_path):
        original_comments = np.asarray(self.df['Original_Comments'])
        f = open(out_file_path, 'w')
        f.write('Reason,Comments,Original_Comments\n')
        for  predicted_label, item, oc in zip(self.predicted_labels, self.production_surveys,original_comments):
            item = item.replace(",", ";")
            oc = oc.replace(",", ";")
            item = item.encode('utf-8')
            #predicted_label = predicted_label.encode('utf-8')
            predicted_label = [x.encode('utf-8') for x in predicted_label]
            line = predicted_label + ',' + item+','+oc
            print(line)
            f.write(line + '\n')
        f.close()

    def production_write_df(self,out_file_path):
        predicted_labels = np.asarray(self.predicted_labels)
        print(predicted_labels)
        self.df['Reason'] = predicted_labels
        self.df = self.flatten_labels(self.df, 'Reason')
        self.df.to_csv(path_or_buf=out_file_path, sep=str(','), encoding='utf-8')

    def flatten_labels(self, df, column, reset_index=True):
        b_flat = pd.DataFrame([[i, x]
                               for i, y in df[column].apply(list).iteritems()
                               for x in y], columns=['I', column])
        b_flat = b_flat.set_index('I')
        df = df.merge(b_flat, left_index=True, right_index=True)
        df = df.rename(columns={column + '_y': column, column + '_x': column + '_original'})
        return df