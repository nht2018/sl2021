

import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import csv
import time
from plot import Plot
# Import Datasets


def CountVect():

    PATH = 'E:/sl2021Project/protein data'
    df_seq = pd.read_csv(PATH+'/2.csv')
    df_char = pd.read_csv(PATH+'/1.csv')

    #print('Datasets have been loaded...')
    # 2). ----- Filter and Process Dataset ------

    # Filter for only proteins
    protein_char = df_char[df_char.macromoleculeType == 'Protein']
    protein_seq = df_seq[df_seq.macromoleculeType == 'Protein']

    # Select only necessary variables to join
    protein_char = protein_char[['structureId', 'classification']]
    protein_seq = protein_seq[['structureId', 'sequence']]
    protein_seq.head()
    protein_char.head()
    model_f = protein_char.set_index('structureId').join(
        protein_seq.set_index('structureId'))
    model_f.head()

    model_f.isnull().sum()
    model_f = model_f.dropna()
    #print('%d rows in updated joined dataset' %model_f.shape[0])
    counts = model_f.classification.value_counts()

    types = np.asarray(counts[(counts > 1000)].index)

    #data = model_f[model_f.classification.isin(types)]
    data = model_f.loc[(model_f.classification == 'HYDROLASE')
                       | (model_f.classification == 'TRANSFERASE')]
    #print('%d filtered dataset' %data.shape[0])
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data['sequence'], data['classification'], test_size=0.2, random_state=1)

    # Exploring CountTokenizer

    # list of text documents
    '''
    text = ["The quick brown fox jumped over the lazy dog."]
    # create the transform
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(text)
    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())
    '''
    vect = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4))
    # Fit and Transform CountVectorizer
    vect.fit(X_train)
    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)
    prediction = dict()
    timee = []
    # Naive Bayes Model
    time_start = time.time()
    model = MultinomialNB()
    model.fit(X_train_df, y_train)
    NB_pred = model.predict(X_test_df)
    prediction["MultinomialNB"] = accuracy_score(NB_pred, y_test)
    time_end = time.time()
    print("MultinomialNB with accuuracy ",
          prediction['MultinomialNB'], ' and runs in ', time_end-time_start)
    timee.append(time_end-time_start)
    time_start = time.time()

    clf_svc = svm.SVC(gamma=0.001, C=100)  # experiment with one-vs-one
    clf_svc.fit(X_train_df, y_train)
    NB_pred = clf_svc.predict(X_test_df)

    prediction["SVM_SVC"] = accuracy_score(y_test, NB_pred)
    time_end = time.time()
    timee.append(time_end-time_start)
    # the plot process does not work well, so the following codes are unuseful
    '''
    X = np.array(X_test_df.todense())
    y = []
    for item in y_test:
        if item == 'HYDROLASE':
            y.append(1)
        else:
            y.append(0)
    Plot(X=X, Y=y)
    '''
    print("SVM_SVC       with accuuracy ", prediction['SVM_SVC'],
          ' and runs in ', time_end-time_start)

    time_start = time.time()
    clf = svm.LinearSVC(random_state=1, C=100)  # experiment with one-vs-rest
    clf.fit(X_train_df, y_train)
    NB_pred = clf.predict(X_test_df)
    # print(classification_report(y_test,NB_pred))
    prediction["SVM_LINEARSVC"] = accuracy_score(y_test, NB_pred)
    print(prediction['SVM_LINEARSVC'])
    time_end = time.time()
    timee.append(time_end-time_start)

    return prediction['MultinomialNB'], prediction['SVM_SVC'], timee[0], timee[1], prediction['SVM_LINEARSVC'], timee[2]


'''
    # test prediction

    a = ['MHIPEGYLSPQTCAVMGAAMVPVLTVAAKKVNKSFDKKDVPAMAIGSAFAFTIMMFNVPIPGGTTAHAIGATLLATTLGPWAASISLTLALFIQALLFGDGGILALGANSFNMAFIAPFVGYGIYRLMLSLKLNKVLSSAIGGYVGINAAALATAIELGLQPLLFHTANGTPLYFPYGLNVAIPAMMFAHLTVAGIVEAVITGLVVYYLLEHHHHHH']

    #print('predicted : ', clf.predict(vect.transform(a)))

    #print('training data : ', model_f.loc[model_f.sequence=='MHIPEGYLSPQTCAVMGAAMVPVLTVAAKKVNKSFDKKDVPAMAIGSAFAFTIMMFNVPIPGGTTAHAIGATLLATTLGPWAASISLTLALFIQALLFGDGGILALGANSFNMAFIAPFVGYGIYRLMLSLKLNKVLSSAIGGYVGINAAALATAIELGLQPLLFHTANGTPLYFPYGLNVAIPAMMFAHLTVAGIVEAVITGLVVYYLLEHHHHHH'])

    scores = cross_val_score(clf, X_train_df, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores

    scores = cross_val_score(clf_svc, X_train_df, y_train, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print(scores)
    scores = cross_val_score(model, X_train_df, y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores
'''


def filebigsmall(bigsmall=5000):
    csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_no_dups.csv',
                    mode='r', encoding='utf-8')
    reader = csv.reader(csv_file)
    num = 0
    f = open('E:\\sl2021Project\\protein data\\1.csv',
             'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    for item in reader:
        if(item):
            num = num+1
            writer.writerow(item)
        if(num > bigsmall):
            break
    f.close()
    csv_file.close()
    csv_file = open('E:\\sl2021Project\\protein data\\pdb_data_seq.csv',
                    mode='r', encoding='utf-8')
    reader = csv.reader(csv_file)
    num = 0
    f = open('E:\\sl2021Project\\protein data\\2.csv',
             'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    for item in reader:
        if(item):
            num = num+1
            writer.writerow(item)
        if(num > bigsmall):
            break
    f.close()
    csv_file.close()


lst1, lst2, lst3, lst4, lst5, lst6 = [], [], [], [], [], []
lst = [i+1 for i in range(40)]
for i in range(40):
    filebigsmall(1000+1000*i)
    a1, a2, a3, a4, a5, a6 = CountVect()
    lst1.append(a1)
    lst2.append(a2)
    lst3.append(a3)
    lst4.append(a4)
    lst5.append(a5)
    lst6.append(a6)
plt.figure()
plt.plot(lst, lst1, label="MultinomialNB", linestyle=":")
plt.plot(lst, lst5, label="SVM", linestyle="--")
plt.plot(lst, lst2, label="OurModel", linestyle="-.")
plt.legend()
plt.title("accuracy")
plt.xlabel('size/1000')
plt.ylabel("accuracy")
plt.savefig('E:/sl2021Project/accurracy')
plt.figure()
plt.plot(lst, lst3, label="MultinomialNB", linestyle=":")
plt.plot(lst, lst6, label="SVM", linestyle="--")
plt.plot(lst, lst4, label="OurModel", linestyle="-.")
plt.legend()
plt.xlabel('size/1000')
plt.ylabel('time/s')
plt.savefig('E:/sl2021Project/time')
