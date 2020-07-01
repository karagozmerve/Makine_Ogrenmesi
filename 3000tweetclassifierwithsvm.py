# -*- coding: utf-8 -*-
"""
@author: Merve
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


sentences_training = []
classification_training = []

path='3000tweet/raw_texts/**/*.txt'

for sayi, tweetdosyasi in enumerate(glob(path, recursive=True)): 
    classification_training.append(tweetdosyasi.split("\\")[1]),
    sentences_training.append((open(tweetdosyasi, encoding="windows-1254").read().replace('\n', ' '))) 

print(sentences_training[0])


vectorizer = TfidfVectorizer(analyzer = "word", lowercase = True)
sent_train_vector = vectorizer.fit_transform(sentences_training)

print(sent_train_vector)
# TF-IDF değeri bulunmaya çalışıldı.
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(sentences_training)
# idf = vectorizer._tfidf.idf_
# print(idf)
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(sentences_training)
# tf = vectorizer.tfidf.tf
# print(tf)
x_train, x_test, y_train, y_test = train_test_split(sent_train_vector.toarray(),classification_training, test_size=0.20)  


svm = SVC(kernel='poly')
svm.fit(x_train, y_train)
prediction = svm.predict(x_test)
print("SVM (kernel=poly) Accuracy :", accuracy_score(y_test, prediction))  
print('SVM (kernel=poly) Clasification: \n', classification_report(y_test, prediction))  
print('SVM (kernel=poly) Confussion matrix: \n', confusion_matrix(y_test, prediction,labels=['1','2','3']))  
print("\n")
print("\n")

# mat = confusion_matrix(y_test, prediction)
# names = np.unique(prediction)
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names, yticklabels=names)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.2 iken kernel=poly ')
# plt.show()
# svm2 = SVC(kernel='rbf')
# svm2.fit(x_train, y_train)
# prediction2 = svm2.predict(x_test)
# print("SVM (kernel=rbf) Accuracy :", accuracy_score(y_test, prediction2))  
# print('SVM (kernel=rbf) Clasification: \n', classification_report(y_test, prediction2))  
# print('SVM (kernel=rbf) Confussion matrix: \n', confusion_matrix(y_test, prediction2,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat2 = confusion_matrix(y_test, prediction2)
# names2 = np.unique(prediction2)
# sns.heatmap(mat2, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names2, yticklabels=names2)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.2 iken kernel=rbf')
# plt.show()
# svm3 =  SVC(kernel='linear')
# svm3.fit(x_train, y_train)
# prediction3 = svm3.predict(x_test)
# print("SVM (kernel=linear) Accuracy :", accuracy_score(y_test, prediction3))  
# print('SVM (kernel=linear) Clasification: \n', classification_report(y_test, prediction3))  
# print('SVM (kernel=linear) Confussion matrix: \n', confusion_matrix(y_test, prediction3,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat3 = confusion_matrix(y_test, prediction3)
# names3 = np.unique(prediction3)
# sns.heatmap(mat3, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names3, yticklabels=names3)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.2 iken kernel=linear')
# plt.show()

# label=['kernel=poly','kernel=rbf','kernel=linear']
# index=np.arange(len(label))
# acc=[accuracy_score(y_test, prediction),accuracy_score(y_test, prediction2),accuracy_score(y_test, prediction3)]
# plt.bar(index,acc)
# plt.xlabel('SVM Classifier',fontsize=15)
# plt.ylabel('Accuracy',fontsize=15)
# plt.title('TEST SİZE=0.2 SVM ',fontsize=20)
# plt.xticks(index,label,fontsize=10,rotation=0)
# plt.show()
# deneme=vectorizer.transform(["turkcell ıyıdır :)"])
# prediction = svm.predict(deneme.toarray())
# print(prediction)
# deneme2=vectorizer.transform(["ilk fırsatta hattımı iptal ettireceğim."])
# prediction2 = svm.predict(deneme.toarray())
# print(prediction2)
# deneme3=vectorizer.transform(["ayyyyyyyyyyyyyyyyyyyy &lt;3"])
# prediction3 = svm.predict(deneme.toarray())
# print(prediction3)
# x_train, x_test, y_train, y_test = train_test_split(sent_train_vector.toarray(),classification_training, test_size=0.5)  


# svm = SVC(kernel='poly')
# svm.fit(x_train, y_train)
# prediction = svm.predict(x_test)
# print("SVM (kernel=poly) Accuracy :", accuracy_score(y_test, prediction))  
# print('SVM (kernel=poly) Clasification: \n', classification_report(y_test, prediction))  
# print('SVM (kernel=poly) Confussion matrix: \n', confusion_matrix(y_test, prediction,labels=['1','2','3']))  
# print("\n")
# print("\n")

# mat = confusion_matrix(y_test, prediction)
# names = np.unique(prediction)
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names, yticklabels=names)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.5 iken kernel=poly ')
# plt.show()
# svm2 = SVC(kernel='rbf')
# svm2.fit(x_train, y_train)
# prediction2 = svm2.predict(x_test)
# print("SVM (kernel=rbf) Accuracy :", accuracy_score(y_test, prediction2))  
# print('SVM (kernel=rbf) Clasification: \n', classification_report(y_test, prediction2))  
# print('SVM (kernel=rbf) Confussion matrix: \n', confusion_matrix(y_test, prediction2,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat2 = confusion_matrix(y_test, prediction2)
# names2 = np.unique(prediction2)
# sns.heatmap(mat2, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names2, yticklabels=names2)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.5 iken kernel=rbf')
# plt.show()
# svm3 =  SVC(kernel='linear')
# svm3.fit(x_train, y_train)
# prediction3 = svm3.predict(x_test)
# print("SVM (kernel=linear) Accuracy :", accuracy_score(y_test, prediction3))  
# print('SVM (kernel=linear) Clasification: \n', classification_report(y_test, prediction3))  
# print('SVM (kernel=linear) Confussion matrix: \n', confusion_matrix(y_test, prediction3,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat3 = confusion_matrix(y_test, prediction3)
# names3 = np.unique(prediction3)
# sns.heatmap(mat3, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names3, yticklabels=names3)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.5 iken kernel=linear')
# plt.show()

# label=['kernel=poly','kernel=rbf','kernel=linear']
# index=np.arange(len(label))
# acc=[accuracy_score(y_test, prediction),accuracy_score(y_test, prediction2),accuracy_score(y_test, prediction3)]
# plt.bar(index,acc)
# plt.xlabel('SVM Classifier',fontsize=15)
# plt.ylabel('Accuracy',fontsize=15)
# plt.title('TEST SİZE=0.5 SVM ',fontsize=20)
# plt.xticks(index,label,fontsize=10,rotation=0)
# plt.show()

# #TEST SİZE=0.7 İKEN
# x_train, x_test, y_train, y_test = train_test_split(sent_train_vector.toarray(),classification_training, test_size=0.7)  


# svm = SVC(kernel='poly')
# svm.fit(x_train, y_train)
# prediction = svm.predict(x_test)
# print("SVM (kernel=poly) Accuracy :", accuracy_score(y_test, prediction))  
# print('SVM (kernel=poly) Clasification: \n', classification_report(y_test, prediction))  
# print('SVM (kernel=poly) Confussion matrix: \n', confusion_matrix(y_test, prediction,labels=['1','2','3']))  
# print("\n")
# print("\n")

# mat = confusion_matrix(y_test, prediction)
# names = np.unique(prediction)
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names, yticklabels=names)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.7 iken kernel=poly ')
# plt.show()
# svm2 = SVC(kernel='rbf')
# svm2.fit(x_train, y_train)
# prediction2 = svm2.predict(x_test)
# print("SVM (kernel=rbf) Accuracy :", accuracy_score(y_test, prediction2))  
# print('SVM (kernel=rbf) Clasification: \n', classification_report(y_test, prediction2))  
# print('SVM (kernel=rbf) Confussion matrix: \n', confusion_matrix(y_test, prediction2,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat2 = confusion_matrix(y_test, prediction2)
# names2 = np.unique(prediction2)
# sns.heatmap(mat2, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names2, yticklabels=names2)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.7 iken kernel=rbf')
# plt.show()
# svm3 =  SVC(kernel='linear')
# svm3.fit(x_train, y_train)
# prediction3 = svm3.predict(x_test)
# print("SVM (kernel=linear) Accuracy :", accuracy_score(y_test, prediction3))  
# print('SVM (kernel=linear) Clasification: \n', classification_report(y_test, prediction3))  
# print('SVM (kernel=linear) Confussion matrix: \n', confusion_matrix(y_test, prediction3,labels=['1','2','3']))  
# print("\n")
# print("\n")
# mat3 = confusion_matrix(y_test, prediction3)
# names3 = np.unique(prediction3)
# sns.heatmap(mat3, square=True, annot=True, fmt='d', cbar=True,
#             xticklabels=names3, yticklabels=names3)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('Test size=0.7 iken kernel=linear')
# plt.show()

# label=['kernel=poly','kernel=rbf','kernel=linear']
# index=np.arange(len(label))
# acc=[accuracy_score(y_test, prediction),accuracy_score(y_test, prediction2),accuracy_score(y_test, prediction3)]
# plt.bar(index,acc)
# plt.xlabel('SVM Classifier',fontsize=15)
# plt.ylabel('Accuracy',fontsize=15)
# plt.title('TEST SİZE=0.7 SVM ',fontsize=20)
# plt.xticks(index,label,fontsize=10,rotation=0)
# plt.show()