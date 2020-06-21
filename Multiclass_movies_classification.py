import pandas as pd
import numpy as np
from collections import Counter
import re
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.metrics import  confusion_matrix, classification_report


#############################        Część 1: import i obróbka dancyh      #############################

#import danych
movies=pd.read_csv("movies.csv",usecols=['movieId','title','genres'])
ratings=pd.read_csv("ratings.csv", usecols=['movieId','userId','rating'])
tags=pd.read_csv("tags.csv", usecols=['movieId','userId','tag'])
tag_relev=pd.read_csv("genome-scores.csv", usecols=['movieId','tagId','relevance'])
tagsIds=pd.read_csv("genome-tags.csv", usecols=['tagId','tag'])
tags=pd.read_csv("tags.csv", usecols=['movieId','userId','tag'])

#łączenie danych w jedną tabelę
md=pd.merge(ratings,movies,on='movieId')
md=pd.merge(md,tags,on=['movieId','userId'])
md=pd.merge(md,tagsIds,on='tag')
md=pd.merge(md,tag_relev,on=['movieId','tagId'])

#sprawdzamy ile filmów ocenił każdy użytkownik
c=Counter(md['userId'])
id,coun = list(c.keys()),list(c.values())

#szukamy userIds którzy ocenili mniej niż 1000 filmów 
#(krok potrzebny, aby algorytmy wykonały się w rozsądnym czasie, przy liczbie wierszy ponad 180 000 trwało bardzo długo)
a=list(map(list, zip(id,coun)))
filtr = [x for x in a if x[1] <1000]
list_to_drop=[x[0] for x in filtr]
print(len(list_to_drop))

#usuwamy użytkowników z mała iloscią filmów
for i in range(0,len(list_to_drop)):
    md=md.drop(md.loc[md['userId']==list_to_drop[i]].index)

#sprawdzamy czy usunęło się wszystko
c=Counter(md['userId'])
id,coun = list(c.keys()),list(c.values())
a=list(map(list, zip(id,coun)))
filtr = [x for x in a if x[1] <10]
list_to_drop=[x[0] for x in filtr]
print(list_to_drop)

#zaokrągalmy oceny filmów w dół (redukcja klas do 6)
md['rating'] = np.floor(md['rating'])

#z tytułu wyciagamy rok premiery
#niestety sposób z wycięciem znaków nie zadziałał, ponieważ w kilku tytułach po ostatnim nawiasie pojawia się jeszcze jeden znak, 
#przez co w wyniku otrzymujemy 007) zamiast 2007
#re.findall zwreaca wszystkie ciągi znaków spełniający podany schemat - tutaj liczba czterocyfrowa 
new = []
for row in md['title'].values:
    new.append(re.findall('(\d{4})', str(row))[-1])
md['year'] = new

#zliczamy unikatowe wartosci w kazdej kolumnie
for col in md.columns:
    c=Counter(md[col])
    id,coun = list(c.keys()),list(c.values())
    print("Unikatowych", col, ":",len(id))

#uzupełnianie brakujących danych (okazuje się ze nic nie brakuje)
for col in md.columns:
    print('Brakujących ', col, ': ', md[col].isnull().sum())

#ustalenie typu danych
md = md.astype({"userId": int, "movieId": int, "tagId": int, "relevance": float,"rating": int, "year": int})

#kolumnę z klasą (rating) wydzielamy to wektora y, pozostałe dane do tabeli X
y = md['rating'].values
X = md.drop(columns=['rating'])

#liczebnosci klas
uni, count = np.unique(y, return_counts=True)
print('Podzial na klasy\n', dict(zip(uni, count)))

#zapis danych po obróbce do pliku (nie trzeba odpalać od nowa całego kodu)
#np.save('label',y)
#np.save('dane',X)


#############################        Częsć 2: Kodowanie, tworzenie potoku, podział na zbiory             #############################

#kodowanie tekstu, tworzenie potoku
pipe_num=Pipeline([('inp',SimpleImputer(missing_values=np.nan, strategy='median')),('scl', StandardScaler())])
pipe_nom=Pipeline([('ohe',OneHotEncoder())])
colnum=["userId","movieId","tagId","relevance","year"]
colnom=["title",'genres','tag']
preprocessor = ColumnTransformer( transformers=[('num', pipe_num, colnum),('cat', pipe_nom, colnom)])
scX=preprocessor.fit_transform(X)
print(scX)

#rozdzielamy zbior na treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(scX, y, test_size=0.2, random_state=1, stratify=y)

#liczebosc każdej klasy w zbiorach
uni, count = np.unique(y_test, return_counts=True)
print('Liczba ocen w zbiorze testowym\n', dict(zip(uni, count)))
uni, count = np.unique(y_train, return_counts=True)
print('Liczba ocen w zbiorze treningowym\n', dict(zip(uni, count)))

#############################        Częsć 3: tworzenie i ocena modeli                     #############################

#Metoda k najbliższych sąsiadów - szukamy najlepszego k kroswalidacją
knn2 = KNeighborsClassifier()
#słownik wartosci k ktore przetestujemy
param_grid = {'n_neighbors': np.arange(2, 10)}

#szukamy najlepszego k
a=time.time()
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X_train, y_train)
print('Czas kroswalidacji KNN: ', time.time()-a)
#sprawdzamy jaka wartosc parametru k dala najlepsze wyniki
print(knn_gscv.best_params_)
bestk=knn_gscv.best_params_.get('n_neighbors')
print(bestk)

#tworzenie modelu z najlepszym k
b=time.time()
knn = KNeighborsClassifier(n_neighbors = bestk)
knn.fit(X_train,y_train)
print('Czas KNN: ', time.time()-b)
y_pred=knn.predict(X_test)

#ocena
classrep = classification_report(y_test, y_pred)
print(classrep)
cmm = confusion_matrix(y_test, y_pred)
print(cmm)



#Drzewa decyzyjne
tree=DecisionTreeClassifier()

#wartości max_depth do testowania
param_grid = {'max_depth': np.arange(2, 10)}

#testowanie powyższych wartości
c=time.time()
tree_gscv = GridSearchCV(tree, param_grid, cv=5)
tree_gscv.fit(X_train, y_train)
print('Czas kroswalidacji drzew decyzyjnych: ', time.time()-c)
#najlepsza wartość
print(tree_gscv.best_params_)
bestdepth=tree_gscv.best_params_.get('max_depth')
print(bestdepth)

#klasyfikator z najlepszym max_depth
d=time.time()
tree2 = DecisionTreeClassifier(criterion='entropy',max_depth=bestdepth)
tree2.fit(X_train,y_train)
print('Czas drzew decyzyjnych: ', time.time()-d)
y_pred=tree2.predict(X_test)

#ocena
classrep = classification_report(y_test, y_pred)
print(classrep)
cmm = confusion_matrix(y_test, y_pred)
print(cmm)


#Lasy losowe

forest = RandomForestClassifier()
#wartości n_estimators do testowania
param_grid = {'n_estimators': np.arange(2, 10)}

#testowanie powyższych wartości
e=time.time()
forest_gscv = GridSearchCV(forest, param_grid, cv=5)
forest_gscv.fit(X_train, y_train)
print('Czas kroswalidacji lasów: ', time.time()-e)
#najlepsza wartość
print(forest_gscv.best_params_)
best=forest_gscv.best_params_.get('n_estimators')
print(best)

#klasyfikator z najlepszym paramtrem
f=time.time()
forest2 = RandomForestClassifier(criterion='entropy',n_estimators=best)
forest2.fit(X_train,y_train)
print('Czas lasów: ', time.time()-f)
y_pred=forest2.predict(X_test)

#ocena
classrep = classification_report(y_test, y_pred)
print(classrep)
cmm = confusion_matrix(y_test, y_pred)
print(cmm)


#Adaboost z Naiwnym klasyfikatorem Bayesowskim
GNB=GaussianNB()
g=time.time()
ada_clf = AdaBoostClassifier(GNB, n_estimators=20, algorithm='SAMME.R', learning_rate=1)
ada_clf.fit(X_train.toarray(), y_train)
print('Czas AdaBoost z GNB: ', time.time()-g)
y_pred=ada_clf.predict(X_test.toarray())

#ocena
classrep = classification_report(y_test, y_pred)
print(classrep)
cmm = confusion_matrix(y_test, y_pred)
print(cmm)
