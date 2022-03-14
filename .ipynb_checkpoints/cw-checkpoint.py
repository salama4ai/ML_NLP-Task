from __future__ import print_function
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import argparse
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train','-t', help='model file')

    args = parser.parse_args()

    trainData  = pd.read_csv(args.train,sep='\t',header=None)
    #read the data in
    train = pd.DataFrame(trainData.iloc[:, 0:2])
    train.columns=['label', 'tweet']

    lb = LabelEncoder().fit(train['label'])
    lb.transform(train['label'])


    X_train_part, X_valid, y_train_part, y_valid =\
        train_test_split(train['tweet'], 
                         lb.transform(train['label']), 
                    test_size=0.1,random_state=17, stratify=train['label'])

    pipeline = Pipeline([
        ('u1', FeatureUnion([
            ('word_features', Pipeline([
                ('ngramw', CountVectorizer(ngram_range=(2, 6), analyzer='word')),
                ('tfidf', TfidfTransformer()),
            ])),
            ('char_features', Pipeline([
                ('ngramc', CountVectorizer(ngram_range=(2, 6), analyzer='char')),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('logit', LogisticRegression()),

    ])


    model = pipeline.fit(X_train_part, y_train_part)
    modelFile = 'model_'+args.train.replace('.txt','').split('/')[-1]+'_FeatureUnion.sav'
    joblib.dump(model, modelFile)
    eprint("Saving model:"+modelFile)


    loaded_model = joblib.load(modelFile)

    pred = loaded_model.predict(X_valid)

    from sklearn import metrics
    print(metrics.classification_report(y_valid, pred, digits=3))

if __name__ == "__main__":
    main()

