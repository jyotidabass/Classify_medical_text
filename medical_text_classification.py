Step 1: Loading the dataset in jupyter
We have been given train data to train our model and our first step would be to load this data. Following commands are used to load your data.
        import pandas as pd
        traindata = pd.read_csv("train.dat",sep='\t',header=None) // loading data in traindata print(traindata) // printing data to see(optional)

Step 2: Creating Vectors for us as CountVectorizer.
Our data is a series of words. We will be using bag of words model to convert text files
into numerical feature vectors.
        from sklearn.feature_extraction.text import CountVectorizer count_vect = CountVectorizer()
        train_counts = count_vect.fit_transform(traindata[1]) train_counts.shape
Step 3: Reducing the weightage of common words, so that these common words should not a affect our result.
        from sklearn.feature_extraction.text import TfidfTransformer tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_counts) train_tfidf.shape
Step 4: Running Machine Learning Algorithms and train our NB classifier on training data We have used Naive Bayes algorithm for this assignment
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(train_tfidf, traindata[0])
        Step 5: Now we will use pipeline from sklearn to build our model
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
        text_classifier = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        (svc, LinearSVC()),])
        text_classifier = text_classifier.fit(traindata[1], traindata[0])
Step 6: Now we have build our model and will test this model on our test data and test its efficiency. To load our data we are doing the same steps as done for the load data
        import nltk
        import csv
        t = open("test.dat")
        predicted = text_classifier.predict(t)
Step 7: Predicted the result and uses dataframe and pandas library to build it 
        result = pd.DataFrame(predicted)
Step 8: Saved the predicted result in ‘ result_predicted_svcdata.dat’  to compare with the ‘format.dat’
        result.to_csv('result_predicted_svcdata.dat',index=False,header=None)
