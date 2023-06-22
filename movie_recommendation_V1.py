'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 2 Building A Movie Recommendation Engine with Naive Bayes

'''
import numpy as np
from collections import defaultdict

'''
We then develop the following function to load the rating data from ratings.dat
'''
def load_rating_data(data_path, n_users, n_movies):
    """
    Load rating data from file and also return the number of ratings for each movie and movie_id index mapping
    @param data_path: path of the rating data file
    @param n_users: number of users
    @param n_movies: number of movies that have ratings
    @return: rating data in the numpy array of [user, movie]; movie_n_rating, {movie_id: number of ratings};
             movie_id_mapping, {movie_id: column index in rating data}
    """
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(float(rating))
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping
'''
to analyze the data distribution
'''
def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')


if __name__ == '__main__':
    data_path = './ml-1m/'
    n_users = 6040
    n_movies = 3952
    # We load the data from ratings.dat using this function:
    data_path_rating = f'{data_path}ratings.dat'
    data, movie_n_rating, movie_id_mapping = load_rating_data(data_path_rating, n_users, n_movies)
    display_distribution(data)
    '''
    As you can see, most ratings are unknown; for the known ones, 
    35% are of rating 4,
    followed by 26% of rating 3,
    and 23% of rating 5,
    and then 11% and 6% of ratings 2 and 1, respectively.
    Since most ratings are unknown, we take the movie with the most known ratings as our target movie:
    '''
    movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
    print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')

    '''
    The movie with ID 2858 is the target movie, and ratings of the rest of the movies are signals.
      We construct the dataset accordingly:
    '''
    X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
    Y_raw = data[:, movie_id_mapping[movie_id_most]]

    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]

    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    print ("we take a look at the distribution of the target movie ratings Y:")
    display_distribution(Y)
    '''
    We can consider movies with ratings greater than 3 as being liked (being recommended):
    '''  
    recommend = 3
    Y[Y <= recommend] = 0
    Y[Y > recommend] = 1

    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} positive samples and {n_neg} negative samples.')

    '''
    As a rule of thumb in solving classification problems, we need to always analyze the label distribution
      and see how balanced (or imbalanced) the dataset is.
      Next, to comprehensively evaluate our classifier's performance, we can randomly split the dataset into two sets, the training and testing sets,
        which simulate learning data and prediction data, respectively. Generally, the proportion of the original dataset to include in the testing split
          can be 20%, 25%, 33.3%, or 40%. We use the train_test_split function from scikit-learn to do the random splitting and to preserve the percentage of samples for each class:
    '''
    '''
    It is a good practice to assign a fixed random_state (for example, 42) during experiments and exploration in order to guarantee that the same training and testing sets
      are generated every time the program runs. This allows us to make sure that the classifier functions and performs well on a fixed dataset before we incorporate randomness and proceed further.
    Another good thing about the train_test_split function is that the resulting training and testing sets will have the same class ratio
    '''
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    '''
    We check the training and testing sizes as follows:
    '''
    print(len(Y_train), len(Y_test))

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)
    '''
     Then, we use the trained model to make predictions on the testing set. We get the predicted probabilities as follows:
    '''
    prediction_prob = clf.predict_proba(X_test)
    print(prediction_prob[0:10])
    '''
    We get the predicted class as follows:
    '''
    prediction = clf.predict(X_test)
    print(prediction[:10])
    '''
    Finally, we evaluate the model's performance with classification accuracy, which is the proportion of correct predictions:
    '''
    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy*100:.1f}%')

    '''
    Beyond accuracy, there are several metrics we can use to gain more insight and to avoid class imbalance effects. 
    These are as follows:
    
    - Confusion matrix
    - Precision
    - Recall
    - F1 score
    - Area under the curve
  
    A confusion matrix summarizes testing instances by their predicted values and true values, presented as a contingency table:
    To illustrate this, we can compute the confusion matrix of our Naïve Bayes classifier. 
    We use the confusion_matrix function from scikit-learn to compute it, but it is very easy to code it ourselves
    '''
    '''
    To illustrate this, we can compute the confusion matrix of our Naïve Bayes classifier. We use the confusion_matrix function from scikit-learn
      to compute it, but it is very easy to code it ourselves:
    '''
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(Y_test, prediction, labels=[0, 1]))

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision_score(Y_test, prediction, pos_label=1)
    recall_score(Y_test, prediction, pos_label=1)
    f1_score(Y_test, prediction, pos_label=1)

    f1_score(Y_test, prediction, pos_label=0)

    from sklearn.metrics import classification_report
    report = classification_report(Y_test, prediction)
    print(report)


    pos_prob = prediction_prob[:, 1]

    thresholds = np.arange(0.0, 1.1, 0.05)
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(pos_prob, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                if y == 1:
                    true_pos[i] += 1
                else:
                    false_pos[i] += 1
            else:
                break

    n_pos_test = (Y_test == 1).sum()
    n_neg_test = (Y_test == 0).sum()
    true_pos_rate = [tp / n_pos_test for tp in true_pos]
    false_pos_rate = [fp / n_neg_test for fp in false_pos]


    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(Y_test, pos_prob))


    from sklearn.model_selection import StratifiedKFold
    k = 5
    k_fold = StratifiedKFold(n_splits=k, random_state=None)

    smoothing_factor_option = [1, 2, 3, 4, 5, 6]
    fit_prior_option = [True, False]
    auc_record = {}

    for train_indices, test_indices in k_fold.split(X, Y):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        for alpha in smoothing_factor_option:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, Y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)


    print('smoothing  fit prior  auc')
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f'    {smoothing}        {fit_prior}    {auc/k:.5f}')


    clf = MultinomialNB(alpha=2.0, fit_prior=False)
    clf.fit(X_train, Y_train)

    pos_prob = clf.predict_proba(X_test)[:, 1]
    print('AUC with the best model:', roc_auc_score(Y_test, pos_prob))

    ################################
    import pandas as pd
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer

    # Load movies data
    data_path_movies = f'{data_path}movies.dat'
    movies = pd.read_csv( data_path_movies,delimiter='::',engine='python',encoding='latin', names=['movie_id', 'movie_title', 'genres'])
    print(movies.shape)
    #print(movies.head())

    # Load users data
    data_path_users = f'{data_path}users.dat'
    users = pd.read_csv(data_path_users, delimiter='::', engine='python', header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

    data_path_ratings = f'{data_path}ratings.dat'
    ratings = pd.read_csv(data_path_ratings, delimiter='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'time'])
    ratings.head()

    # Merge movies, ratings, and users data
    merged_data = pd.merge(pd.merge(movies, ratings), users)

    # Preprocess the genres column
    merged_data['genres'] = merged_data['genres'].apply(lambda x: ' '.join(x.split('|')))
    print("### each movie may be of several genres ####")
    print(merged_data['genres'])

    # Create feature matrix and target variable
    X = merged_data['genres']
    y = merged_data['rating']

    # Convert genres to a matrix of token counts
    vectorizer = CountVectorizer()
    X_matrix = vectorizer.fit_transform(X)

    # Train Naïve Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_matrix, y)

    # Recommend top 10 movies to the user
    user_input = input("Enter your preferred movie genres: ")
    user_input_matrix = vectorizer.transform([user_input])
    movie_scores = nb_model.predict_proba(user_input_matrix)[0]
    top_movies_indices = movie_scores.argsort()[-10:][::-1]
    recommended_movies = movies.iloc[top_movies_indices]

    print("############ Recommended Movies:#########")
    for index, row in recommended_movies.iterrows():
        print('*)', row['movie_title'])

    ###############################
    # Get user ID from input
    user_id = input("Enter the user ID: ")

    # Get movies rated by the user
    user_movies = merged_data[merged_data['user_id'] == int(user_id)]
    user_movies_matrix = vectorizer.transform(user_movies['genres'])

    # Predict ratings for user movies
    user_ratings = nb_model.predict(user_movies_matrix)

    # Combine movie titles and prediccomedyted ratings
    user_movies['Predicted_Rating'] = user_ratings

    # Sort movies based on predicted ratings
    user_movies = user_movies.sort_values(by='Predicted_Rating', ascending=False)

    # Get top 10 recommended movies
    recommended_movies = user_movies.head(10)

    # Display recommended movies
    print("Recommended Movies for User", user_id + ":")
    for index, row in recommended_movies.iterrows():
        print(row['movie_title'], "(Predicted Rating:", row['Predicted_Rating'], ")")

    print("\n#####################")    
    ################################
    import plotly.express as px
    '''The explode() method converts each element of the specified column(s) into a row.'''
    
    '''Let’s plot the distribution of movies per genre. For this, 
    we will have to turn the genres column into a list and “explode” it.
    Note that each movie may be of several genres, and in this case, it will be counted multiple times.
    '''

    movies_exploded = movies.explode('genres')
    movie_count_by_genre = px.histogram(movies_exploded, x='genres', title='Movie count by genre')
    movie_count_by_genre.show()
    '''
    We will also plot the distribution of movies per movie release year.
    '''
    import re
    movies['year'] = movies['movie_title'].apply(lambda movie_title: re.search('\((\d*)\)', movie_title).groups(1)[0])
    movie_count_by_year = px.histogram(movies, x='year', height=400, title='Movie count by year')
    movie_count_by_year.show()
    '''
    Let's view the users. The "occupation" column is encoded with a number representing each occupation,
    but for EDA we are interested in the actual data. We will get it by extracting the relevant rows from the README file
    of the dataset and swapping values in the dataset accordingly.
    '''
    data_path_readme_text = f'{data_path}README'
    readme_text = np.array(open(data_path_readme_text).read().splitlines())
    start_index = np.flatnonzero(np.core.defchararray.find(readme_text,'Occupation is chosen')!=-1)[0]
    end_index = np.flatnonzero(np.core.defchararray.find(readme_text,'MOVIES FILE DESCRIPTION')!=-1)[0]
    occupation_list = [x.split('"')[1] for x in readme_text[start_index:end_index][2:-1].tolist()]
    occupation_dict = dict(zip(range(len(occupation_list)), occupation_list))
    ####
    users['occupation'] = users['occupation'].replace(occupation_dict)
    #print(users.head())
    print(f'There are {len(pd.unique(ratings["user_id"]))} unique users in the dataset')
    print(f'There are {len(pd.unique(ratings["movie_id"]))} unique movies in the dataset')
    '''
    Now let’s combine all three tables (movies, users, and ratings) and see if there are any differences between male and female ratings per movie genre.
    Notice that here too we will normalize by the total amount of males/females to get a better answer to our question.
    Again, we see some typical male/female differences in this dataset (men gave higher ratings to actions movies than women and vice versa for romance movies).
    This is a good sanity check for the data as the plots make sense.
    '''
    combined_ratings = pd.merge(pd.merge(movies_exploded, ratings, on='movie_id'), users, on='user_id')
    combined_ratings_data = combined_ratings.groupby(['genres', 'gender']).agg({'rating': ['mean', 'count']}).reset_index()
    combined_ratings_data.columns = [' '.join(col).strip() for col in combined_ratings_data.columns.values]

    combined_ratings_data.loc[combined_ratings_data['gender'] == 'F', 'rating count'] /= len(combined_ratings[combined_ratings['gender'] == 'F'])
    combined_ratings_data.loc[combined_ratings_data['gender'] == 'M', 'rating count'] /= len(combined_ratings[combined_ratings['gender'] == 'M'])

    ratings_by_gender_and_genre = px.bar(combined_ratings_data, x='genres', y='rating count', color='gender', barmode='group')
    ratings_by_gender_and_genre.show()

   
   

