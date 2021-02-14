
# BUCKITO : A MOVIE RECOMMENDATION ENGINE  :movie_camera:

**FIND INCREDIBLE MOVIES TO WATCH** :clapper::v:

Have you ever wondered how Netflix manages to recommend so many random movies that all somehow fit your preferences in one way or another? Most, if not all, of it has to do with data science, but more specifically, big data, machine learning, and artificial intelligence.Taking Netflix's Recommendation we'have also planned build our own.In this Project, we are going to Build a Movie Recommendation a hybrid Movie Recommendation Engine as there various kind of Recommendation Like Content and Collabrative filtering,The dataset we are going to we have used is the Movie dataset Online available which have Almost 4800+ Movies and movies upto 2017 and and also have collected remaining data of years 2018-2020 from wikipidia,We have also used The Movie database(TMDB) API for other importent details.


### TRY OUT OUR WEB-APPLICATION :fire:
-If you want to view the deployed model, click on the FOLLOWING LINK
 [BUCKITO](http://buckito.hashigma.com/):purple_heart:\
 Beautiful isn't it.\
-If you are searching for Code, Algorithms, Which similarity metrics I used and much more Please Open "FinalModel.ipynb" file inside Notebook Folder.

### SCREENSHOTS :fireworks:
![Landing_page](/Images/4.PNG)
![Popular_page](/Images/2.PNG)
![Modale](/Images/3.PNG)
![Main page](/Images/1.PNG)


### DATASETS & SOURCES :clipboard:
[THE TMDB DATASET](https://www.kaggle.com/tmdb/tmdb-movie-metadata):clipboard:\
[WIKIPIDIA-2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018) [WIKIPIDIA-2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)\
[Application Programming Interface](https://developers.themoviedb.org/3):blue_book:



### HYBRID MODELS:ferris_wheel:

**Popularity based recommendation engine:**:signal_strength:
Perhaps, this is the simplest kind of recommendation engine that you will come across. The trending list you see in YouTube or Netflix is based on this algorithm. It keeps a track of view counts for each movie/video and then lists movies based on views in descending order(highest view count to lowest view count). Pretty simple but, effective. Right?

**Content based recommendation engine:**:arrow_forward:
This type of recommendation systems, takes in a movie that a user currently likes as input. Then it analyzes the contents (storyline, genre, cast, director etc.) of the movie to find out other movies which have similar content. Then it ranks similar movies according to their similarity scores and recommends the most relevant movies to the user.

**Collaborative filtering based recommendation engine:**:two_men_holding_hands:
This algorithm at first tries to find similar users based on their activities and preferences (for example, both the users watch same type of movies or movies directed by the same director). Now, between these users(say, A and B) if user A has seen a movie that user B has not seen yet, then that movie gets recommended to user B and vice-versa. In other words, the recommendations get filtered based on the collaboration between similar user’s preferences (thus, the name “Collaborative Filtering”). One typical application of this algorithm can be seen in the Amazon e-commerce platform, where you get to see the “Customers who viewed this item also viewed” and “Customers who bought this item also bought” list.


[REACT REPOSITORY](https://github.com/Priyanshu-C/BUCKITO)

### HOW TO RUN FASTAPI
**Upgrade pip before install fastapi**
python -m venv fastapi

**Upgrade pip before install fastapi**
python -m pip install --upgrade pip

**To run the server**
uvicorn server:app --reload

**To run the FASTAPI POSTMAN**
http://127.0.0.1:8000/docs

