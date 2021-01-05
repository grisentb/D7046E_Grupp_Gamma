import pandas as pd
import random

class Movie:
  def __init__(self, name="No title", genres = [], length = 0, year = None, rating = 0.0):
    self.name = name
    self.genres = genres
    self.length = length
    self.year = year
    self.rating = rating

class MoviePicker:
  def __init__(self, allMovies):
    self.movies = allMovies

  def pickNewMovie(self, movie, sentiment):
    #How much simularity are we looking for depending on the sentiment from the classifier
    if sentiment == 0:
      sentimentRequirement = 2
    elif sentiment == 1:
      sentimentRequirement = 3
    elif sentiment == 2:
      sentimentRequirement = None
    elif sentiment == 3:
      sentimentRequirement = 4
    else:
      sentimentRequirement = 5
    print(sentimentRequirement)
    #Do we want to find a simulkar movie or not
    randomMovie = self.movies[random.randint(0, len(self.movies))]
    if sentimentRequirement == None or movie.name == None:
      print("None \n {} \n {}".format(sentimentRequirement, movie.name))
      return randomMovie
    elif sentimentRequirement > 3:
      print("Liked it")
      while self.simularity(movie, randomMovie) < sentimentRequirement:
        randomMovie = self.movies[random.randint(0, len(self.movies))]
      print(self.simularity(movie, randomMovie))
      return randomMovie
    elif sentimentRequirement <= 3:
      print("Didn't like it")
      while self.simularity(movie, randomMovie) > sentimentRequirement:
        randomMovie = self.movies[random.randint(0, len(self.movies))]
      return randomMovie
    return None 

  def simularity(self, movieA, movieB): #Gives a simularity value between 1-7 depending on genres and release year
    simularityValue = 1
    genresInCommon = 0
    for aGenre in movieA.genres:
      for bGenre in movieB.genres:
        if aGenre == bGenre:
          genresInCommon += 1
          break
    genreRatio = float(genresInCommon) / float(len(movieA.genres)) #It is a relational increas of simularity value depending of the genres simularity. If all genres are the same simularityValues increases with 5 points
    simularityValue += genreRatio * 5

    yearDifference = abs(int(movieA.year) - int(movieB.year))
    if yearDifference < 10:
      simularityValue += 2
    return simularityValue

allMovies = []

data = pd.read_csv("IMDb movies.csv", header=None)
data.columns = ['imdb_title_id','title','original_title','year','date_published','genre','duration','country','language','director','writer','production_company','actors','description','avg_vote','votes','budget','usa_gross_income','worlwide_gross_income','metascore','reviews_from_users','reviews_from_critics']
#data.columns = ['movie']

k = 0
for i in data.iterrows():
  movie = i[1]
  movieObject = Movie(movie.title, movie.genre, movie.duration, movie.year, movie.reviews_from_users)
  allMovies.append(movieObject)

#moviePicker = new MoviePicker(allMovies)

moviePicker = MoviePicker(allMovies)

oldMovie = allMovies[76423]
#print(specificMovie.name)
#oldMovie = allMovies[random.randint(0,len(allMovies))]
sentiment = 0
newMovie = moviePicker.pickNewMovie(allMovies[0], sentiment)
print("Old movie: {} \n  genres: {} \n ################### \n New Movie {} \n genres: {}".format(oldMovie.name, oldMovie.genres, newMovie.name, newMovie.genres))
