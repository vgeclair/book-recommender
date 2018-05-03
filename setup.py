import recommender
import data_processing

data_processing.get_ratings_matrix()
data_processing.get_tags_matrix()
data_processing.get_AL()

rec = Recommender()

rec.matrixFactorization()
