from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()
    rankings = ml.getPopularityRanks()
    return (data, rankings)
np.random.seed(0)
random.seed(0)

# Tải tập dữ liệu chung cho các thuật toán đề xuất
(evaluationData, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)

# SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

# random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(True)

