from MovieLens import MovieLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)
np.random.seed(0)
random.seed(0)

(ml, evaluationData, rankings) = LoadMovieLensData()
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")
# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")
# random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)
evaluator.SampleTopNRecs(ml)
