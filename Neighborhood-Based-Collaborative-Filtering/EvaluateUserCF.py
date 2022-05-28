from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

def LoadMovieLensData():
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

ml, data, rankings = LoadMovieLensData()

evalData = EvaluationData(data, rankings)

trainSet = evalData.GetLOOCVTrainSet()
sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()
# Xây dựng dict thành danh sách các cặp (int (movieID), predictedrating)
topN = defaultdict(list)
k = 10
for uiid in range(trainSet.n_users):
    
    similarityRow = simsMatrix[uiid]
    
    similarUsers = []
    for innerID, score in enumerate(similarityRow):
        if (innerID != uiid):
            similarUsers.append( (innerID, score) )
    
    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
    
    # Nhận những thứ họ đã xếp hạng và thêm xếp hạng cho từng mặt hàng, được tính theo mức độ tương tự của người dùng
    candidates = defaultdict(float)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = trainSet.ur[innerID]
        for rating in theirRatings:
            candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
        
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1
        
    # lấy các mặt hàng được xếp hạng cao nhất từ những người dùng tương tự:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            topN[int(trainSet.to_raw_uid(uiid))].append( (int(movieID), 0.0) )
            pos += 1
            if (pos > 40):
                break
    
# Measure
print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))   


