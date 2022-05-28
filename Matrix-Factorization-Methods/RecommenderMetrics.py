import itertools
from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))
        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]
        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        # Đối với mỗi xếp hạng còn lại
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Nó có nằm trong top 10 dự đoán cho người dùng này không?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1
            total += 1
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # Đối với mỗi xếp hạng còn lại
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Chỉ xem xét khả năng giới thiệu những thứ mà người dùng thực sự thích ...
            if (actualRating >= ratingCutoff):
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1
                total += 1
        return hits/total

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # Đối với mỗi xếp hạng còn lại
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank
            total += 1
        return summation / total

    # Phần trăm người dùng có ít nhất một đề xuất "tốt"
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1
        return hits / numUsers

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1
        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n
