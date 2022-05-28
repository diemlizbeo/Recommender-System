from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData

class EvaluatedAlgorithm:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {} 
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMetrics.MAE(predictions)
        
        if (doTopN):
            # Đánh giá top 10 với Leave One Out testing
            if (verbose):
                print("Đánh giá top-N với leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())        
           # Xây dựng dự đoán cho tất cả các xếp hạng không có trong training set
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet())
            # Tính toán 10 gợi ý hàng đầu cho mỗi người dùng
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Tính toán các số liệu về tỷ lệ truy cập và xếp hạng ...")
            # Xem tần suất ta đề xuất một bộ phim mà người dùng thực sự đã xếp hạng
            metrics["HR"] = RecommenderMetrics.HitRate(topNPredicted, leftOutPredictions)   
            # Xem tần suất ta đề xuất một bộ phim mà người dùng thực sự thích
            metrics["cHR"] = RecommenderMetrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            # tính toán ARHR
            metrics["ARHR"] = RecommenderMetrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
        
            # Đánh giá thuộc tính của các đề xuất trên full training set
            if (verbose):
                print("Tính toán các đề xuất với tập dữ liệu đầy đủ ...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print("Phân tích coverage, diversity, and novelty...")
            # In mức độ phù hợp của người dùng với xếp hạng dự đoán tối thiểu là 4,0:
            metrics["Coverage"] = RecommenderMetrics.UserCoverage(topNPredicted, evaluationData.GetFullTrainSet().n_users, ratingThreshold=4.0)
            # tính đa dạng
            metrics["Diversity"] = RecommenderMetrics.Diversity(topNPredicted, evaluationData.GetSimilarities())
            
            # Đo lường tính mới (xếp hạng phổ biến trung bình của các đề xuất):
            metrics["Novelty"] = RecommenderMetrics.Novelty(topNPredicted, evaluationData.GetPopularityRankings())
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm
    
    