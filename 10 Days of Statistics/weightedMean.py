class MMM:
    def weightedMean(X, W):
        sum_X = sum([a*b for a,b in zip(X,W)])
        return round((sum_X/sum(W)),1)

if __name__ == '__main__':
    N = int(input())
    arr = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    print(MMM.weightedMean(arr, weights))