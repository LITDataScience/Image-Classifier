import math
import statistics

class MMM:
    def calculateMean(arr):
        mean = 0
        mean = statistics.mean(arr)
        return mean

    def calculateMedian(arr):
        med = 0.0
        med = statistics.median(arr)
        return med

    def calculateMode(arr):
        mode = 0
        size = len(arr)
        count, max = 0, 0
        copy = arr
        copy.sort()
        current = 0
        for i in copy:
            if (i == current):
                count += 1
            else:
                count = 1
                current = i
            if (count > max):
                max = count
                mode = i
        return mode

if __name__ == '__main__':
    N = int(input())
    arr = list(map(int, input().split()))
    print(MMM.calculateMean(arr))
    print(MMM.calculateMedian(arr))
    print(MMM.calculateMode(arr))