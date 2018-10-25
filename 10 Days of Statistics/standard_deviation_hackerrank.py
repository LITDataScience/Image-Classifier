from math import *
import statistics

class MMM:
    def calculateSD(arr):
        sd = 0
        mean = 0
        mean = statistics.mean(arr)
        data = [(val-mean)**2 for val in arr]
        return (sum(data)/float(len(data)))**0.5

if __name__ == '__main__':
    N = int(input())
    arr = list(map(int, input().split()))
    print(MMM.calculateSD(arr))