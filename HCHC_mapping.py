import numpy as np
import math
from pandas import DataFrame
import matplotlib.pyplot as pyplot
from utils import MnistDataset


data = np.load('embedded_MNIST.npy')
dataset = MnistDataset()
Y = dataset.y
nx = len(Y)
n_clu = len(np.unique(Y))


def DGBC(data):
    m, n = data.shape[0], data.shape[1]
    corr = np.corrcoef(data.T)
    vertex = (1-corr)/2
    dimorder = getShortestHamiltonianCycle(vertex)
    dimorder = shift(dimorder, 7)
    dimorder = dimorder.astype(np.int)
    sumcorr = 0
    corr = np.zeros(n)
    for i in range(n-1):
        corr[i] = vertex[dimorder[i], dimorder[i + 1]]
        sumcorr = sumcorr + vertex[dimorder[i], dimorder[i + 1]]

    corr[n - 1] = vertex[dimorder[n - 1], dimorder[0]]
    sumcorr = sumcorr + vertex[dimorder[n - 1], dimorder[0]]
    angle = np.zeros(n)
    angle[0] = -math.pi / 2
    GBCLocation = np.zeros((m + n, 2))
    GBCLocation[m, 0] = math.cos(angle[0])
    GBCLocation[m, 1] = math.sin(angle[0])

    for i in range(1, n):
        angle[i] = angle[i - 1] + 2 * math.pi * (corr[i - 1] / sumcorr)
        GBCLocation[i + m, 0] = math.cos(angle[i])
        GBCLocation[i + m, 1] = math.sin(angle[i])

    for i in range(m):
        tempsum = 0
        for j in range(n):
            tempsum = tempsum + data[i, dimorder[j]]
            GBCLocation[i, 0] = 0
            GBCLocation[i, 1] = 0

            if tempsum == 0:
                GBCLocation[i, 0] = 0
                GBCLocation[i, 1] = 0
            else:
                for j in range(n):
                    GBCLocation[i, 0] = GBCLocation[i, 0] + (data[i, dimorder[j]] / tempsum) * GBCLocation[m + j, 0]
                    GBCLocation[i, 1] = GBCLocation[i, 1] + (data[i, dimorder[j]] / tempsum) * GBCLocation[m + j, 1]

    return GBCLocation, dimorder

def getShortestHamiltonianCycle(dist):
    n = len(dist)
    l = 1 << n
    dp = np.ones((l, n)) * float('Inf')
    dp[1][0] = 0
    for mask in range(1, l, 2):
        for i in range(1, n, 1):
            if ((mask & 1 << i) != 0):
                for j in range(0, n, 1):
                    if ((mask & 1 << j) != 0):
                        dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + dist[j, i])


    res = float('Inf')
    for i in range(1, n, 1):
        res = min(res, dp[(1 << n) - 1][i] + dist[i, 0])
    cur = (1 << n) - 1
    last = 0
    order = np.zeros(n)
    for i in range(n-1, 0, -1):
        bj = 1
        for j in range(1, n):
            if ((cur & 1 << j) != 0 and (dp[cur][bj] + dist[bj, last] > dp[cur][j] + dist[j, last])):
                bj = j
        order[i] = bj
        cur ^= 1 << bj
        last = bj
        #order = [3,7,6,0,1,8,4,9,5,2]
    return order

def shift(order, num):
    lent = len(order)
    copyorder =  np.zeros(lent)
    for i in range(lent):
        copyorder[i]=order[(i + num) % (lent)]

    return copyorder




GBCLocation, order = DGBC(data)
pyplot.rcParams['savefig.dpi'] = 600
a = GBCLocation[0:nx,0]
b = GBCLocation[0:nx,1]
ax = GBCLocation[nx:nx+n_clu,0]
bx = GBCLocation[nx:nx+n_clu,1]


theta = np.linspace(0, 2 * np.pi, 200)

x = np.cos(theta)
y = np.sin(theta)

df = DataFrame(dict(x=a, y=b, label=Y))
colors = {0: 'grey', 1: 'blue', 2: 'green',  3: 'purple', 4: 'cyan', 5: 'olive', 6: 'darkblue', 7: 'deeppink',  8: 'orange',9: 'darkcyan'}
fig, axx = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=axx, kind='scatter', s=0.002, x='x', y='y', marker='h', color=colors[key])
pyplot.scatter(ax, bx, c="#d62728", alpha=1, s=60, linewidths=1, zorder=1, edgecolors="white")
pyplot.plot(x, y, c="#d62728", linewidth=1.5, linestyle="--", zorder=0)
pyplot.axis('equal')
pyplot.axis('off')

fig.set_size_inches(5, 5)
pyplot.savefig('MNIST.jpg',bbox_inches='tight', dpi=300, pad_inches=0)
pyplot.show()

