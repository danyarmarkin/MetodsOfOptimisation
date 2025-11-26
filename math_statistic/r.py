r = [
    [1, 0.69, 0.58, 0.55],
    [0.69, 1, 0.46, 0.5],
    [0.58, 0.46, 1, 0.41],
    [0.55, 0.5, 0.41, 1],
]

rca = [[[0] * 4 for _ in range(4)] for _ in range(4)]

for i in range(0, 4):
    for j in range(0, 4):
        for k in range(0, 4):
            if i == j or i == k or j == k:
                continue
            rc = (r[i][j] - r[i][k] * r[j][k]) / ((1 - r[i][k] ** 2) * (1 - r[j][k] ** 2)) ** 0.5
            rca[i][j][k] = rc
            print(f"rx{i}x{j}|{k} = {rc:0.2f}")

print(*rca, sep="\n")
rcv = [0]*3
for i in range(3):
    j = (i + 1) % 3
    k = (i + 2) % 3
    rc = (rca[0][i + 1][j + 1] - rca[0][k + 1][j + 1] * rca[i + 1][k + 1][j + 1]) / (
            (1 - rca[0][k + 1][j + 1] ** 2) * (1 - rca[i + 1][k + 1][j + 1] ** 2)) ** 0.5
    print(f"ryx{i + 1}|x{j + 1}x{k + 1} = {rc:0.3f}")
    rcv[i] =rc

m = 1 - (1 - r[0][1] ** 2) * (1 - rca[0][2][1] ** 2) * (1 - rcv[2] ** 2)
print(m)

import numpy as np

ra = np.array(r)
print(1 - np.linalg.det(ra) / np.linalg.det(ra[1:, 1:]))