import numpy as np
D = 10000


def create_random_vector():
    hv = np.random.choice([-1, 1], D)
    return hv


def sim(hv1, hv2):
    return np.dot(hv1, hv2) / D

N = 10

Xs = []
Ys = []

hash_table = np.zeros(D)

for i in range(N):
    x = create_random_vector()
    y = create_random_vector()
    hash_table += x * y

    Xs.append(x)
    Ys.append(y)


x_query = Xs[0]
x_query[1] = -1
y_true = Ys[0]
print(x_query)

h = hash_table * x_query
for i in range(N):
    print("Ys[i]: ", Ys[i])
    print(i, sim(h, Ys[i]))

