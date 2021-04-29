
#Can be used in programming challenge: K-nearest neighbor, linear discriminate analysis
import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt



switch_kernel = 3
def kernel(datapoint1, datapoint2):
    if switch_kernel == 1:
        return np.dot(datapoint1, datapoint2)
    elif switch_kernel == 2:
        p = 2
        return math.pow(np.dot(datapoint1, datapoint2) + 1, p)
    elif switch_kernel == 3:
        dist = np.linalg.norm(datapoint1 - datapoint2)
        sigma = 5
        exp = math.pow(dist, 2) / (2 * math.pow(sigma, 2))
        return math.pow(math.e, -exp)

def init_matrix(N : int, targets, inputs):
    p_matrix = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            p_matrix[i,j] = targets[i]*targets[j] * kernel(inputs[i], inputs[j])
    return p_matrix

def objective(alpha):
    sum_points = 0
    sum_alpha = 0
    for i in range(N):
        sum_alpha += alpha[i]
        for j in range(N):
            sum_points += alpha[i]*alpha[j]*p_matrix[i,j]
    return 0.5*sum_points - sum_alpha

def zerofun(alpha):
    return np.dot(alpha, targets)



def calculate_b_val(): #Equation 7
    sum_b_val = 0
    select_vector = s_vector[0][2] #
    for i in range(len(s_vector)):
        sum_b_val += s_vector[i][0]*s_vector[i][1]*kernel(s_vector[i][2], select_vector)
    return sum_b_val-s_vector[0][1]

def indicator(x,y): #Equation 6
    sum_indicator = 0
    for i in range(len(s_vector)):
        sum_indicator += s_vector[i][0]*s_vector[i][1]*kernel([x,y], s_vector[i][2])
    return sum_indicator - b



np.random.seed(100)
spread = 0.7
classA = np.concatenate(
    (np.random.randn(10, 2) * spread + [1.5, 0.5],
     np.random.randn(10, 2) * spread + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * spread + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]),
     -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, : ]
targets = targets[permute]

#---------------------------------------------
C = 100
bounds=[(0, C) for b in range(N)]
#bounds = [(0, None) for b in range(N)]
start = np.zeros(N)
p_matrix = init_matrix(N, targets, inputs)
XC = {'type':'eq', 'fun':zerofun}
ret = minimize(objective, start, bounds=bounds, constraints=XC)
alphas = [round(i, 5) for i in ret['x']]

s_vector = []
for i in range(N):
    if alphas[i] != 0:
        s_vector.append([alphas[i], targets[i], inputs[i]])

b = calculate_b_val()
#print(b)

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
# 1 = targets, 0 = alphas, 2 = inputs
grid = np.array([[indicator(x, y)
                  for x in xgrid]
                 for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

plt.axis('equal')
#plt.savefig('svmplot.pdf')
plt.show()