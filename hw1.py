import numpy as np
import matplotlib.pyplot as plt

N = 20
d = 2

# Generate random training data
X = np.random.uniform(-1, 1, size = (N, d+1))
X[:,0] = 1

#Generate random target function: f(x) = w^Tx
w = np.random.uniform(-1, 1, size = (d + 1))

#Compute true labels for the training data
Y = np.sign(np.dot(X,w))
ind_pos = np.where(Y==1)[0] #positive examples
ind_neg = np.where(Y==-1)[0] #negative examples

#Plot points
plt.clf()
plt.plot(X[ind_pos, 1], X[ind_pos, 2], 'ro')
plt.plot(X[ind_neg, 1], X[ind_neg, 2], 'bx')

print(Y)
print(X)
print(w)

#Adjust Perceptron
#Establish boolean and index variables for adjustment
correct_vector = False
error_var = False
i = 0
t = 1

#Generate starting test hypothesis
f = np.random.uniform(-1, 1, size = (d + 1))
# plt.plot(f)

#While loop to run through x values and check against perceptron
while correct_vector == False:
    Z = np.sign(np.dot(X,f))
    print(Z)
    for i in range(len(X)):
        if Y[i] == 1 and Z[i] == -1:
            break
        elif Y[i] == -1 and Z[i] == 1:
            break
    if i == len(X) - 1:
        correct_vector = True
        break
    else:
        f = f + np.dot(X[i,:], Y[i])
        t += 1
        i = 0

print("Number of updates = %d" % t)

line_x = np.linspace(-1, 1)
line_y = f[np.column_stack(np.ones(len(line_x))), line_x]
plt.plot(line_x, line_y, 'c')

#Plot specs
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 1.4')

plt.show()
