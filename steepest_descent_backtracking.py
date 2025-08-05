import numpy as np
 
def f(x):
    return (x[0]**2 + 1)**2 + x[1]**2

def gradient_of_function(x):
    return np.array([4 * x[0] * (x[0]**2 + 1), 2 * x[1]])

def norm(x):
    summ = 0
    for i in x:
        summ += i**2
    return (summ)**0.5

x = np.array([5.0, 2.0]) 

while norm(gradient_of_function(x)) >= 10**(-2):
    gradient = gradient_of_function(x)
    descent = -gradient
    alpha = 1.0
    
    while f(x + alpha * descent) > f(x) + 0.25 * alpha * np.dot(gradient, descent):
        alpha =alpha*0.5

    x = x + alpha*descent

print(f"Minimum is {x} , function value is {f(x)}")
