import numpy as np
from scipy.optimize import minimize_scalar 

def f(x):
    return(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

def gradient(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

def hessian(x):
    return np.array([[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]])

def inverse_hessian(x):
    return np.linalg.inv(hessian(x))

eppsilon = 10 ** (-5)
x_1 = np.array([0, 0])
x_2 = np.array([2.0,2.0])
x_3 = np.array([-1.0,-1.0])

def newton_method(x_k, eppsilon):
    i = 0
    while np.linalg.norm(gradient(x_k)) >= eppsilon:
        i = i + 1
        d_k =  -1 * np.matmul(np.linalg.inv(hessian(x_k)), gradient(x_k))
        x_k_1 = x_k + d_k
        x_k = x_k_1
    
    print(i)
    return x_k

print("|||||||||||||Newton's method|||||||||||||")
print(newton_method(x_1, eppsilon))
print(newton_method(x_2, eppsilon))
print(newton_method(x_3, eppsilon))


x_1 = np.array([0, 0])
x_2 = np.array([2.0,2.0])
x_3 = np.array([-1.0,-1.0])

def steepest_descent(x_k, eppsilon):
    i = 0
    while np.linalg.norm(gradient(x_k)) >= eppsilon:
        i = i + 1
        phi_k = lambda alpha: f(x_k - alpha * gradient(x_k))
        alpha_k = minimize_scalar(phi_k, bounds = (0,50))
        x_k_1 = x_k - alpha_k.x * gradient(x_k) 
        x_k = x_k_1
    
    print(i)
    return x_k

print("|||||||||||||Steepest descent method|||||||||||||")
print(steepest_descent(x_1, eppsilon))
print(steepest_descent(x_2, eppsilon))
print(steepest_descent(x_3, eppsilon))

x_1 = np.array([0, 0])
x_2 = np.array([2.0,2.0])
x_3 = np.array([-1.0,-1.0])

def conjugate_gradient(x0, eppsilon): # keeps overflwoing
    d_0 = gradient(x0)
    if d_0[0] == 0 and d_0[1] == 0:
        return x0
    d_0 = -d_0
    i = 0
    while np.linalg.norm(gradient(x0)) >= eppsilon:
        phi_0 = lambda a: f(x0 + a * d_0)
        alpha_0 = minimize_scalar(phi_0, bounds=(0, 5))
        x0 = x0 + alpha_0.x * d_0
        g_1 = gradient(x0)

        if g_1[0] == 0 and g_1[1] == 0:
            return x0
        b0 = np.matmul(np.transpose(g_1), g_1 - d_0) / (np.matmul(np.transpose(d_0), g_1 - d_0))
        d_0 = -g_1 + b0 * d_0
        d_0 = g_1
        i += 1

    return (i, x0)

print("|||||||||||||Conjugate gradient method|||||||||||||")
print(conjugate_gradient(x_1, eppsilon))
print(conjugate_gradient(x_2, eppsilon))
print(conjugate_gradient(x_3, eppsilon))

# We can see that the best choice here is newton's method, since it took only 2 iterations 
# to approximate the minimizer of the given function within the given precisoin. We can also see that 
# newton's method and conjugate gradient have the same result, but the number of iterations of the last one
# is greater than the other one. Therefore, we choose newton's method.
