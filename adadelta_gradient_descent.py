from math import sqrt
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib import pyplot



def objective(x,y):
    return (x+2*y-7)**2+(2*x+y-5)**2

def derivative_x(x,y):
    return 10*x+8*y-34

def derivative_y(x,y):
    return 8*x+10*y-38

def adadelta_gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, rho, epsilon=1e-3):
    solutions_x, solutions_y, scores = list(), list(), list()
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
    sq_param_avg = [0.0 for _ in range(bounds.shape[0])]
    initial_point_x, initial_point_y = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0]), bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
    for n in range(num_iter):
        gradient=derivative_x(initial_point_x, initial_point_y)+derivative_y(initial_point_x, initial_point_y)
        for i in range (gradient.shape[0]):
            sg = gradient[i]**2
            sq_grad_avg[i] = (sq_grad_avg[i]*rho)+(sg*(1-rho))
        for i in range (initial_point_x.shape[0]):
            alpha = (epsilon + sqrt(sq_param_avg[i])) / (epsilon + sqrt(sq_grad_avg[i]))
            change = alpha*gradient[i]
            sq_param_avg[i] = (sq_param_avg[i]*rho) + (change**2 * (1-rho))
        initial_point_x, initial_point_y = initial_point_x - change, initial_point_y - change
        solution_eval = objective(initial_point_x, initial_point_y)
        solutions_x.append(initial_point_x)
        solutions_y.append(initial_point_y)
        scores.append(solution_eval)
        print(n, initial_point_x, initial_point_y, solution_eval)
    return [solutions_x, solutions_y, scores]

bounds = np.array([[-5,5]])
num_iter=501
rho=0.95
solutions_x, solutions_y, scores = adadelta_gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, rho, epsilon=1e-3)
input_x, input_y = np.arange(bounds[0,0], bounds[0,1]), np.arange(bounds[0,0], bounds[0,1])
result = objective(input_x, input_y)
out = []
out1 = []
out2 = []
out_x = np.append(out, solutions_x)
out_y = np.append(out1, solutions_y)
out_z = np.append(out2, scores)


xaxis, yaxis = np.linspace(-5, 5, num=100), np.linspace(-5, 5, num=100)
x,y = np.meshgrid(xaxis, yaxis)

result = objective(x,y)
pyplot.contour(x, y, result, levels=100, cmap='jet')
pyplot.plot(out_x, out_y, 'o', color='black')
plt.show()
##min, max = -5, 5
##input_x, input_y = np.linspace(min, max, num=15), np.linspace(min, max, num=15)
##
##input_x, input_y = np.meshgrid(input_x, input_y)
##input_x, input_y = input_x.flatten(), input_y.flatten()
##
##result = objective(input_x, input_y)
####
####tri = mtri.Triangulation(input_x, input_y)
##result = result.flatten()
##
##fig=plt.figure()
##ax=plt.axes(projection='3d')
##ax.plot_surface(input_x, input_y, result, cmap=cm.hsv)
##ax.plot(out_x, out_y, out_z, 'o',color='w')
##plt.show()
