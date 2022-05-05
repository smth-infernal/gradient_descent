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

def ada_gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, step_size):
    solutions_x, solutions_y, scores = list(), list(), list()
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    initial_point_x, initial_point_y = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0]), bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
    for n in range(num_iter):
        gradient=derivative_x(initial_point_x, initial_point_y)+derivative_y(initial_point_x, initial_point_y)
        for i in range (gradient.shape[0]):
            sq_grad_sums[i] += gradient[i]**2
        for i in range (initial_point_x.shape[0]):
            alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
        initial_point_x, initial_point_y = initial_point_x - alpha*gradient, initial_point_y - alpha*gradient
        solution_eval = objective(initial_point_x, initial_point_y)
        solutions_x.append(initial_point_x)
        solutions_y.append(initial_point_y)
        scores.append(solution_eval)
        print(n, initial_point_x, initial_point_y, solution_eval)
    return [solutions_x, solutions_y, scores]

bounds = np.array([[-5,5]])
num_iter=1000
step_size=0.2
solutions_x, solutions_y, scores = ada_gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, step_size)
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

##input_x, input_y = np.meshgrid(input_x, input_y)
##input_x, input_y = input_x.flatten(), input_y.flatten()
##
##result = objective(input_x, input_y)
##
##tri = mtri.Triangulation(input_x, input_y)
##
##fig=plt.figure()
##ax= plt.axes(projection='3d')
##ax.plot_trisurf(input_x, input_y, result, triangles=tri.triangles, cmap=cm.hsv)
##plt.show()
