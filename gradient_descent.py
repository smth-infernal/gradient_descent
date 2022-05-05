from math import sqrt
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot


def objective(x,y):
    return (x+2*y-7)**2+(2*x+y-5)**2

def derivative_x(x,y):
    return 10*x+8*y-34

def derivative_y(x,y):
    return 8*x+10*y-38

def gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, step_size):
    solutions_x, solutions_y, scores = list(), list(), list()
    initial_point_x, initial_point_y = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0]), bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
    for n in range(num_iter):
        gradient=derivative_x(initial_point_x, initial_point_y)+derivative_y(initial_point_x, initial_point_y)
        initial_point_x, initial_point_y = initial_point_x - step_size*gradient, initial_point_y - step_size*gradient
        solution_eval = objective(initial_point_x, initial_point_y)
        solutions_x.append(initial_point_x)
        solutions_y.append(initial_point_y)
        scores.append(solution_eval)
##        print(n, initial_point_x, initial_point_y, solution_eval)
    return [solutions_x, solutions_y, scores]

bounds = np.array([[-5,5]])
num_iter=501
step_size=0.05
solutions_x, solutions_y, scores = gradient_descent(objective, derivative_x, derivative_y, bounds, num_iter, step_size)
input_x, input_y = np.arange(bounds[0,0], bounds[0,1]), np.arange(bounds[0,0], bounds[0,1])
out = []
out1 = []
out2 = []
out_x = (np.append(out, solutions_x)).tolist()
out_y = (np.append(out1, solutions_y)).tolist()
out_z = (np.append(out2, scores)).tolist()



xaxis, yaxis = np.linspace(-5, 5, num=100), np.linspace(-5, 5, num=100)
x,y = np.meshgrid(xaxis, yaxis)

result = objective(x,y)
pyplot.contour(x, y, result, levels=100, cmap='jet')
pyplot.plot(out_x, out_y, '.', color='black', alpha=0.7)
plt.show()

##fig = go.Figure()
##fig.add_surface(x=x, y=y, z=result)
##
##fig.add_scatter3d(x=out_x, y=out_y, z=out_z, mode='lines+markers',
##                  marker=dict(color='white', size=8))
##
##fig.show()
##min, max = -5, 5
##input_x, input_y = np.arange(min, max), np.arange(min, max)
##
##input_x, input_y = np.meshgrid(input_x, input_y)
####input_x, input_y = input_x.flatten(), input_y.flatten()
##
##results = objective(input_x, input_y)
##print(results)
##
####tri = mtri.Triangulation(input_x, input_y)
##
##fig=plt.figure()
##ax= plt.axes(projection='3d')
##ax.plot_surface(input_x, input_y, results, cmap=cm.hsv)
##ax.plot(out_x, out_y, out_z, '.-', color='black', alpha=0.5)
##plt.show()
