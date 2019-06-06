# Four_Step_Runge_Kutta_ODE1.py
# 
# Implementation of the classic fourth-order method also refered as the
# "original" Runge–Kutta method. This method is an implicit four step
# Runge-Kutta method which solves an intial value problem numerically. 
#
# Date: 06.06.2019.
# Created by Joshua Simon.

import matplotlib.pyplot as plt
from math import exp, sqrt 

# Four step Runge-Kutta method (RK4)
def runge_kutta (f, x_0, y_0, h):
    k_0 = f(x_0, y_0)
    k_1 = f(x_0 + h/2.0, y_0 + h/2.0 * k_0)
    k_2 = f(x_0 + h/2.0, y_0 + h/2.0 * k_1)
    k_3 = f(x_0 + h, y_0 + h * k_2)

    k = 1/6 * (k_0 + 2.0*k_1 + 2.0*k_2 + k_3)

    x_1 = x_0 + h
    y_1 = y_0 + h * k

    return (x_1, y_1)

# First order ordinary differential equation (ODE)
def f (x, y):
    return (5*x**2 - y) / (exp(x+y))

# Initial values
x_0 = 0.0
y_0 = 1.0

# Step length 
h = 0.1

x_values = []
y_values = []

x_values.append(x_0)
y_values.append(y_0)

# Calculate solution
for i in range(100):
    (x_0 ,y_0) = runge_kutta(f, x_0, y_0, h)
    x_values.append(x_0)
    y_values.append(y_0)
    print(x_0, y_0)

# Plot solution
plt.plot(x_values, y_values)
plt.show()
