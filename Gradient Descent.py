# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

dataset = pd.read_csv(filedialog.askopenfilename()) 
dataset

# # All function


def gradient_descent(x,y,learning_rate=0.1,iterations= 1000):
    m_curr = b_curr = 0
    n = len(x)
    data = {'m': [], 'b': [],'cost':[], 'iteration': []}
    df = pd.DataFrame(data)
    for i in range(iterations):
        try:
            y_predicted = m_curr * x + b_curr
            cost = ((1/n) * sum([val**2 for val in (y-y_predicted)]))
            md = -(2/n) * sum(x * (y - y_predicted))
            bd = -(2/n) * sum(y - y_predicted)
            m_curr = m_curr - learning_rate * md
            b_curr = b_curr - learning_rate * bd
            #print ([m_curr, b_curr,cost, i])
            df.loc[i] = [m_curr, b_curr,cost, i]
        except OverflowError as e:
            print('Its an Overflow error, please check.')
            return df        
    return df

def mini_batch_gradient_descent(x, y,learning_rate=0.1,iterations= 1000):
    m_curr = b_curr = 0
    n = len(x)
    batch_size = 20
    data = {'m': [], 'b': [],'cost':[], 'iteration': []}
    df = pd.DataFrame(data)

    for i in range(iterations):
        try:

            random_index = np.random.randint(n)
            x_i = x[random_index:random_index+batch_size]
            y_i = y[random_index:random_index+batch_size]
            y_predicted = m_curr * x_i + b_curr
            cost = (1/n) * sum([val**2 for val in (y_i-y_predicted)])
            md = -(2/batch_size) * np.sum(x_i * (y_i - y_predicted))
            bd = -(2/batch_size) * np.sum(y_i - y_predicted)
            m_curr = m_curr - learning_rate * md
            b_curr = b_curr - learning_rate * bd
            df.loc[i] = [m_curr, b_curr,cost, i]
        except OverflowError as e:
            print('Its an Overflow error, please check.')
            return df     
    return df

def stochastic_gradient_descent(x, y,learning_rate=0.1,iterations= 1000):
    data = {'m': [], 'b': [],'cost':[], 'iteration': []}
    df = pd.DataFrame(data)
    m_curr = b_curr = 0
    n = len(x)


    for i in range(iterations):
        random_index = np.random.randint(n)
        x_i = x[random_index]
        y_i = y[random_index]
        y_predicted = m_curr * x_i + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * x_i * (y_i - y_predicted)
        bd = -(2/n) * (y_i - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        df.loc[i] = [m_curr, b_curr,cost, i]

    return df

gd_00001 = gradient_descent(dataset['x'],dataset['y'],0.0001)
gd_01 = gradient_descent(dataset['x'],dataset['y'])
mbgd_00001 = mini_batch_gradient_descent(dataset['x'],dataset['y'],0.0001)
mbgd_01 = mini_batch_gradient_descent(dataset['x'],dataset['y'],0.1)
sgd_00001 = stochastic_gradient_descent(dataset['x'],dataset['y'],0.0001)
sgd_01 = stochastic_gradient_descent(dataset['x'],dataset['y'])



# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( gd_00001['iteration'] ,gd_00001['m'])
ax[1].plot(gd_00001['iteration'],gd_00001['b'])
ax[2].plot( gd_00001['iteration'],gd_00001['cost'])

fig.suptitle('m and b in each epoch of Gradient Descent with a learning rate of 0.0001'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( gd_01['iteration'] ,gd_01['m'])
ax[1].plot(gd_01['iteration'],gd_01['b'])
ax[2].plot( gd_01['iteration'],gd_01['cost'])

fig.suptitle('m and b in each epoch of Gradient Descent with a learning rate of 0.1'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")
    i.set_yscale('log')

# Show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( sgd_00001['iteration'] ,sgd_00001['m'])
ax[1].plot(sgd_00001['iteration'],sgd_00001['b'])
ax[2].plot( sgd_00001['iteration'],sgd_00001['cost'])

fig.suptitle('m and b in each epoch of Stochastic Gradient Descent with a learning rate of 0.0001'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( sgd_01['iteration'] ,sgd_01['m'])
ax[1].plot(sgd_01['iteration'],sgd_01['b'])
ax[2].plot( sgd_01['iteration'],sgd_01['cost'])

fig.suptitle('m and b in each epoch of Stochastic Gradient Descent with a learning rate of 0.1'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( mbgd_00001['iteration'] ,mbgd_00001['m'])
ax[1].plot(mbgd_00001['iteration'],mbgd_00001['b'])
ax[2].plot( mbgd_00001['iteration'],mbgd_00001['cost'])

fig.suptitle('m and b in each epoch of Mini Batch Gradient Descent with a learning rate of 0.0001'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")

# Show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(1, 3,figsize = (17,6))
# Plot the three lines
ax[0].plot( mbgd_01['iteration'] ,mbgd_01['m'])
ax[1].plot(mbgd_01['iteration'],mbgd_01['b'])
ax[2].plot( mbgd_01['iteration'],mbgd_01['cost'])

fig.suptitle('m and b in each epoch of Mini Batch Gradient Descent with a learning rate of 0.1'
             ,fontsize = 25)

# Set titles for the subplots
ax[0].set_title('m per epoch')
ax[1].set_title('b per epoch')
ax[2].set_title('cost per epoch')

ax[2].set_yscale('log')

# Set labels for the x and y axes
ax[0].set_ylabel("m")
ax[1].set_ylabel("b")
ax[2].set_ylabel("cost")
for i in ax:
    i.set_xlabel("epoch")
    i.set_yscale('log')

# Show the plot
plt.show()






