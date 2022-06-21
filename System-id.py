

"""
Referenced from:
SISO system ID sample code for students
by J R Forbes, 2022/02/02
"""

# %%
# Libraries
import numpy as np
import control
import os
from matplotlib import pyplot as plt

# Custom libraries 
import d2c
import discrete_functions as df


# %% 

# Define the path for the file of the input output datasets
# Note that this path must only contain the csv files
path = 'C:/Users/Joshua Xu/Downloads/load_data_sc/DATA/'

# Read in input-output data from the path
files = os.listdir(path)

# Data list contains a list of all the data stored as numpy array
data_list = []

for i in files:
    data_read = np.loadtxt( path + i,
                            dtype=float,
                            delimiter=',',
                            skiprows=1,
                            usecols=(0, 1, 2, 3))
    data_list.append(data_read)


# %%
t = []
r = []
u = []
y = []
for data in data_list:
    t.append(data[:,0])
    r.append(data[:,1])
    u.append(data[:,2])
    y.append(data[:,3])


# %%
print(np.shape(A))
# %%
for j in ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)):
    print(j)
    print("\n")
    for i in range(len(y)):
        print(i)
        # Solve for x.
        n = j[0]  # select according to the testing parameters 
        m = j[1]  # select according to the testing parameters
        
        if (n, m) == (1, 0):
            A, b = df.DiscreteTime1_0(y[i], u[i])
        elif (n, m) == (2, 0):
            A, b = df.DiscreteTime2_0(y[i], u[i])
        elif (n, m) == (2, 1):
            A, b = df.DiscreteTime2_1(y[i], u[i])
        elif (n, m) == (3, 0):
            A, b = df.DiscreteTime3_0(y[i], u[i])
        elif (n, m) == (3, 1):
            A, b = df.DiscreteTime3_1(y[i], u[i])
        elif (n, m) == (3, 2):
            A, b = df.DiscreteTime3_2(y[i], u[i])

        A_T = np.transpose(A)

        x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T,A)),A_T),b)
        print('For dataset ' + str(i))
        print('The normalised parameter estimates are\n', x,'\n')


        # Compute the uncertainty and relative uncertainty. 
        N = y[i].size
        cov_A = 1/(N-(n+m+1))*np.linalg.norm(b-np.matmul(A,x))**2*np.linalg.inv(np.matmul(A_T,A))
        sigma = np.sqrt(np.diag(cov_A))  # You change.
        rel_unc = sigma/np.abs(x) *100  # You change.

        print('The standard deviation is', sigma)
        print('The relative uncertainty is', rel_unc, '%\n')

        # Compute the MSO, MSE and NMSE
        MSO = (1/N)*(np.linalg.norm(b))**2
        MSE = (1/N)*(np.linalg.norm(b - np.matmul(A, x)))**2
        NMSE = MSE/MSO 

        print('The NMSE is', NMSE)

# %%
x1_0 = list()
for i in range(len(y)):
    # Solve for x.
    n = 1  # select according to the testing parameters 
    m = 0  # select according to the testing parameters
    
    A, b = df.DiscreteTime1_0(y[i], u[i])
    A_T = np.transpose(A)
    x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T,A)),A_T),b)
    x1_0.append(x)

print(x1_0)

x3_1 = list()
for i in range(len(y)):
    # Solve for x.
    n = 3  # select according to the testing parameters 
    m = 1  # select according to the testing parameters
    
    A, b = df.DiscreteTime3_1(y[i], u[i])
    A_T = np.transpose(A)
    x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_T,A)),A_T),b)
    x3_1.append(x)

# Calculate the NMSE test for (1, 0)
for i in range(5):
    print(i)
    sum = 0
    for j in range(5):
        N = y[j].size
        A_test, b_test = df.DiscreteTime1_0(y[j], u[j])
        # Compute the MSO, MSE and NMSE
        MSO = (1/N)*(np.linalg.norm(b_test))**2
        MSE = (1/N)*(np.linalg.norm(b_test - np.matmul(A_test, x1_0[i])))**2
        NMSE = MSE/MSO 
        sum += NMSE
        print('The NMSE test (1, 0) is', NMSE)  
    print("Sum:", sum/5) 

# Calculate the NMSE test for (3, 1)
for i in range(5):
    print(i)
    sum = 0
    for j in range(5):
        N = y[j].size
        A_test, b_test = df.DiscreteTime3_1(y[j], u[j])
        # Compute the MSO, MSE and NMSE
        MSO = (1/N)*(np.linalg.norm(b_test))**2
        MSE = (1/N)*(np.linalg.norm(b_test - np.matmul(A_test, x3_1[i])))**2
        NMSE = MSE/MSO
        sum += NMSE 
        print('The NMSE test (3, 1) is', NMSE)
    print("Sum:", sum/5) 
print(x3_1[0])


#### Calculate all the TF of the models and (% VAF)
# (3, 1)
for i in range(5):
    y_bar = max(abs(y[i]))
    u_bar = max(abs(u[i]))

    print(f'y max/u max is {(y_bar / u_bar)}, and x coefficients are {x3_1[i]}')


    # Create discrete model using coefficients found above 
    P_d = (y_bar / u_bar) * control.tf(np.array([x3_1[i][3], x3_1[i][4]]), np.array([1,x3_1[i][0], x3_1[i][1], x3_1[i][2]]), 0.01)
    print('The discrete-time TF is,', P_d)

    # Discrete to Continuous System ID TF 
    P_c = d2c.d2c(P_d)
    print('The continuous-time TF is,', P_c)
    print(i)

    # Calculate the % VAF for each dataset to sit how the model fits the dataset.
    for j in range(5):
        y_test = y[j]
        N = y_test.size
        t_id, y_id = control.forced_response(P_d, t[j], u[j])
        e = y_id - y_test
        var_error = (1/N)*(np.matmul(e.transpose(), e))
        var_y = (1/N)*(np.matmul(y_test.transpose(), y_test))

        VAF_test = (1  - (var_error/var_y)) * 100
        print('The %VAF is', VAF_test)

        plt.plot(t[j], y_id, linestyle='dashed', label ='y_id', color='blue')
        plt.plot(t[j], y_test, linestyle='dashed', label ='y_test', color='orange')
        
        plt.title('Error plot of the ' + str(i) +'th model on the ' + str(j) + 'th dataset')
        # Set the x axis label
        plt.xlabel('time (sec)')
        # Set the y axis label
        plt.ylabel('forces (kN)')

        plt.legend()

        # plt.show()

# (1, 0)
for i in range(5):
    y_bar = max(abs(y[i]))
    u_bar = max(abs(u[i]))

    print(f'y max/u max is {(y_bar / u_bar)}, and x coefficients are {x1_0[i]}')

    # Create discrete model using coefficients found above 
    P_d = (y_bar / u_bar) * control.tf(np.array([x1_0[i][1]]), np.array([1, x1_0[i][0]]), 0.01)
    print('The discrete-time TF is,', P_d)

    # Discrete to Continuous System ID TF 
    P_c =  d2c.d2c(P_d)
    print('The continuous-time TF is,', P_c)
    print(i)
    VAF_average = np.zeros(5)

    # Calculate the % VAF for each dataset to sit how the model fits the dataset.
    for j in range(5):
        y_test = y[j]
        N = y_test.size
        t_id, y_id = control.forced_response(P_d, t[j], u[j])

        # find the absolute errors.
        e = y_id - y_test

        # find the variances, to find VAF
        var_error = (1/N)*(np.matmul(e.transpose(), e))
        var_y = (1/N)*(np.matmul(y_test.transpose(), y_test))

        VAF_test = (1  - (var_error/var_y)) * 100
        VAF_average[j] = VAF_test
        print('The %VAF is', VAF_test)

        plt.plot(t[j], y_id, linestyle='dashed', label ='y_id', color='blue')
        plt.plot(t[j], y_test, linestyle='dashed', label ='y_test', color='orange')
        
        plt.title('response plot of the ' + str(i) +'th model on the ' + str(j) + 'th dataset')
        # Set the x axis label
        plt.xlabel('time (sec)')
        # Set the y axis label
        plt.ylabel('forces (kN)')

        plt.legend()

        plt.show()


    
    print('the average %VAF is ' + str(np.mean(VAF_average)))




# plotting the relative and absolute errors of the nominal plant

P_d1 = (y_bar / u_bar) * control.tf(np.array([x1_0[4][1]]), np.array([1, x1_0[4][0]]), 0.01)
P_c =  d2c.d2c(P_d)

t_id, y_id = control.forced_response(P_d, t[4], u[4])
y_test = y[4]
e = y_id - y_test
print(e)
e_relative = 2*e/(np.abs(y_id) + np.abs(y_test))
print(e_relative)


plt.plot(t[4], y_id, linestyle='dashed', label ='y_id', color='blue')
plt.plot(t[4], y_test, linestyle='dashed', label ='y_test', color='orange')

plt.title('response plot of the nominal model on the its training dataset')
# Set the x axis label
plt.xlabel('time (sec)')
# Set the y axis label
plt.ylabel('forces (kN)')

plt.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t_id, e, linestyle='dashed', color='blue')
ax1.set_title('the absolute error of the nominal plant on its training dataset')
ax1.set_xlabel('time (sec)')
ax1.set_ylabel('error (kN)')
ax2.plot(t_id, e_relative, linestyle = 'dashed', color = 'red')
ax2.set_title('the relative error of the nominal plant on its training dataset')
ax2.set_xlabel('time (sec)')
ax2.set_ylabel('relative difference')
plt.subplots_adjust(hspace = 0.5)
plt.show()




