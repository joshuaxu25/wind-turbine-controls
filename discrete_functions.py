import numpy as np

"""
Form the A and b matrix depending on the order we choose
Functions are named with n, m as the two numbers in that order

"""

# ODE of the form y1 + a0*y0 = b0*u0, 
# where n = 1, m = 0
# y1 = -a0*y0+b0*u0
# b is formed by y1, A is formed by -y0 and u0
def DiscreteTime1_0 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[1:]

    # Create the A matrix
    A = np.zeros((N-1,2))
    A[:,0] = -yk[:N-1]
    A[:,1] = uk[:N-1]
    

    return A,b

# ODE of the form y2 + a1*y1 + a0*y0 = b0*u0, 
# where n = 2, m = 0
# y2 = -a1*y1 - a0*y0 + b0*u0
# b is formed by y2, A is formed by -y1, -y0, u0
def DiscreteTime2_0 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[2:]

    # Create the A matrix
    A = np.zeros((N-2,3))
    A[:,0] = -yk[1:N-1]
    A[:,1] = -yk[:N-2]
    A[:,2] = uk[:N-2]

    #verify A is full column rank
    A_T = np.transpose(A)
    print("the determinant of A_T*A is " + str(np.linalg.det(np.matmul(A_T,A))))


    # Verify if the problem is well-conditioned
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_cond = np.max(S)/np.min(S)
    print("The conditional of A is", A_cond)

    return A,b

# ODE of the form y2 + a1*y1 + a0*y0 = b1*u1 + b0*u0,
# where n = 2, m = 1
# y2 = -a1*y1 - a0*y0 + b1*u1 + b0*u0
# b is formed by y2, A is formed by -y1, -y0, u1, u0
def DiscreteTime2_1 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[2:]

    # Create the A matrix
    A = np.zeros((N-2,4))
    A[:,0] = -yk[1:N-1]
    A[:,1] = -yk[:N-2]
    A[:,2] = uk[1:N-1]
    A[:,3] = uk[:N-2]

    # Verify A is full column rank
    A_T = np.transpose(A)
    print("the determinant of A_T*A is " + str(np.linalg.det(np.matmul(A_T,A))))

    # Verify if the problem is well-conditioned
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_cond = np.max(S)/np.min(S)
    print("The conditional of A is", A_cond)

    return A,b

# ODE of the form y3 + a2*y2 + a1*y1 + a0*y0 = b0*u0, 
# where n = 3, m = 0
# y3 = - a2*y2 - a1*y1 - a0*y0 + b0*u0
# b is formed by y3, A is formed by -y2, -y1, -y0, u0
def DiscreteTime3_0 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[3:]

    # Create the A matrix
    A = np.zeros((N-3,4))
    A[:,0] = -yk[2:N-1]
    A[:,1] = -yk[1:N-2]
    A[:,2] = -yk[:N-3]
    A[:,3] = uk[:N-3]

    # Verify A is full column rank
    A_T = np.transpose(A)
    print("the determinant of A_T*A is " + str(np.linalg.det(np.matmul(A_T,A))))


    # Verify if the problem is well-conditioned
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_cond = np.max(S)/np.min(S)
    print("The conditional of A is", A_cond)


    return A,b

# ODE of the form y3 + a2*y2 + a1*y1 + a0*y0 = b1*u1 + b0*u0, 
# where n = 3, m = 1
# y3 = - a2*y2 - a1*y1 - a0*y0 + b1*u1 + b0*u0
# b is formed by y3, A is formed by -y2, -y1, -y0, u1, u0
def DiscreteTime3_1 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[3:]

    # Create the A matrix
    A = np.zeros((N-3,5))
    A[:,0] = -yk[2:N-1]
    A[:,1] = -yk[1:N-2]
    A[:,2] = -yk[:N-3]
    A[:,3] = uk[1:N-2]
    A[:,4] = uk[:N-3]

    # Verify A is full column rank
    A_T = np.transpose(A)
    print("the determinant of A_T*A is " + str(np.linalg.det(np.matmul(A_T,A))))


    # Verify if the problem is well-conditioned
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_cond = np.max(S)/np.min(S)
    print("The conditional of A is", A_cond)    



    return A,b

# ODE of the form y3 + a2*y2 + a1*y1 + a0*y0 = b2*u2 + b1*u1 + b0*u0, 
# where n = 3, m = 2
# y3 = - a2*y2 - a1*y1 - a0*y0 + b2*u2 * b1*u1 + b0*u0
# b is formed by y3, A is formed by -y2, -y1, -y0, u2, u1, u0
def DiscreteTime3_2 (y,u):
    y_max = max(abs(y))
    u_max = max(abs(u))

    yk = y/y_max
    uk = u/u_max
    N = y.size

    # Create the b matrix
    b = yk[3:]

    # Create the A matrix
    A = np.zeros((N-3,6))
    A[:,0] = -yk[2:N-1]
    A[:,1] = -yk[1:N-2]
    A[:,2] = -yk[:N-3]
    A[:,3] = uk[2:N-1]
    A[:,4] = uk[1:N-2]
    A[:,5] = uk[:N-3]

    # Verify A is full column rank
    A_T = np.transpose(A)
    print("the determinant of A_T*A is " + str(np.linalg.det(np.matmul(A_T,A))))

    # Verify if the problem is well-conditioned
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_cond = np.max(S)/np.min(S)
    print("The conditional of A is", A_cond)


    return A,b