import numpy as np
TOLERANCE = 1e-5

def floor_precision(num, precision):
    return np.floor(num * 10**precision) / 10**precision

def scaled_partial_pivot(aug_mat:np.array, col):
    relative_mag = [row[col] / row.max() for row in aug_mat]
    max = -float('inf')
    max_col = None
    for i in range(len(relative_mag)):
        if relative_mag[i] > max:
            max_col = i
            max = relative_mag[i]
    aug_mat[[col, max_col]] = aug_mat[[max_col, col]]
    return aug_mat

def partial_pivot(aug_mat:np.array, col):
    max_each_col = aug_mat.argmax(axis=0)
    aug_mat[[col, max_each_col[col]]]= aug_mat[[max_each_col[col], col]] 
    return aug_mat


def gaussian_elimination(aug_mat: np.array, precision: int, pivot: int = 0):
    row_num, col_num =aug_mat.shape
    dim = col_num - 1
    if row_num < dim:
        print('under determined')
        return []
    elif row_num > dim:
        print('over determined')
        return []

    for i in range(dim):
        if pivot == 1:
            aug_mat = partial_pivot(aug_mat, i)
        elif pivot == 2:
            aug_mat = scaled_partial_pivot(aug_mat, i)
        for row in range(dim):
            if i == row: continue
            # aug_mat[row] = floor_precision(aug_mat[row], precision)
            if precision is None:
                aug_mat[row] = np.subtract(aug_mat[row],(aug_mat[i] * (aug_mat[row][i] /aug_mat[i][i])))
            else:
                aug_mat[row] = floor_precision(np.subtract(aug_mat[row], floor_precision(aug_mat[i] * floor_precision(aug_mat[row][i] /aug_mat[i][i], precision), precision)),precision)
        print(f'no.{i} iteration:')
        print(aug_mat)
        print('\n')
    
    solution = []
    for i in range(dim):
        if precision is None:
            solution.append(aug_mat[i][col_num - 1] / aug_mat[i][i])
        else:
            solution.append(floor_precision(aug_mat[i][col_num - 1] / aug_mat[i][i], precision))


    return solution



def LU_factorize(A: np.array, n):
    dim = A.shape[0]
    A = A*(1/n)
    I_n = np.identity(dim) * n

    concat_mat = np.concatenate((I_n, A), axis= 1)
    print(concat_mat)
    print('\n')

    for i in range(dim, 2*dim):
        for j in range(i-dim, dim):
            if j == (i-dim): continue
            concat_mat[j] = np.subtract(concat_mat[j], concat_mat[i-dim] * (concat_mat[j][i]/ concat_mat[i-dim][i]))
        print(f'no.{i-dim} iteration:')
        print(concat_mat)
        print('\n')

    L = concat_mat[:, 0:dim]
    U = concat_mat[:, dim:2*dim]

    return L, U



def Jacobi_mat(A: np.array, b):
    L_plus_U = A.copy()
    D = A.copy()

    for row in range(A.shape[0]):
        for col in range(A.shape[1]):
            if row == col : 
                L_plus_U[row][col] = 0
            else:
                D[row][col] = 0

    D_inv = np.linalg.inv(D)

    update = lambda x: D_inv.dot(b - L_plus_U.dot(x))
    x = np.array([0, 0, 0])

    iterations = 0
    while not np.allclose(A.dot(x), b, atol=TOLERANCE, rtol=0):
        iterations += 1
        x = update(x)
    
    return x, A.dot(x)-b, iterations


def Jacobi(A: np.array, b, x):

    n = A.shape[0]
    iterations = 0
    while not np.allclose(A.dot(x), b, atol=TOLERANCE, rtol=0) and iterations < 1000:
        print(x)
        x_next = x.copy()
        iterations += 1
        for i in range(n):  
            x_term = 0
            for j in range(0, n):
                if i == j: continue
                x_term += A[i,j] * x[j]

            x_next[i] = (b[i] - x_term) * (1/ A[i,i])

        x = x_next

    return x, A.dot(x)-b, iterations



def gauss_seidel(A: np.array, b, x):

    n = A.shape[0]
    iterations = 0
    while not np.allclose(A.dot(x), b, atol=TOLERANCE, rtol=0)  and iterations < 1000:
        print(x)
        iterations += 1
        for i in range(n):  
            x_term = 0
            for j in range(0, n):
                if i == j: continue
                x_term += A[i,j] * x[j]

            x[i] = (b[i] - x_term) * (1/ A[i,i])

    return x, A.dot(x)-b, iterations