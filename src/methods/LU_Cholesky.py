import math

def choleskyDecomposition(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            sum_val = 0.0

            if j == i: # Diagonal
                for k in range(j):
                    sum_val += lower[j][k]**2
                
                temp = matrix[j][j] - sum_val
                if temp <= 0:
                    raise ValueError("Matrix is not Positive Definite!")
                lower[j][j] = math.sqrt(temp)
            
            else: # Off-Diagonal
                for k in range(j):
                    sum_val += lower[i][k] * lower[j][k]
                
                lower[i][j] = (matrix[i][j] - sum_val) / lower[j][j]
                
    return lower