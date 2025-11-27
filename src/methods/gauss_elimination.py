import numpy as np
import copy

def Pivoting(A, b, s, n, k, steps):
# A: Coef. of matrix A; 2-D array
# b: Coef. of vector b; 1-D array it is None if i didn't pass one
# n: Dimension of the system of equations
# s: n-element array for storing scaling factors
# k: is the row we are in now

  # initializing p (our pivoting row) with the value of k
  p = k     
    
  if s is not None:

    # Finding the largest scaled coefficient in column k
    big = abs(A[k][k] / s[k])
    for i in range(k+1,n) :
      dummy = abs(A[i][k] / s[i]) # dummy number for the scaled value
      if dummy > big:
        big = dummy
        p = i # new pivoting row

    # Swap row p and row k if p != k
    if p != k:
      # Swaping row p and row k
      
      steps.append("A[" + str(p) + "] <-> A[" + str(k) + "]")
      
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))
      else:
        steps.append(copy.deepcopy(A))
        
      dummy = s[p]
      s[p] = s[k]
      s[k] = dummy
  else:
  # Finding the largest coefficient in column k
    big = abs(A[k][k])
    for i in range(k+1,n) :
      dummy = abs(A[i][k] ) # dummy number for the  value
      if dummy > big:
        big = dummy
        p = i # new pivoting row

    # Swap row p and row k if p != k
    if p != k:
      steps.append("A[" + str(p) + "] <-> A[" + str(k) + "]")
      for j in range(k,n) :
        dummy = A[p][j]
        A[p][j] = A[k][j]
        A[k][j] = dummy
      if b is not None:
        dummy = b[p]
        b[p] = b[k]
        b[k] = dummy
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))
      else:
        steps.append(copy.deepcopy(A))


def forward_elimination(A, b, s, n, tol, steps):
# A: Coef. of matrix A; 2-D array
# b: Coef. of vector b; 1-D array it is None if i didn't pass one
# n: Dimension of the system of equations
# tol: Tolerance; smallest possible scaled pivot allowed
# s: n-element array for storing scaling factors

  if s is not None and b is not None:
    for k  in range(n-1):
      Pivoting(A, b, s, n, k, steps) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append(f"A[{i}] -> A[{i}] + {round(-factor, 6)} * A[{k}]")
        for j in range(k,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        b[i] = b[i] - factor * b[k]
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  elif s is None and b is not None: # no scaling
    for k  in range(n-1):
      Pivoting(A, b, None, n, k) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append(f"A[{i}] -> A[{i}] + {round(-factor, 6)} * A[{k}]")
        for j in range(k,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        b[i] = b[i] - factor * b[k]
        steps.append((copy.deepcopy(A), copy.deepcopy(b)))

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  elif s is not None and b is None:
    for k  in range(n-1):
      Pivoting(A, None, s, n, k) # Partial Pivoting
      if abs(A[k][k] / s[k]) < tol:  # Check for singularity
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        steps.append(f"A[{i}] -> A[{i}] + {round(-factor, 6)} * A[{k}]")
        for j in range(k,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        steps.append(copy.deepcopy(A))

    if abs(A[n-1][n-1]/s[n-1]) < tol: # Check for singularity
      return -1
  else: # no scaling, b is None
    for k  in range(n-1):
      Pivoting(A, None, None, n, k) # Partial Pivoting
      if abs(A[k][k]) < tol:
        return -1
      for i in range(k+1,n):
        factor = A[i][k] / A[k][k]
        if abs(factor) < tol :
            continue
        steps.append(f"A[{i}] -> A[{i}] + {round(-factor, 6)} * A[{k}]")
        for j in range(k,n):
          A[i][j] = A[i][j] - factor * A[k][j]
        steps.append(copy.deepcopy(A))

    if abs(A[n-1][n-1]) < tol: # Check for singularity
      return -1
  return 0 # Return 0 for successful elimination


def back_substitution(A, b, n):
  x = [0.0 for _ in range(n)]
  x[n-1] = b[n-1] / A[n-1][n-1]
  for i in range(n-2,-1,-1):
    sum_val = 0.0
    for j in range(i+1,n):
      sum_val = sum_val + A[i][j] * x[j]
    x[i] = (b[i] - sum_val) / A[i][i]
  return x

def gauss_elimination(A, b , n, tol, scaling = False):
# A: Coef. of matrix A; 2-D array
# b: Coef. of vector b; 1-D array it is None if i didn't pass one
# n: Dimension of the system of equations
# tol: Tolerance; smallest possible scaled pivot allowed
# scaling: A boolean showing we want scaling or not
  steps = [] # initializing the steps list
  if b is not None:
      steps.append((copy.deepcopy(A), copy.deepcopy(b)))
  else:
      steps.append(copy.deepcopy(A))
      
      
      
  if scaling:
    s = [] # as an n-element array for storing scaling factors
    for i in range(n):
      s.append(abs(A[i][0]))
      for j in range(1,n) :
        if abs(A[i][j]) > s[i]:
          s[i] = abs(A[i][j])
    temp = forward_elimination(A, b, s, n, tol, steps)  # forward elimination
    
  else: # no scalling
    temp = forward_elimination(A, b, None, n, tol, steps)  # forward elimination, passing None for s
      
    # If forward elimination returned singular pivot
    if temp == -1:
        return None
    
    rankA = 0
    rankAb = 0

    for i in range(n):
        if not np.allclose(A[i], 0, atol=1e-12):
            rankA += 1
        if b is not None:
            if not np.allclose(np.append(A[i], b[i]), 0, atol=1e-12):
                rankAb += 1

    # no solution
    if b is not None and rankA < rankAb:
        return None

    # infinite number of solutions
    if rankA < n:
        return 2

    # Unique solution
    if b is not None:
        return (back_substitution(A, b, n), steps)
    else:
        return (A, steps)

