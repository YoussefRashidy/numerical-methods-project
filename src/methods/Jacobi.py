from src.utils.precision_rounding import toDecimal, toFloats, intializeContext
import decimal

def isDiagonalyDominant(A) :
    #Check if Matrix A is diagonaly dominant 
    dim = A.shape
    # Flags to check system
    at_least_one_strictly_greater = None 
    if(dim[0] != dim[1]) : raise Exception("Matrix must be square")
    for i in range(dim[0]) :
        nonDiagonalSum = 0 
        for j in range(dim[0]) :
            if(j==i) : continue 
            nonDiagonalSum += abs(A[i][j]) 
        if(A[i][i] > nonDiagonalSum) :
            at_least_one_strictly_greater = True
        elif(A[i][i] == nonDiagonalSum) :
            if(at_least_one_strictly_greater == None) : at_least_one_strictly_greater = False
        else :
            return False    
    return at_least_one_strictly_greater            
            
# Jacobi with relaxation for improved convergence (called weighted Jacobi)
# the parameter n can be removed as we can get the no of vars from the npArray
def Jacobi_noNorm(A,b,n,x,maxIterations,ErrorTolerance,relax = 1 , significantFigs = 7 , rounding = True) :
    # Setting up signifcant figs and rounding/chopping
    intializeContext(significantFigs,rounding)
    #Copying the arrays to avoid modifying original arrays
    A = A.copy()
    b = b.copy()
    x = x.copy()
    #converting floats to decimals 
    A = toDecimal(A)
    b = toDecimal(b)
    x = toDecimal(x)
    relax = decimal.Decimal(str(relax))
    #Array to hold new values
    xNew = np.zeros(n,dtype=object)
    #List to hold iteration details
    iteration_details = []
    # Calculating first iteration before applying relaxation    
    for i in range(n) :
        sum = b[i]
        for j in range(n) :
            if(i==j) :
                continue
            sum -= A[i][j] * x[j]
        xNew[i] = sum/A[i][i]
    x = xNew.copy()
     # Storing details of first iteration    
    details = {
            'iteration' : 1,
            'xNew' : toFloats(xNew),
            'maxError' : '_'
        }     
    iteration_details.append(details)
    iteration = 2
    # Loop until convergence or max iterations reached 
    while (True) :
        belowTolerance = True
        maxError = decimal.Decimal('0')
        for i in range(n) :
            oldX = x[i]
            sum = b[i]
            for j in range(n) :
                if(i==j) :
                    continue
                sum -= A[i][j] * x[j]
            xNew[i] = relax*sum/A[i][i] + (1-relax)*oldX
            if (belowTolerance and xNew[i] != 0) :
                estimatedError = abs((xNew[i]-oldX)/xNew[i]) * 100
                maxError = max(maxError, estimatedError)
                if(estimatedError > ErrorTolerance):
                    belowTolerance = False
        details = {
            'iteration' : iteration,
            'xNew' : toFloats(xNew),
            'maxError' : float(maxError)
        }         
        iteration_details.append(details)
        iteration+=1
        x = xNew.copy()
        if(belowTolerance or iteration >= maxIterations) :
            break
    return toFloats(xNew) , iteration_details            