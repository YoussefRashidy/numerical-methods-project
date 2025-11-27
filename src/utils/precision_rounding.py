import decimal

def toDecimal(array) :
    # flatten the array convert floats to decimal and reshape to original shape
    originalShape = array.shape
    array = array.flatten().astype(object) 
    for i in range(array.size) :
        array[i] = decimal.Decimal(str(array[i]))
    array = array.reshape(originalShape)   
    return array


def intializeContext(significantFigs,rounding) :
    #Intialize the context with specified signFigs and rounding/chopping
    decimal.getcontext().prec = significantFigs 
    if( rounding ) :
        decimal.getcontext().rounding = decimal.ROUND_HALF_UP
    else :
        decimal.getcontext().rounding = decimal.ROUND_DOWN 
        
def toFloats (array) :
    #Convert the decimals back to float
    originalShape = array.shape
    array = array.flatten().astype(object)
    for i in range(array.size) :
        array[i] = float(array[i]) 
    array = array.reshape(originalShape)    
    return array            
    