import sympy as sp
import decimal

def parse_exp(expr_str , var_name='x') :
    x = sp.symbols(var_name)
    expr = sp.sympify(expr_str)
    
    def decimal_func(x_decimal) :
        if not isinstance(x_decimal,decimal.Decimal) :
            x_decimal = decimal.Decimal(x_decimal)
            
        result = expr.evalf(decimal.getcontext().prec,subs={x:x_decimal})
        return decimal.Decimal(str(result))
    
    return decimal_func   

def parse_derivative(expr_str , var_name='x' ) :
    x = sp.symbols(var_name)
    expr = sp.sympify(expr_str)
    deriv = sp.diff(expr)
    
    def decimal_derv(x_decimal) :
        if not isinstance(x_decimal,decimal.Decimal) :
            x_decimal = decimal.Decimal(x_decimal)
        result = deriv.evalf(decimal.getcontext().prec,subs = {x:x_decimal})
        return decimal.Decimal(str(result))
    return decimal_derv
    