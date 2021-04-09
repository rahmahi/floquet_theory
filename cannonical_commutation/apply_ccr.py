import sympy
from sympy import *
from sympy.physics.quantum import *
from sympy.core.operations import AssocOp

def apply_ccr(expr, ccr, reverse=False):
    if not isinstance(expr, Basic):
        raise TypeError("The expression to simplify is not a sympy expression.")
        
    if not isinstance(ccr, Eq):
        if isinstance(ccr, Basic):
            ccr = Eq(ccr, 0)
        else:
            raise TypeError("The canonical commutation relation is not a sympy expression.")
    
    comm = None
    
    for node in preorder_traversal(ccr):
        if isinstance(node, Commutator):
            comm = node
            break
            
    if comm is None:
        raise ValueError("The cannonical commutation relation doesn not include a commutator.")
        
    solutions = solve(ccr, comm)
    
    if len(solutions) != 1:
        raise ValueError("There are more solutions to the cannonical commutation relation.")
        
    value = solutions[0]
    
    A = comm.args[0]
    B = comm.args[1]
    
    if reverse:
        (A, B) = (B, A)
        value = -value
    
    def is_expandable_pow_of(base, expr):
        return isinstance(expr, Pow) \
            and base == expr.args[0] \
            and isinstance(expr.args[1], Number) \
            and expr.args[1] >= 1
    
    
    def walk_tree(expr):
        if not isinstance(expr, AssocOp) and not isinstance(expr, Function):
            return expr.copy()
        
        elif not isinstance(expr, Mul):
            return expr.func(*(walk_tree(node) for node in expr.args))
        
        else:
            args = [arg for arg in expr.args]
            
            for i in range(len(args)-1):
                x = args[i]
                y = args[i+1]
                
                if B == x and A == y:
                    args = args[0:i] + [A*B - value] + args[i+2:]
                    return walk_tree( Mul(*args).expand() )
                
                if B == x and is_expandable_pow_of(A, y):
                    ypow = Pow(A, y.args[1] - 1)
                    args = args[0:i] + [A*B - value, ypow] + args[i+2:]
                    return walk_tree( Mul(*args).expand() )
                
                if is_expandable_pow_of(B, x) and A == y:
                    xpow = Pow(B, x.args[1] - 1)
                    args = args[0:i] + [xpow, A*B - value] + args[i+2:]
                    return walk_tree( Mul(*args).expand() )
                
                if is_expandable_pow_of(B, x) and is_expandable_pow_of(A, y):
                    xpow = Pow(B, x.args[1] - 1)
                    ypow = Pow(A, y.args[1] - 1)
                    args = args[0:i] + [xpow, A*B - value, ypow] + args[i+2:]
                    return walk_tree( Mul(*args).expand() )
            
            return expr.copy()
            
    
    return walk_tree(expr)
   

Basic.apply_ccr = lambda self, ccr, reverse=False: apply_ccr(self, ccr, reverse)
