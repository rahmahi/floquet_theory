import sympy
from sympy import *
from sympy.physics.quantum import *
from sympy.core.operations import AssocOp

def apply_operator(expr, eqns):
    if not isinstance(expr, Basic):
        raise TypeError("The expression to simplify is not a sympy expression.")
    
    if not isinstance(eqns, list) and not isinstance(eqns, tuple):
        eqns = (eqns,)
    
    
    rules = []
    
    
    class Rule(object):
        operator = None
        ketSymbol = None
        result = None
        generic = False
    
    
    def is_operator(op):
        return isinstance(op, Operator) \
        or isinstance(op, Dagger) \
        and isinstance(op.args[0], Operator)
    
    
    for eqn in eqns:
        if not isinstance(eqn, Eq):
            raise TypeError("One of the equations is not a valid sympy equation.")
        
        lhs = eqn.lhs
        rhs = eqn.rhs
        
        if not isinstance(lhs, Mul) \
        or len(lhs.args) != 2 \
        or not is_operator(lhs.args[0]) \
        or not isinstance(lhs.args[1], KetBase):
            raise ValueError("The left-hand side has to be an operator applied to a ket.")
        
        rule = Rule()
        rule.operator = lhs.args[0]
        rule.ketSymbol = lhs.args[1].args[0]
        rule.result = rhs
        
        if not isinstance(rule.ketSymbol, Symbol):
            raise ValueError("The left-hand ket has to contain a simple symbol.")
        
        for ket in preorder_traversal(rhs):
            if isinstance(ket, KetBase):
                for symb in preorder_traversal(ket):
                    if symb == rule.ketSymbol:
                        rule.generic = True
                        break
                        
        rules.append(rule)
    
    
    def is_expandable_pow_of(base, expr):
        return isinstance(expr, Pow) \
            and base == expr.args[0] \
            and isinstance(expr.args[1], Number) \
            and expr.args[1] >= 1
            
            
    def is_ket_of_rule(ket, rule):
        if not isinstance(ket, KetBase):
            return False
        
        if rule.generic:
            for sym in preorder_traversal(ket):
                if sym == rule.ketSymbol:
                    return True
            return False
                
        else:
            return ket.args[0] == rule.ketSymbol
    
    
    def walk_tree(expr):
        if not isinstance(expr, AssocOp) and not isinstance(expr, Function):
            return expr.copy()
        
        elif not isinstance(expr, Mul):
            return expr.func(*(walk_tree(node) for node in expr.args))
        
        else:
            args = [arg for arg in expr.args]
            
            for rule in rules:
                A = rule.operator
                ketSym = rule.ketSymbol
                
                for i in range(len(args)-1):
                    x = args[i]
                    y = args[i+1]

                    if A == x and is_ket_of_rule(y, rule):
                        ev = rule.result
                        
                        if rule.generic:
                            ev = ev.subs(rule.ketSymbol, y.args[0])
                        
                        args = args[0:i] + [ev] + args[i+2:]
                        return walk_tree( Mul(*args).expand() )

                    if is_expandable_pow_of(A, x) and is_ket_of_rule(y, rule):
                        xpow = Pow(A, x.args[1] - 1)
                        ev = rule.result
                        
                        if rule.generic:
                            ev = ev.subs(rule.ketSymbol, y.args[0])
                        
                        args = args[0:i] + [xpow, ev] + args[i+2:]
                        return walk_tree( Mul(*args).expand() )
                
            
            return expr.copy()
            
    
    return walk_tree(expr)
   

Basic.apply_operator = lambda self, *eqns: apply_operator(self, eqns)
