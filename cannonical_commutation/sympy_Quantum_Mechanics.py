from __future__ import division
import sys
#sys.path.insert(0, "~/.local/lib/python3.6/site-packages")

import sympy
from sympy import *
from sympy.physics.quantum import *

hbar = Symbol('hbar')

x = Operator('x')
p = Operator('p')
ccr = Eq( Commutator(x, p), I*hbar )

expr = x * (x**2 * p * x**2 + 1)
expr.expand().apply_ccr(ccr)
# => 3iħ x^4 + p x^5 + x


H = Operator('H')
a = Operator('a')
ad = a.adjoint()
n = Symbol('n')
O = Symbol('Omega')

up     = Eq( ad*Ket(n), sqrt(n+1)*Ket(n+1) )
down   = Eq( a *Ket(n), sqrt(n)  *Ket(n-1) )
energy = Eq( H *Ket(n), hbar*O*( n + 1/2 )*Ket(n) )

expr = a * H * ad * Ket(n)
expr.apply_operator(up, down, energy)
# => Ωℏn^2 |n⟩ + 2.5Ωℏn |n⟩ + 1.5Ωℏ |n⟩