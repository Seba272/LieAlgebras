# Various functions depending on LieAlgebra5.py
# that could be implemented later in LieAlgebra5


def left_invariant_vf(la:LieAlgebra,p:Array,smply=False):
    """Returns a left-invariant frame

p is an array of shape (dim,1).
la is a Lie algebra.
if smply==True, a simplify command is applied before returning the vectors.

The result is obtained by writing two vectors x and y,
apply xy = bch(x,y)
and differentiate xy in y, 
and substitute y->0 and x->p.
Here there is a simplification, if smply==True.
    """
    dim = la.dimension
    x = la.a_vector_array('x')
    y = la.a_vector_array('y')
    # x.shape = (dim, 1)
    xy = la.bch(x,y)
    y_to_zero = {y[j,0]:0 for j in range(dim)}
    x_to_p = {x[j,0]:p[j,0] for j in range(dim)}
    livf = []
    for i in range(dim):
        xyi = diff(xy,y[i]).reshape(dim,1)
        xyi = xyi.subs(y_to_zero).subs(x_to_p)
        if smply:
            simplify(xyi)
        livf.append( xyi )
    return livf

def right_invariant_vf(la:LieAlgebra,p:Array,smply=False):
    """Returns a right-invariant frame

see also left_invariant_vf().

p is an array of shape (dim,1).
la is a Lie algebra.
if smply==True, a simplify command is applied before returning the vectors.

The result is obtained by writing two vectors x and y,
apply xy = bch(x,y)
and differentiate xy in x, 
and substitute x->0 and y->p.
Here there is a simplification, if smply==True.
    """
    dim = la.dimension
    x = la.a_vector_array('x')
    y = la.a_vector_array('y')
    # x.shape = (dim, 1)
    xy = la.bch(x,y)
    y_to_p = {y[j,0]:p[j,0] for j in range(dim)}
    x_to_zero = {x[j,0]:0 for j in range(dim)}
    livf = []
    for i in range(dim):
        xyi = diff(xy,x[i]).reshape(dim,1)
        xyi = xyi.subs(x_to_zero).subs(y_to_p)
        if smply:
            simplify(xyi)
        livf.append( xyi )
    return livf

def left_invariant_coframe(la:LieAlgebra,p=None):
    if type(p) is Array:
        x = p
    else:
        x = la.a_vector_array('x')
    dim = la.dimension
    y = la.a_vector_array('y')
    y_to_x = {y[j,0]:x[j,0] for j in range(dim)}
    xinv_y = la.bch(-x,y)
    DLxinv_atx = diff(xinv_y,y).reshape(dim,dim).transpose().subs(y_to_x)
    coframe0 = eye(dim).rowspace()
    coframe = [v.multiply(DLxinv_atx) for v in coframe0]
    return coframe

def D_left_tranaslation(la:LieAlgebra,p,q=None):
    """Returns the differential at q of the left-translation by p.
    
    """
    dim = la.dimension
    x = la.a_vector_array('x')
    y = la.a_vector_array('y')
    _p = p.copy()
    if q == None:
        _q = 0*x
    else:
        _q = q.copy()
    x_to_p = {x[j,0]:_p[j,0] for j in range(dim)}
    y_to_q = {y[j,0]:_q[j,0] for j in range(dim)}
    xy = la.bch(x,y)
    Dxy = diff(xy,y).reshape(dim,dim).transpose()
    DLp_at_q = Dxy.subs(x_to_p).subs(y_to_q)
    return DLp_at_q

# For blow-up of intrinsic graphs:
# implement:
# \delta_t( f(w)^{-1} f( w f(w) (\delta_{1/t}u) f(w)^{-1} ) )
def ff(la,f,t,w,u):
    res = la.dil(1/t,u)
    right = -f(w)
    left = la.bch(w,f(w))
    res = la.bch(res,right)
    res = la.bch(left,res)
    res = f(res)
    left = right
    res = la.bch(left,res)
    res = la.dil(t,res)
    return res

def show_bra(la):
    basis = la.basis_symbolic
    dim = la.dimension
    M = zeros(dim+1,dim+1)
    for i in range(dim):
        M[i+1,0] = M[0,i+1] = basis[i]
    for i in range(dim):
        for j in range(dim):
    #         print(i,j)
            M[i+1,j+1] = la(basis[i],basis[j])
    display(M)
    
from tabulate import tabulate

def show_bra_tab(la):
    basis = la.basis_symbolic
    dim = la.dimension
    M = zeros(dim+1,dim+1)
    for i in range(dim):
        M[i+1,0] = M[0,i+1] = basis[i]
    for i in range(dim):
        for j in range(dim):
    #         print(i,j)
            M[i+1,j+1] = la(basis[i],basis[j])

    M_list = [[M[i,j] for j in range(M.cols)] for i in range(M.rows)]
    tabella = tabulate(M_list,headers="firstrow", tablefmt="html")
    display(tabella)
    
def export_bra(la):
    basis = la.basis_symbolic
    dim = la.dimension
    list_bras = []
    for j in range(dim):
        for i in range(j):
            a = basis[i]
            b = basis[j]
            ab = la(a,b)
            if ab != 0*a:
                list_bras.append('(' + str(a) + ', ' + str(b) + ') :' + str(ab))
    return list_bras
