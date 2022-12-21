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
