from sympy import *

def build_monomials_nc(variables,order):
        """
        Build a list of lists of non commutative monomials.

        Inputs:
        -------
        *variables*: list of (non commutative) variables
        *order*: maximal order we are interested in

        Output:
        -------
        a list ``monomials``, where ``monomials[k]`` is the list of all monomials of order ``k``, for ``k`` from 0 to *order*.
        Notice that ``len(monomials) = order + 1``.

        Examples
        ========

        >>> x,y,z = symbols('x y z',commutative=False)
        >>> monomials = build_monomials_nc([x,y,z],2)
        >>> monomials[0]
        [1]
        >>> monomials[1]
        [x,y]
        >>> monomials[2]
        [x**2,x*y,y*x,y**2]
        >>> len(monomials)
        3

        """
        monomials = []
        if order >= 0:
            monomials = [[1]]
        if order >= 1:
            monomials += [variables]
        if order > 1:
            for j in range(order-1):
                monomials += [[ a*b for a in monomials[-1] for b in variables ]]
        return monomials

def isubs(expr,rules,MAX=100): 
    """
    Iterated Symbolic Substitions.

    Applies *rules* to *expr* iteratively until *expr* does not change anymore,
    or *MAX* iterations are done.
    The output consists of a pair: the simplified expression and the number of iterations performed.
    If the number of iteration is equal to *MAX*, it means that the simplification was not complete.
    Otherwise, the number of iterations is between 1 and *MAX*-1.

    NB! ``.subs(rules)`` applies rules only once. This is why we need an iterated version.

    NB! Every time rules are applied, the expression is also expanded.
    
    Examples
    ========

    (These examples need to be re-run)

    >>> x,y,z = symbols('x y z',commutative=False)
    >>> rules = {y*x: x*y - z, z*x: x*z, z*y: y*z}
    >>> isubs(y*x,rules)
    (x*y - z, 2)
    >>> isubs(y*x*x*x*x*x*x*x*x*x*x*x*x*x,rules,100)
    (-13*x**12*z + x**13*y, 14)
    >>> isubs(y*x*x*x*x*x*x*x*x*x*x*x*x*x,rules,5)
    max iter reached!
    (-5*x**4*z*x**8 + x**5*y*x**8, 5)

    """
    expr_simplified = expand(expr)
    iterazione = 0
    while iterazione < MAX:
        iterazione+=1
        new_expr_simplified = expand(expr_simplified.subs(rules))
        if new_expr_simplified == expr_simplified :
            return new_expr_simplified, iterazione
        else:
            expr_simplified = new_expr_simplified
    print('Warning from isubs(): Max iterations reached!')
    return expr_simplified, iterazione

def isubs_force(expr,rules,MAX=100): 
    return isubs(expr,rules,MAX)[0]

# questo lo possiamo fare usando ism
def dualize(expr,base_from,base_to):
    dimension = len(base_from)
    for j in range(dimension):
        expr = expr.subs(base_from[j],base_to[j])
    return expr

def build_weight_list(growth_vector):
    """
    Builds the list of weights.

    Given a list *growth_vector*, that is, a list of non-negative integers ``[d1,d2,...]``,
    it builds the list of weigths ``[1,1,...,2,2,...]`` where ``j`` appears ``dj`` times.

    Examples
    ========

    >>> build_weight_list([2,1,0,3])
    [1,1,2,4,4,4]

    """
    step = len(growth_vector)
    weights = []
    for w in range(step):
        weights += [ w+1 for i in range(growth_vector[w]) ]
    return weights

def weight_from_growth_vector(growth_vector,multi_index):
    """
    Compute the weight of a multi-index.
    
    Given a list *growth_vector* and a list *multi_index*, ``weights`` computes the weight of the multi-index.
    Notice that we need ``sum(growth_vector) = len(multi_index)``.

    Mathemattically, a multi-index is a tuple $I = (i_1,\dots,i_n)$ of the length of the dimension ``dim = sum(growth_vector)``.
    The growth vector defines a list of weights $(w_1,\dots,w_n)$ and the weight of $I$ is $w(I) = \sum_{i=1}^n w_iI_i$.

    Examples
    ========

    >>> weight_from_growth_vector([2,1,0,3],(1,2,3,4,5,6)]
    69

    """
    weights = build_weight_list(growth_vector)
    dimension = sum(growth_vector)
    # TODO: check that dimension == len(multi_index)
    res = 0
    for i in range(dimension):
        res += weights[i]*multi_index[i]
    return res

def weight_from_weights(weights,multi_index):
    """
    Compute the weight of a multi-index, using the list of weights.
    
    TODO: Rewrite this:

    Given a list *growth_vector* and a list *multi_index*, ``weights`` computes the weight of the multi-index.
    Notice that we need ``sum(growth_vector) = len(multi_index)``.

    Mathemattically, a multi-index is a tuple $I = (i_1,\dots,i_n)$ of the length of the dimension ``dim = sum(growth_vector)``.
    The growth vector defines a list of weights $(w_1,\dots,w_n)$ and the weight of $I$ is $w(I) = \sum_{i=1}^n w_iI_i$.

    Examples
    ========

    >>> weight_from_weights([1, 1, 2, 4, 4, 4],(1,2,3,4,5,6))
    159

    """
    dimension = len(weights)
    # TODO: check that dimension == len(multi_index)
    res = 0
    for i in range(dimension):
        res += weights[i]*multi_index[i]
    return res

def build_graded_indices_dict(growth_vector, depth):
    """
    Builds a dictionary of indices.

    Builds a dictionary ``I`` so that ``I[k]`` is the list of all tuples ``(i_1,...,i_n)`` (where ``n`` is the dimension, i.e., ``sum(growth_vector)``) whose weight is ``k``.
    ``k`` runs from 1 to *depth*.
    
    NB! ``I[k]`` is ordered how the function ``set`` orders it.

    Examples
    ========

    >>> build_graded_indices_dict([2,1], 3)
    {0: [(0, 0, 0)],
     1: [(1, 0, 0), (0, 1, 0)],
     2: [(0, 2, 0), (1, 1, 0), (0, 0, 1), (2, 0, 0)],
     3: [(1, 0, 1), (1, 2, 0), (2, 1, 0), (3, 0, 0), (0, 3, 0), (0, 1, 1)]}
    >>> build_graded_indices_dict([2],3)
    {0: [(0, 0],
     1: [(1, 0), (0, 1)],
     2: [(1, 1), (2, 0), (0, 2)],
     3: [(0, 3), (1, 2), (2, 1), (3, 0)]}

    """
    dimension = sum(growth_vector)
    step = len(growth_vector)
    weights = build_weight_list(growth_vector)
    graded_indices_dict = {0:[tuple(0 for j in range(dimension))]}
    """
    The algorithm is:
    For j from 1 to *depth*, we construnct I[j] as follows:
    for each k from 1 to dim, 
    we add 1 to at the k-th slot of some multi-index idx of lower weight,
    obtaining a new multi-index idx_new whose weight is w(idx_new) = w(idx) + w[k],
    which must be equal to j.
    So, w(idx) = j-w[k], i.e., idx is in I[j-w[k]].
    If j-w[k] = 0, then we must have idx=0.
    If j-w[k] < 0, then we cannot have any suitable idx.
    """
    for j in range(1,depth+1):
        new_Ij = []
        for k in range(dimension):
            if j-weights[k] < 0:
                continue
            if j-weights[k] == 0:
                idx_n = [0 for j in range(dimension)]
                idx_n[k]+=1
                new_Ij.append(tuple(idx_n))
                continue
            # else:
            for idx in graded_indices_dict[j-weights[k]]:
                idx_n = list(idx)
                idx_n[k]+=1
                new_Ij.append(tuple(idx_n))
        # The previous loop produces duplicates, which are eliminated by ``list(set())``.
        graded_indices_dict[j] = list(set(new_Ij))
    return graded_indices_dict

def mul_list(ll):
    """Multiply elements of a list, in order."""
    res = 1
    for l in ll:
        res = res * l
    return res

def monomial_ordered(variables,multi_index):
    """
    Produces $b^I = b_1^{I_1}\cdots b_n^{I_n}$.

    This method usually makes sense only for commutative variables.

    Example
    =======
    
    >>> x,y,z = symbols('x y z',commutative=False)
    >>> monomial_ordered([x, y, z],(0, 2, 1))
    y**2*z

    """
    # TODO: all the checkings!
    res = 1
    for j in range(len(variables)):
        res = res * Pow(variables[j],multi_index[j])
        #res *= variables[j]**multi_index[j]
    return res

def noncomm_pol_dict(pol):
    """
    Represent a polynomial *pol* as a dictionary ``pol_dict[monomial] = coefficient``.

    *pol* is considered as a polynomial over a commutative ring (the coefficients) with noncommutative variables.
    
    Example
    =======

    >>> x, y = symbols('x y', commutative = False)
    >>> s = symbols('s', commutative = True)
    >>> p = 2 + 3*x + 4*s*x + 5*x*x + 6*x*y + 7*y*x
    >>> noncomm_pol_dict(p)
    {1: 2, x: 4*s + 3, x**2: 5, x*y: 6, y*x: 7}
    >>> noncomm_pol_dict(x*y)
    {1: 0, x*y: 1}

    """
    poll = expand(pol)
    # poll = 2 + 3*x + 4*s*x + 5*x*x + 6*x*y + 7*y*x
    pol_addends = flatten(poll.as_coeff_add())
    # pol_addends = [2, 3*x, 5*x**2, 4*s*x, 6*x*y, 7*y*x]
    pol_dict = {}
    for pa in pol_addends:
        pm = pa.args_cnc() # in sympy.core.expr
        coeff = mul_list(pm[0])
        mon = mul_list(pm[1])
        pol_dict[mon] = pol_dict.setdefault(mon,0) + coeff
    # pol_dict = {1: 2, x: 4*s + 3, x**2: 5, x*y: 6, y*x: 7}
    return pol_dict

#ANOTHER WAY TO IMPLEMENT IT:
#x,y,z = symbols('x y z', commutative = False)
#basis1 = [x,y+z]
#basis2 = [x,y,z]
#def coeff2(vect,b):
#    vect = expand(vect)
#    if isinstance(vect,Add):
#        return sum([coeff2(v,b) for v in vect.args])
#    coeff_c = 1
#    if isinstance(vect,Mul):
#        vect_yc = vect.args_cnc()[0]
#        vect_nc = vect.args_cnc()[1]
#        coeff_c = Mul(*vect_yc)
#        vect = Mul(*vect_nc)
##         return Mul(*vect_yc) * coeff2(vect.args[-1],b)
#    if vect == b:
#        return coeff_c
#    return 0
#for a in basis1:
#    for b in basis2:
#        print(a,b,coeff2(a,b))
#    print()
    

def reverse_order(monomial): 
    """
    Takes ``x*y`` and outputs ``y*x``. Only for monomials.
    TODO: Extend it to polynomials?
    """
    mon = monomial.as_coeff_mul()[1]
    mon = list(mon)
    mon.reverse()
    return mul_list(mon)

def polynomial_build(coeff_str,variables,degree):
    """
    Builds the general form polynomial of *degree* in *variables*.

    Examples:
    =========
    
    >>> x, y, z = symbols('x y z')
    >>> polynomial_build('f',[x],2)
    f_{(0,)} + f_{(1,)}*x + f_{(2,)}*x**2
    >>> polynomial_build('f',[x,y],2)
    f_{(0, 0)} + f_{(0, 1)}*y + f_{(0, 2)}*y**2 + f_{(1, 0)}*x + f_{(1, 1)}*x*y + f_{(2, 0)}*x**2
    >>> polynomial_build('f',[x-z,y],2)
    f_{(0, 0)} + f_{(0, 1)}*y + f_{(0, 2)}*y**2 + f_{(1, 0)}*(x - z) + f_{(1, 1)}*y*(x - z) + f_{(2, 0)}*(x - z)**2

    """
    num_var = len(variables)
    indices = build_graded_indices_dict([num_var],degree)
    coeff={}
    for d in range(degree+1):
        coeff[d] = [Symbol(coeff_str+'_{'+str(pp)+'}') for pp in indices[d]]
    pol = 0
    for d in range(degree+1):
        pol_d = zip(coeff[d],indices[d])
        pol = pol + sum(pp[0]*monomial_ordered(variables,pp[1]) for pp in pol_d)
    return pol
def maximal_lin_ind(vectors : list):
    """
    Produces a maximal subset of *vectors* of linearly independent vectors.

    *vectors* is a list of arrays.
    The output will be a list of arrays.

    Example
    =======
    
    >>> from random import randrange
    >>> vlist = []
    >>> dim = 10
    >>> how_many = 10
    >>> for i in range(how_many):
    >>>     vlist.append([randrange(100) for j in range(dim)])
    >>> m = Matrix(maximal_lin_ind(vlist))
    >>> m.rank()

    Of course, in this code, the rank will be almost always maximal, by statistical reasons.

    """
    if len(vectors) == 0:
        return []
    dim = max(vectors[0].shape)
    if dim == 0:
        return []
    vect_cp = vectors.copy()
    vect_li = []
    rank = 0
    M = Matrix()
    while True:
        v = vect_cp.pop()
        M = M.col_insert(0,Matrix(v))
        if M.rank() > rank :
            rank += 1
            vect_li.append(v)
        if rank == dim or len( vect_cp ) == 0:
            break
    return vect_li

def from_symbols_to_list(vector, basis: list):
    """
Returns the coefficients of *vector* with respect to *basis*.

Given a symbolic expression *vector* and a list of symbols *basis*,
it returns a list of commutative elements that are the cooeficients of the vector in that basis.

NOTA BENE:
    - elements of the basis should be symbols, not linear combinations of them (which are ignored)
    - if *vector* contains other noncommutative symbols other than *basis*, they are ignored.

    """
    v = expand(vector)
    v_coeff_dict = noncomm_pol_dict(v)
    v_list = [v_coeff_dict.get(b,0) for b in basis]
    return v_list
def combinations(lists: list):
    """
Given a list of lists, returns a list with all choices.

Example:
========

>>> combinations([[1,2],[3,4,5],[6,7]])
[[1, 3, 6],
 [1, 3, 7],
 [1, 4, 6],
 [1, 4, 7],
 [1, 5, 6],
 [1, 5, 7],
 [2, 3, 6],
 [2, 3, 7],
 [2, 4, 6],
 [2, 4, 7],
 [2, 5, 6],
 [2, 5, 7]]


    """
    res = [[]]
    for l in lists:
        res_new = []
        for ll in res:
            for lll in l:
                res_new.append(ll + [lll])
        res = res_new
    return res

def my_contraction(tensr,idxs: list):
    """
If tensr.shape is (1,2,2,1),
then my_shape(tesr) is (covect, vect)
and my_contraction(tensr,(0,1)) contracts the covector and the vector,
returing what should be a scalar, that is, a tensor of shape (1,1).
    """
    indices = []
    for i,j in idxs:
        indices.append( (2*i,2*j+1) )
        indices.append( (2*i+1,2*j) )
    return tensorcontraction(tensr,*indices)

