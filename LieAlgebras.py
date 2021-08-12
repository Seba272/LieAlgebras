from sympy import *
import time as time

#TODO:
#    - class Tensor_algebra with:
#        - abstract basis, dual basis, contractions.
#    - class jet with

# For docstring, I use this formatting:
# https://docs.sympy.org/latest/documentation-style-guide.html

# SOME MATERIAL FOR NON-COMMUTATIVE ALGEBRAS :
# This will be used for tensors and vector fields.
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

def build_graded_indeces_dict(growth_vector, depth):
    """
    Builds a dictionary of indeces.

    Builds a dictionary ``I`` so that ``I[k]`` is the list of all tuples ``(i_1,...,i_n)`` (where ``n`` is the dimension, i.e., ``sum(growth_vector)``) whose weight is ``k``.
    ``k`` runs from 1 to *depth*.
    
    NB! ``I[k]`` is ordered how the function ``set`` orders it.

    Examples
    ========

    >>> build_graded_indeces_dict([2,1], 3)
    {1: [(1, 0, 0), (0, 1, 0)],
     2: [(0, 2, 0), (1, 1, 0), (0, 0, 1), (2, 0, 0)],
     3: [(1, 0, 1), (1, 2, 0), (2, 1, 0), (3, 0, 0), (0, 3, 0), (0, 1, 1)]}
    >>> build_graded_indeces_dict([2],3)
    {1: [(1, 0), (0, 1)],
 2: [(1, 1), (2, 0), (0, 2)],
 3: [(0, 3), (1, 2), (2, 1), (3, 0)]}

    """
    dimension = sum(growth_vector)
    step = len(growth_vector)
    weights = build_weight_list(growth_vector)
    graded_indeces_dict = {}
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
            for idx in graded_indeces_dict[j-weights[k]]:
                idx_n = list(idx)
                idx_n[k]+=1
                new_Ij.append(tuple(idx_n))
        # The previous loop produces duplicates, which are eliminated by ``list(set())``.
        graded_indeces_dict[j] = list(set(new_Ij))
    return graded_indeces_dict

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
        res *= variables[j]**multi_index[j]
    return res

def noncomm_pol_dict(pol):
    """
    Represent a polynomial *pol* as a dictionary ``pol_dict[monomial] = coefficiet``.

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

def reverse_order(monomial): 
    """
    Takes ``x*y`` and outputs ``y*x``. Only for monomials.
    TODO: Extend it to polynomials?
    """
    mon = monomial.as_coeff_mul()[1]
    mon = list(mon)
    mon.reverse()
    return mul_list(mon)

def maximal_lin_ind(vectors):
    """
    Produces a maximal subset of *vectors* of linearly independent vectors.

    *vectors* can be any iterable of vectors.
    The output will be a list of lists.

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
    card = len(vectors)
    if card == 0:
        return []
    dim = len(vectors[0])
    if dim == 0:
        return []
    max_iterations = max(card, dim)
    #vect_cp = vectors.copy()
    vect_cp = [list(v) for v in vectors]
    vect_li = []
    rank = 0
    while 1:
        v = vect_cp.pop()
        m = Matrix(vect_li + [v])
        if m.rank() > rank :
            rank += 1
            vect_li += [v]
        if rank == dim or vect_cp == []:
            break
    return vect_li











class LieAlgebra:
    name = 'LieAlgebra'
    _dimension = None
    _basis_symbols = None
    _std_basis = None
    _an_outside_basis = None
    _use_outside_brackets = False
    _structure_constants = None
    _rules_structure_constants = None
    _rules_basis_symbols = None
    _rules_vector_fields = None
    _dual_basis_symbols = None
    is_nilpotent = None
    _step = None
    is_graded = None
    is_stratified = None
    _growth_vector = None
    _graded_basis_symbols = None
    _basis_HD = {}
    _a_basis_of_brackets_ = None
    _a_basis_of_brackets_graded_ = None
    _a_basis_of_brackets_matrix_ = None
#    def __init__(self,dim,struct_const='Gamma'):
#        self._dimension = dim
#        self.structure_constants_build(struct_const)

    def __call__(self,v,w):
        return self.brackets(v,w)

    def brackets(self,v,w):
        if self._use_outside_brackets:
            return self.brackets_outside(v,w)
        typev = type(v)
        if typev == list or typev == Array:
            return self.brackets_strct_consts(v,w)
        if typev == Matrix:
            return v*w - w*v
        print('Error 61150e34 : There is no brackets that match it.')
        return None
    
    def dimension(self):
        """
        Returns the dimension of self.
        If it is not set yet, it asks for it.
        """
        if self._dimension == None:
            ans = input('Dimension of self not declared yet: do you want to set it now? (Y/n)')
            if ans[0] == 'y' or ans[0] == 'Y':
                dim = int(input('Dimension = '))
                self.dimension_set(dim)
        return self._dimension

    def dimension_set(self,dim):
        self._dimension = dim

    def dimension_get(self):
        return self._dimension

    def basis_symbols(self):
        if self._basis_symbols == None:
            self.basis_symbols_build()
        return self._basis_symbols
    
    def basis_symbols_build(self,smbl='b'):
        if self.is_graded :
            if self._graded_basis_symbols == None:
                self.graded_basis_symbols_build(smbl)
            self._basis_symbols = flatten(self.graded_basis_symbols())
        else:
            dim = self.dimension()
            self._basis_symbols = [symbols(smbl+'_'+str(j) for j in range(dim))]

    def basis_symbols_set(self,basis):
        self._basis_symbols = basis

    def std_basis(self):
        """
        Returns the standard basis of self.

        Returns a list [e1,...,en] where n is the dimension of the domain and ej is a list of zeros with a 1 at the j-th place (counting from 1).

        """
        if self._std_basis == None:
            self.std_basis_build()
        return self._std_basis

    def std_basis_build(self):
        dim = self.dimension()
        std_basis = eye(dim).columnspace()
        std_basis = [ list(ec) for ec in std_basis ]
        self._std_basis = std_basis

    def an_outside_basis(self):
        """
        Returns a basis chosen by the user.

        In some cases, the standard basis corresponds to objects defined by the user, for instance matrices.
        
        """
        if self._an_outside_basis == None:
            print('There is no objective basis.')
        return self._an_outside_basis

    def an_outside_basis_set(self,basis):
        self._an_outside_basis = basis

    def brackets_outside(self,v,w):
        """
        Sometimes, we have brackets defined by the user on the elements from the outside basis.
    
        """
        pass

    def use_outside_brackets(self,doit=True):
        """
        Sets to use by default the brackets defined by the user.

        Example:
        ========
        >>> self.use_outside_brackets()
        We use the the outside brackets.
        >>> self.use_outside_brackets(False)
        We now go back to normal.

        (In fact, there is no output or printing).

        """
        self._use_outside_brackets = doit

    def structure_constants(self):
        """
        Returns the structur constants of self.

        If they are not here yet, we build them.
        """
        if self._structure_constants == None:
            self.structure_constants_build()
        return self._structure_constants

    def structure_constants_build(self,rules=None):
        """
        Build structure constants of self using a dictionary *rules*.

        *rules* is a dictionary with entries (index1,index2):[(coeff, index_vect),...]
        {(i,j):[(a,2),(b,3)]}
        means that bracket(bi,bj) = a*b2 + b*b3
        Always i<j !!

        Examples
        ========
        
        Heisenberg Lie algebra (dim = 3): heis = {(0,1) : [(1,2)]}
        so(2) (dim = 3) : so2 = { (0,1):[(1,2)] , (1,2):[(2,1)] , (0,2):[(-2,0)] }
        so(2)xR (dim = 4) : so2R = so2 
        A_{4,3} (dim = 4) : a43 = { (1,3) : [(-1,1)] , (2,3):[(-1,0)] }

        """
        if rules == None:
            self._structure_constants_build_abstract(rules)
        elif type(rules) == str:
            self._structure_constants_build_examples(rules)
        elif type(rules) == dict:
            self._structure_constants_build_from_rules(rules)
        else:
            print('Error from structure_constants_build: *rules* must be ``None``, a string or a dictionary. Instead, it was of type ', type(rules))

    def _structure_constants_build_examples(self,name):
        pass
        """
        if name == 'heis':
            self._structure_constants_build_from_rules({(0,1) : [(1,2)]})
        if name == 'so2':
            self._structure_constants_build_from_rules({(0,1) : [(1,2)]})
        if name == 'so2':
            self._structure_constants_build_from_rules({(0,1) : [(1,2)]})
        if name == 'so2':
            self._structure_constants_build_from_rules({(0,1) : [(1,2)]})
        """

    def _structure_constants_build_abstract(self,struct_const):
        """
        Output: tensor of type (2,1) : G[_a,_b,^c]
        brackets(B_a,B_b) = G[_a,_b,^c] B_c    
        """
        dim = self._dimension
        G = self._structure_constants_symbol = IndexedBase(struct_const, real=True)
        self._structure_constants = MutableDenseNDimArray([G[a,b,c] for a in range(dim) for b in range(dim) for c in range(dim)],(dim,dim,dim))
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    if a==b:
                        self._structure_constants[a,b,c] = 0
                    if a<b:
                        self._structure_constants[a,b,c] = -self._structure_constants[b,a,c]

    def _structure_constants_build_from_rules(self,rules):
        dim = self._dimension
        # put constants to zero:
        self._structure_constants = MutableDenseNDimArray.zeros(dim,dim,dim)
        # update the structure constants that have to
        for r in rules:
            res = rules[r]
            for a in res:
                self._structure_constants[r[0],r[1],a[1]] = a[0]
                self._structure_constants[r[1],r[0],a[1]] = -a[0]

    def brackets_strct_consts(self,v,w):
        """
        brackets(v,w)[^k] = Gamma[_i,_j,^k]v[^i]w[^j] using structure constants.
        """
        structure_constants = self.structure_constants()
        return tensorcontraction(tensorproduct(structure_constants,v,w),(0,3),(1,4))

    def check_jacobi(self):
        """
        G[_a,_d,^e] G[_b,_c,^d] + G[_b,_d,^e] G[_c,_a,_d] + G[_c,_d,^e] G[_a,_b,^d]
        but it is easier to check it on symbolic vectors.
        """
        dim = self._dimension
        xx = IndexedBase('x', real=True)
        yy = IndexedBase('y', real=True)
        zz = IndexedBase('z', real=True)
        x = MutableDenseNDimArray([xx[j] for j in range(dim)],(dim))
        y = MutableDenseNDimArray([yy[j] for j in range(dim)],(dim))
        z = MutableDenseNDimArray([zz[j] for j in range(dim)],(dim))
        res = simplify(self.brackets(self.brackets(x,y),z) + self.brackets(self.brackets(y,z),x) + self.brackets(self.brackets(z,x),y))
        try:
            res = res.applyfunc(simplify)
        except:
            None
        return res
    
    def from_symbols_to_array(self,v):
        basis = self.basis_symbols()
        v = expand(v)
        v_coeff_dict = noncomm_pol_dict(v)
        v_array = Array([v_coeff_dict[b] for b in basis])
        return v_array

    def from_array_to_symbols(self,v):
        basis = self.basis_symbols()
        v_basis = list(zip(list(v),basis)) # [(v1,b1),(v2,b2),...]
        v_symbol = sum([prod(vb) for vb in v_basis]) # v1*b1 + v2*b2 + ...
        return v_symbol

    def brackets_symbols(self,v,w):
        """
        ....
        """
        basis = self.basis_symbols()
        dim = self.dimension()
        v = expand(v)
        v_array = self.from_symbols_to_array(v)
        w = expand(w)
        w_array = self.from_symbols_to_array(w)
        vw_array = self.brackets(v_array,w_array)
        vw = self.from_array_to_symbols(vw_array)
        return vw

    def rules_basis_symbols(self):
        if _rules_basis_symbols == None:
            rules_basis_symbols_build()
        return _rules_basis_symbols

    def rules_basis_symbols_set(self,rules):
        self._rules_basis_symbols = rules

    def rules_basis_symbols_build(self):
        pass

    def rules_vector_fields(self):
        if self._rules_vector_fields == None:
            self.rules_vector_fields_build()
        return self._rules_vector_fields

    def rules_vector_fields_build(self):
        """
        Example in the Heisenberg group:
        rules[y*x] = x*y - z
        i.e.,
        rules = { y*x : x*y - z }
        """
        basis = self.basis_symbols()
        dim = self.dimension()
        self._rules_vector_fields = {}
        for i in range(dim):
            for j in range(i):
                bi = Array(flatten(eye(dim)[i,:]))
                bj = Array(flatten(eye(dim)[j,:]))
                bk = list(self.brackets(bi,bj))
                res = basis[j]*basis[i]
                for k in range(dim):
                    res += bk[k]*basis[k]
                self._rules_vector_fields[basis[i]*basis[j]] = res

    def a_vector(self,v):
        """
        Returns a vector of self.

        *v* is a string that is used to define symbols v1,...,vn for the components of the output.
        The output is an Array.
        """
        return Array(symbols(v+'[:%d]'%self._dimension))

    def _multbra_u(self,r,v,s,w,u):
        """
        Compute
        \[
        [v,[v,...,[v,[w,[w,...,[w,u]]]]]]]
        \]
        con r volte v e s volte w
        """
        uu = u
        for i in range(s):
            uu = self.brackets(w,uu)
        for j in range(r):
            uu = self.brackets(v,uu)
        return uu

    def _multbra(self,r,v,s,w):
        """
        Compute
        \[
        [v,[v,...,[v,[w,...[w,[v,...[v,w...
        \]
        r[1] volte v, s[1] volte w, r[2] volte v, s[2] volte w, .... , r[n] volte v, s[n] volte w.
        Attenzione che gli ultimi sono delicati, perché [w,w]=0, etc...
        """
        n=len(r)
        if n == 0:
            return 0*w
        if s[-1]>1:
            return 0*w
        if s[-1] == 0 and r[-1]>1:
            return 0*v
        if s[-1] == 0 and r[-1] == 0:
            return self._multbra(r[:-1],v,s[:-1],w) # drop the last element and start again
        if s[-1]==1:
            u = w
            u = self._multbra_u(r[-1],v,0,w,u)
        if s[-1]==0:
            u = v
        for i in range(n-1):
            u = self._multbra_u(r[i],v,s[i],w,u)
        return u

    def _list_rs(self,n,N):
        """
        Output: list of 
        [r1,s1,r2,s2,....,rn,sn]
        with rj+sj>0 and r1+s1+...+rn+sn <= N
        """
        pairs = []
        for r in range(N+1):
            for s in range(N+1-r):
                pairs.append([r,s])
        pairs.pop(0) # the first element is [0,0]
        if n == 0 or n>N:
            return []
        RS = pairs
        for j in range(1,n):
            RS_prec = RS
            RS = []
            for rs in RS_prec:
                for pair in pairs:
                    if sum(rs + pair) <= N :
                        RS.append(rs+pair)
        return RS

    def bch_trnc(self,v,w,N):
        """\
        Baker–Campbell–Hausdorff formula
        https://en.wikipedia.org/wiki/Baker–Campbell–Hausdorff_formula
        with sum up to N
        """
        res = 0*v
        for n in range(1,N+1):
            coef = (-1)**(n-1)/n
            RS = self._list_rs(n,N)
            RS_sum = 0*v
            for rs in RS:
                somma = sum(rs)
                rs_fact = [factorial(x) for x in rs]
                prodotto = mul_list(rs_fact)
                r = [rs[2*x] for x in range(n)]
                s = [rs[2*x+1] for x in range(n)]
                RS_sum = RS_sum + (somma*prodotto)**(-1) * self._multbra(r,v,s,w)
            res = res + coef * RS_sum
        return res

    def bch(self,v,w):
        """
        Baker–Campbell–Hausdorff formula
        https://en.wikipedia.org/wiki/Baker–Campbell–Hausdorff_formula
        with sum up to the step of self,
        which needs to be nilpotent.
        Otherwise, use bch_trnc(v,w,N) with level of precision N.
        """
        step = self.step()
        if type(step) == int :
            return self.bch_trnc(v,w,step)
        else:
            print('Error from bch(): This algebra is not nilpotent or self.step is not set. Use bch_trnc(v,w,N) with level of precision N. Use self.declare_nilpotent(step) if you think self is nilpotent.')
            return None

    def declare_nilpotent(self,step=None,isit=True):
        self.is_nilpotent = isit
        self.step_set(step)

    def check_nilpotent(self):
        print('Is nilpotent? ',self.is_nilpotent,' step: ',self.step())
        return None

    def declare_graded(self,growth_vector=None,step=None,isit=True):
        if step == None and growth_vector != None:
            step = len(growth_vector)
        self.declare_nilpotent(step,isit)
        self.is_graded = isit
        self.growth_vector_set(growth_vector)
        if growth_vector != None:
            self.dimension_set(sum(growth_vector))

    def declare_stratified(self,growth_vector=None,step=None,isit=True):
        self.declare_graded(growth_vector,step,isit)
        self.is_stratified = isit
    
    def step(self):
        if self._step == None:
            self.step_set()
        return self._step

    def step_set(self,step=None):
        if self._step != None:
            print("At the moment, the step is ",self.step())
        if step==None:
            self._step = int(input("What is the step? "))
        else:
            self._step = step
        #print("step set to ",self.step," (You should check that the Lie algebra is nilpotent!)")

    def growth_vector(self):
        return self._growth_vector

    def growth_vector_set(self,gr_vect):
        self._growth_vector = gr_vect

    def graded_basis_symbols(self):
        if self._graded_basis_symbols == None:
            self.graded_basis_symbols_build()
        return self._graded_basis_symbols

    def graded_basis_symbols_set(self,basis):
        self._graded_basis_symbols = basis

    def graded_basis_symbols_build(self,smbl='b'):
        """
        Per Heisenberg:
        x,y,z = symbols('x y z',commutative=False)
        self.graded_basis_symbols = [[x,y],[z]]
        self.basis_symbols = flatten(self.graded_basis_symbols)
        xd,yd,zd = symbols('x^+ y^+ z^+',commutative=False) # dual elements
        self.dual_basis_symbols = [xd,yd,zd]
        """
        if self.is_graded != True :
            print('Error from graded_basis_symbols_build : The algebra must be graded')
            return None
        self._graded_basis_symbols = []
        growth_vector = self.growth_vector()
        for j in range(len(growth_vector)):
            base_j_layer = []
            for k in range(growth_vector[j]):
                base_j_layer.append(symbols(smbl+'^'+str(j)+'_'+str(k),commutative=False))
            self._graded_basis_symbols.append(base_j_layer)
        self._basis_symbols = flatten(self.graded_basis_symbols)

    def dual_basis_symbols(self):
        if self._dual_basis_symbols == None:
            self.dual_basis_symbols_build()
        return self._dual_basis_symbols

    def dual_basis_symbols_build(self,pre_symbol_for_dual='@',post_symbol_for_dual=''):
        basis = self.basis_symbols()
        self._dual_basis_symbols = [ \
                symbols(pre_symbol_for_dual + vect.name + post_symbol_for_dual,commutative=False) \
                for vect in basis ]

    def basis_HD(self,order):
        if self._basis_HD.get(order,None) == None :
            self.basis_HD_build(order)
        return self._basis_HD[order]

    def basis_HD_build(self,order):
        """
        Computes a basis $(A_I)_{I\in \scr I^a}$ of $HD^a(\mathfrak g;\R)$, for $a$ equal to *order*.

        Example 1:
        ==========

        >>> from LieAlgebras import *
        >>> heis = LieAlgebra(3)
        >>> heis.dimension = 3
        >>> heis.structure_constants_build({(0,1) : [(1,2)]})
        >>> heis.declare_stratified(True,2,[2,1])
        >>> x,y,z = symbols('x y z',commutative=False)
        >>> heis.graded_basis_symbols = [[x,y],[z]]
        >>> heis.basis_symbols = flatten(heis.graded_basis_symbols)
        >>> xd,yd,zd = symbols('x^+ y^+ z^+',commutative=False) # dual elements
        >>> heis.dual_basis_symbols = [xd,yd,zd]
        >>> heis.build_rules_vector_fields()
        >>> heis.basis_HD(1)
        {(1, 0, 0): x^+, (0, 1, 0): y^+}
        >>> heis.basis_HD(2)
        {(0, 2, 0): y^+**2,
         (1, 1, 0): x^+*y^+ + y^+*x^+,
         (0, 0, 1): -x^+*y^+,
         (2, 0, 0): x^+**2} 
        >>> heis.basis_HD(3)
        {(1, 0, 1): -x^+*y^+*x^+ - 2*x^+**2*y^+,
         (1, 2, 0): x^+*y^+**2 + y^+*x^+*y^+ + y^+**2*x^+,
         (2, 1, 0): x^+*y^+*x^+ + x^+**2*y^+ + y^+*x^+**2,
         (3, 0, 0): x^+**3,
         (0, 3, 0): y^+**3,
         (0, 1, 1): -2*x^+*y^+**2 - y^+*x^+*y^+}

        Example 2:
        ==========

        >>> from LieAlgebras import *
        >>> heis = LieAlgebra(3)
        >>> heis.dimension = 3
        >>> heis.structure_constants_build({(0,1) : [(1,2)]})
        >>> heis.declare_stratified(True,2,[2,1])
        >>> heis.build_graded_basis_symbols()
        >>> heis.build_dual_basis_symbols('@','#')
        >>> heis.build_rules_vector_fields()
        >>> heis.basis_HD(2)
        {(0, 2, 0): @b^0_1#**2,
         (1, 1, 0): @b^0_0#*@b^0_1# + @b^0_1#*@b^0_0#,
         (0, 0, 1): -@b^0_0#*@b^0_1#,
         (2, 0, 0): @b^0_0#**2}

        """
        basis = self.basis_symbols()
        dual_basis = self.dual_basis_symbols()
        basis_V1 = self.graded_basis_symbols()[0]
        rules = self.rules_vector_fields()
        basis_T_a_V1 = build_monomials_nc(basis_V1,order)[order]
        indeces_a_hom = build_graded_indeces_dict(self.growth_vector(),order)[order]
        basis_HD_a = {}
        for idx in indeces_a_hom:
            A = 0
            mon = monomial_ordered(basis,idx)
            for xi in basis_T_a_V1:
                tau_bb = noncomm_pol_dict(isubs_force(xi,rules))
                tau_bb_I = tau_bb.get(mon,0)
                A += tau_bb_I * dualize(reverse_order(xi),basis,dual_basis)
            basis_HD_a[idx] = A
        self._basis_HD[order] = basis_HD_a

    def a_basis_of_brackets_graded(self):
        """
        Returns three data for extending Lie algebra morphisms from the first layer to the whole Lie algebra.

        Returns a basis in coordinates for *self* out of the set of all brackets of vectors of the first strata.
        And a dictionary to see for each vector who are his parents.
        And a matrix whose j-th colomun are the coeffiecients of ej in the basis given above.

        Works only for stratified Lie algebras!
        """
        if not self.is_stratified:
            print('Error 61137d91: self must be stratified. Method cannot work.')
        if not self._a_basis_of_brackets_graded_:
            self._a_basis_of_brackets_build()
        if not self._a_basis_of_brackets_matrix_:
            self._a_basis_of_brackets_matrix_build()
        return self._a_basis_of_brackets_graded_ , self._a_basis_of_brackets_dict_ , self._a_basis_of_brackets_matrix_

#    def _a_basis_of_brackets_(self):
#        """
#        Returns a basis in coordinates for *self* out of the set of all brackets of vectors of the first strata.
#        And a dictionary to see for each vector who are his parents.
#
#        Works only for stratified Lie algebras.
#        """
#        if not self.is_stratified:
#            print('Error from _a_basis_of_brackets_: self must be stratified. Method cannot work.')
#        if self._a_basis_of_brackets = None:
#            _a_basis_of_brackets_build()
#        return self._a_basis_of_brackets, self._a_basis_of_brackets_dict
    
    def _a_basis_of_brackets_build(self):
        """
        Builds ``self._a_basis_of_brackets``, that is, a basis in coordinates for *self* out of the set of all brackets.

        Works only for stratified Lie algebras.
        """
        if not self.is_stratified:
            print('Error from _a_basis_of_brackets_build_: self must be stratified. Method cannot work.')
            return 
        dimension = self.dimension()
        dimension_1 = self.growth_vector()[0]
        basis_1 = eye(dimension)[:,dimension_1].columnspace()
        basis = [basis_1]
        prev_len = 0
        new_len = len(flatten(basis))
        basis_decoupling = {}
        while prev_len < new_len:
            prev_len = new_len
            vects_trpl = [[v,w,self.brackets(v,w)] for v in basis_1 for w in basis[-1]]
            vects = [VV[2] for VV in vects_trpl]
            basis.append(maximal_lin_ind(vects.reverse())) # Since this method starts from the last element, for the subsequent lines it is better to reverse vects
            for bb in basis[-1]:
                for VV in vects_trpl:
                    if bb == VV[2]:
                        basis_decoupling[bb] = VV[:2]
                        continue
            new_len = len(flatten(basis))
        self._a_basis_of_brackets_graded_ = basis
        self._a_basis_of_brackets_ = flatten(basis)
        self._a_basis_of_brackets_dict_ = basis_decoupling

    def _a_basis_of_brackets_matrix(self):
        """
        Returns a matrix M whose colomuns are the coordinates of the standard basis in the new basis.
        """
        if self._a_basis_of_brackets_matrix_ == None:
            self._a_basis_of_brackets_matrix_build()
        return self._a_basis_of_brackets_matrix_

    def _a_basis_of_brackets_matrix_build(self):
        basis = self._a_basis_of_brackets_graded()[0]
        self._a_basis_of_brackets_matrix_ = Matrix(basis).transpose().inv()


class LieAlgebra_Morphism:
    _lie_algebra_domain = None
    _std_basis_domain = None
    _lie_algebra_range = None
    _map_dict = None
    _map_list = None # a list; entry j is L(e_j)

    def __call__(self,x):
        """
        Apply the morphism self to x.

        *x* is suppose to be a list. Anyway, ``list(x)`` is performed.
        The output is also a list.

        """
        x = list(x)
        L = self.map_list()
        y_list = zip(L,x)
        y_prod = [ yy[0]*yy[1] for yy in y_list ]
        y = sum(y_prod)
        return y

    def std_basis_domain(self):
        """
        Returns the standard basis of the domain.

        Returns a list [e1,...,en] where n is the dimension of the domain and ej is a list of zeros with a 1 at the j-th place (counting from 1).

        """
        if self._std_basis_domain == None:
            self.std_basis_domain_build()
        return self._std_basis_domain

    def std_basis_domain_build():
        dim = self.lie_algebra_domain().dimension()
        std_basis = eye(dim).columnspace()
        std_basis = [ list(ec) for ec in std_basis ]
        self._std_basis_domain = std_basis 

    def map_list(self):
        if self._map_list == None:
            print('Map_list not defined yet!')
        return self._map_list

    def map_list_build(self):
        mdict = self.map_dict()
        dimension = self.lie_algebra_domain().dimension()
        std_basis = self.std_basis_domain()
        mlist = [ mdict.get(ec,None) for ec in std_basis ]
        if None in set(mlist):
            print('Error 61136c45: map_dict does not contain infos for all elements of the standard basis!'  )
        return mlist

    def map_dict(self):
        if self._map_dict == None:
            print('Error 611374fd: the map is not defined. Use self.map_dict_set() or self.extend_from_V1().')
        return self._map_dict

    def map_dict_set(self,md):
        self._map_dict = md

    def lie_algebra_domain(self):
        return self._lie_algebra_domain

    def lie_algebra_domain_set(self,lie_alg):
        self._lie_algebra_domain = lie_alg
    
    def lie_algebra_range(self):
        return self._lie_algebra_range

    def lie_algebra_range_set(self,lie_alg):
        self._lie_algebra_range = lie_alg
    
    def extend_from_V1(self,diz_V1):
        la_dom = self.lie_algebra_domain()
        la_ran = self.lie_algebra_range()
        if not la_dom.is_stratified:
            print('Error 61136ccf: The Lie algebra of the domain must be stratified. Method breaks down.')
            return
        dim = la_dom.dimension()
        dim1 = la_dom.growth_vector()[0]
        basis_std = std_basis_domain()
        basis1 = basis_std[:dim1]
        bas_bra , bas_bra_dict , bas_bra_matr = la_dom._a_basis_of_brackets_graded()
        # See https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        # The second dictionary prevails. Thus, this function can change the map.
        m_dict = self.map_dict() | diz_V1 
        # bas_bra is a list
        # basj is a list, a basis of a layer j
        # b is a vector or list of dim numbers
        # bas_bra_dict[b] is a pair of vectors v,w, one in layer 1 and the other in layer j-1, so that brakets(v,w) = b
        for basj in bas_bra:
            for b in basj:
                v , w = bas_bra_dict[b]
                m_dict[b] = la_ran(m_dict[v],m_dict[w]) # la_ran.__call__ is the lie brackets in the taget lie-algebra.
        # Now m_dict is a large dictionary containing all informations we have at the moment about the map.
        # We want not to say what is the map applied to the standard basis of lie_dom
        bas_bra_flat = flatten(bas_bra)
        for j in range(dimension_1,dimension):
            ej = basis_std[j]
            mj = bas_bra_matr.col(j)
            c_b = zip(mj,bas_bra_flat)
            y = [ c*m_dict[b] for [c,b] in c_b ]
            y = sum(y)
            m_dict[ej] = y
        self.map_dict_set(m_dict)

    def check(self):
        """
        Checks that everything is alright.

        Checks that:
        * The map is defined on the whole domain (i.e., on the standard basis)
        * The map is a Lie algebra morphism (i.e., the graph is closed under Lie brackets)

        """
        map_list = self.map_list() # Check is inside the method.
        la_dom = self.lie_algebra_domain()
        la_ran = self.lie_algebra_range()
        basis_std = std_basis_domain()
        dim = la_dom.dimension()
        pairs = [ (basis_std[i],basis_std[j]) : for i in range(j) for j in range(dim)]
        for (b1,b2) in pairs:
            test = simplify( self(la_dom(b1,b2)) - la_range(self(b1),self(b2)) )
            if test != 0:
                print('The map is not a morphism: ')
                print('b1 = ',b1)
                print('b2 = ',b2)
                print('self(la_dom(b1,b2)) = ', simplify(self(la_dom(b1,b2))) )
                print('la_range(self(b1),self(b2)) = ', simplify(la_range(self(b1),self(b2))) )
                break
        if map_list == None or test !=0:
            return False
        else:
            return True







            
        









class RiemLieAlgebra(LieAlgebra):
    """\
Subclass of LieAlgebra: in addition to Lie algebra's methods and things, there is a also a scalar product and the corresponding left-invariant Riemannian structure.
    """
    def __init__(self,dim,struct_const='Gamma',scalar_prod='g'):
        self._dimension = dim
        self.structure_constants_build(struct_const)
        self.scalar_product_build(scalar_prod)
    def scalar_product_build(self,m=None):
        """\
ma is a dim x dim square matrix that is used to define the scalar product in A.
        """
        if m == None or type(m) == str :
            self._scalar_product_build_abstract(m)
        else:
            self._scalar_product_build_from_matrix(m)
    def _scalar_product_build_abstract(self,scalar_prod):
        dim = self._dimension
        G = self._scalar_product_symbol = IndexedBase(scalar_prod, real=True)
        self._scalar_product_tensor = MutableDenseNDimArray([G[a,b] for a in range(dim) for b in range(dim)],(dim,dim))
        for a in range(dim):
            for b in range(dim):
                if b<a:
                    self._scalar_product_tensor[a,b] = self._scalar_product_tensor[b,a]
    def _scalar_product_build_from_matrix(self,m):
        dim = self._dimension
        if not m.is_Matrix or not m.shape == (dim,dim):
            print("Error: must be a square matrix of rank", dim)
            return None
        self._scalar_product_tensor = MutableDenseNDimArray(m)
    def scalar_product_tensor(self):
        """\
Output: tensor of type (2,0): M[_a,_b] = <B_a,B_b>
        """
        try:
            self._scalar_product_tensor
        except AttributeError:
            self.scalar_product_build()
        return self._scalar_product_tensor
    def scalar_product(self,v,w):
        """\
v and w of class Array
Compute scalar product of v and w in Lie algebra A.
<v,w> = v[^i] M[_i,_j] w[^j]
        """
        M = self.scalar_product_tensor()
        return tensorcontraction(tensorproduct(v,M,w),(0,1),(2,3))
    def scalar_product_inverse_build(self):
        sc_pr_inv = self.scalar_product_tensor().tomatrix().inv()
        self._scalar_product_inverse_tensor = MutableDenseNDimArray(sc_pr_inv)
    def scalar_product_inverse_tensor(self):
        """\
Output: tensor of type (0,2) : M[^a,^b] = (M.inv)[a,b]
        """
        try:
            self._scalar_product_inverse_tensor
        except AttributeError:
            self.scalar_product_inverse_build()
        return self._scalar_product_inverse_tensor
    def connection_tensor(self):
        """\
Output: tensor of type (3,0) : D[_a,_b,_c] = < D_{B_a} B_b, B_c >
        """
        try:
            self._connection_tensor
        except AttributeError:
            self.connection_tensor_build()
        return self._connection_tensor
    def connection_tensor_build(self):
        """\
< D_x y,z > =
1/2 x[a]y[b]z[c] ( G[a,b,k] M[k,c] - M[a,k] G[b,c,k] + G[l,a,k] M[k,b] delta[l,c] )   
= D[a,b,c] x[a]y[b]z[c]
        """
        G = self._structure_constants
        M = self._scalar_product_tensor
        delta = eye(self._dimension)
        u = tensorcontraction(tensorproduct(G,M),(2,3))
        u -= tensorcontraction(tensorproduct(M,G),(1,4))
        u += tensorcontraction(tensorproduct(G,M,delta),(2,3),(0,5))
        self._connection_tensor = u/2
    def connection_scalar(self,x,y,z):
        """\
Returns a scalar that is (The Koszul formula):
\[
\\langle \\nabla_x y , z \\rangle = \\frac{1}{2} \\left( 
    \\langle [x,y],z \\rangle + \\langle [z,y],x \\rangle + \\langle [z,x],y \\rangle
    \\right)
\]
(Nabla_xy)[k] = Nabla[i,j,k]x[i]y[j]

< D_x y,z > =
1/2 x[a]y[b]z[c] ( G[a,b,k] M[k,c] - M[a,k] G[b,c,k] + G[l,a,k] M[k,b] delta[l,c] ) 
=  D[a,b,c] x[a]y[b]z[c]
        """
        D = self.connection_tensor()
        return tensorcontraction(tensorproduct(D,x,y,z),(0,3),(1,4),(2,5))
    def connection_map_tensor(self):
        """\
Output: tensor of type (2,1) : D[_a,_b,^c] = D[_a,_b,_k] M[^k,^c]
(using the inverse of the scalar product)
        """
        try:
            self._connection_map_tensor
        except AttributeError:
            self.connection_map_tensor_build()
        return self._connection_map_tensor
    def connection_map_tensor_build(self):
        """
D[_a,_b,^c] = D[_a,_b,_k] M[^k,^c] =  D[a,b,k] Minv(k,c)      
        """
        D = self.connection_tensor()
        Minv = self.scalar_product_inverse_tensor()
        self._connection_map_tensor = tensorcontraction(tensorproduct(D,Minv),(2,3))
    def connection_map(self,x,y):
        """\
(D_xy)[^k] = D[_a,_b,^k] x[a]y[b]
        """
        D = self.connection_map_tensor()
        return tensorcontraction(tensorproduct(D,x,y),(0,3),(1,4))
    def curvature_map(self,x,y,z):
        """\
Returns a vector.
Curvature tensor computed as
\\[
R_{x,y}z 
\\nabla_{[x,y]} z - \\nabla_x \\nabla_y z + \\nabla_y \\nabla_x z 
\\]
The sign convention is the same as in O'Neill, Milnor;
but opposit of Le Donne.
        """
        vect1 = self.connection_map(self.brackets(x,y),z)
        vect2 = self.connection_map(x,self.connection_map(y,z))
        vect3 = self.connection_map(y,self.connection_map(x,z))
        return vect1 - vect2 + vect3
    def curvature_tensor_evaluated(self,x,y,z,u):
        """\
Returns a scalar.
\\[
R(x,y,z,u) = \\langle R_{x,y}z , u \\rangle .
\\]
        """
        return self.scalar_product( self.curvature_map(x,y,z) , u )
    def curvature_tensor(self):
        """\
Returns the Riemannian curvature tensor (4,0):
R[_a,_b,_c,_d]
= G[_a,_b,^k] D[_k,_c,^j] M[_j,_d] \\
    - D[_a,_k,^j] D[_b,_c,^k] M[_j,_d] \\ 
    + delta[_a,^l] D[_b,_k,^j] D[_l,_c,^k] M[_j,_d] 
= G[_a,_b,^k] D[_k,_c,_d] \\
    - D[_a,_k,_l] D[_b,_c,^k] delta[^l,_d] \\ 
    + delta[_a,^l] D[_b,_k,_j] D[_l,_c,^k] delta[^j,_d] .
        """
        try:
            self._curvature_tensor
        except AttributeError:
            self.curvature_tensor_build()
        return self._curvature_tensor
    def curvature_tensor_build(self):
        """\
<R(x,y,z),t>
= x[a]y[b]z[c]t[d] ( G[a,b,k] D[k,c,j] M[j,d] - D[a,k,j] D[b,c,k] M[j,d] + delta[a,l] D[b,k,j] D[l,c,k] M[j,d]  )
Returns the Riemannian curvature tensor (4,0):
R[_a,_b,_c,_d]
= G[_a,_b,^k] D[_k,_c,^j] M[_j,_d] \\
    - D[_a,_k,^j] D[_b,_c,^k] M[_j,_d] \\ 
    + delta[_a,^l] D[_b,_k,^j] D[_l,_c,^k] M[_j,_d] 
= G[_a,_b,^k] D[_k,_c,_d] \\
    - D[_a,_k,_l] D[_b,_c,^k] delta[^l,_d] \\ 
    + delta[_a,^l] D[_b,_k,_j] D[_l,_c,^k] delta[^j,_d] .
        """
        G = self.structure_constants()
        M = self.scalar_product_tensor()
        D = self.connection_tensor()
        Dmap = self.connection_map_tensor()
        delta = eye(self._dimension)
        u = tensorcontraction(tensorproduct(G,D),(2,3))
        u -= tensorcontraction(tensorproduct(D,Dmap,delta),(1,5),(2,6))
        u += tensorcontraction(tensorproduct(delta,D,Dmap,delta),(1,5),(3,7),(4,8))
#        u = tensorcontraction(tensorproduct(G,D,M),(2,3),(5,6))
#        u -= tensorcontraction(tensorproduct(D,D,M),(1,5),(2,6))
#        u += tensorcontraction(tensorproduct(delta,D,D,M),(1,5),(3,7),(4,8))
        self._curvature_tensor = u
    def curvature(self,x,y,z,t):
        """\
<R(x,y,z),t>
= x[a]y[b]z[c]t[d] ( G[a,b,k] D[k,c,j] M[j,d] - D[a,k,j] D[b,c,k] M[j,d] + delta[a,l] D[k,b,j] D[l,c,k] M[j,d]  )
=  R[_a,_b,_c,_d] x[^a]y[^b]z[^c]t[^d]
        """
        R = self.curvature_tensor()
        return tensorcontraction(tensorproduct(R,x,y,z,t),(0,4),(1,5),(2,6),(3,7))
    def check_bianchi_identity(self):
        T = self.curvature_tensor()
        dim = self._dimension
        err = []
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    for d in range(dim):
                        test = simplify(T[a,b,c,d] + T[a,c,d,b] + T[a,d,b,c])
                        if test != 0 :
                            err.append([[a,b,c,d],test])
        return err
    def check_symmetries_curvature(self):
        T = self.curvature_tensor()
        dim = self._dimension
        err = []
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    for d in range(dim):
                        test1 = simplify(T[a,b,c,d] + T[b,a,c,d])
                        if test1 != 0:
                            err.append([[a,b,c,d],1,test1])
                        test2 = simplify(T[a,b,c,d] + T[a,b,d,c])
                        if test2 != 0:
                            err.append([[a,b,c,d],2,test2])
                        test3 = simplify(T[a,b,c,d] - T[c,d,a,b])
                        if test3 != 0:
                            err.append([[a,b,c,d],3,test3])
        return err    
    def cook(self):
        print(time.asctime(),"  This can be a long process: take a book to read.")
        print(time.asctime(),"  Building or simplifying the tensor of the scalar product.")
        self._scalar_product_tensor = self.scalar_product_tensor().applyfunc(simplify)
        print(time.asctime(),"  Building or simplifying the tensor of the inverse of the scalar product")
        self._scalar_product_inverse_tensor = self.scalar_product_inverse_tensor().applyfunc(simplify)
        print(time.asctime(),"  Building or simplifying the tensor of the connection")
        self._connection_tensor = self.connection_tensor().applyfunc(simplify)
        print(time.asctime(),"  Building or simplifying the tensor of the map of the connection")
        self._connection_map_tensor = self.connection_map_tensor().applyfunc(simplify)
        print(time.asctime(),"  Building or simplifying the tensor of the curvature")
        self._curvature_tensor = self.curvature_tensor().applyfunc(simplify)
        print(time.asctime(),"  Finished! I hope you had a good time. Bye!")
        print(":)")




def idee():
    testo="""
# HOW TO DEFINE A SYMBOLIC SCALAR PRODUCT:
from sympy import *
import lie_algebras as la
# so(2) x R :
so2 = la.LieAlgebra(4)
so2.define_brackets({ (0,1):[(1,2)] , (1,2):[(2,1)] , (0,2):[(-2,0)] })

# A43 :
A = la.LieAlgebra(4)
A.define_brackets({ (1,3) : [(-1,1)] , (2,3):[(-1,0)] })

ma = Matrix(MatrixSymbol('ma',4,4))
ma[0,3] = ma[1,3] = ma[2,3] = 0
ma[3,0] = ma[3,1] = ma[3,2] = 0
ma[0,0] = 1 
ma[2,0] = 0
ma[1,1] = 1
for i in range(4):
    for j in range(i,4):
        ma[i,j] = ma[j,i]
A.define_scalar_product(ma)

# The linear system :
L = Matrix(MatrixSymbol('L',4,4))
def map_lin(v):
    return tensorcontraction(tensorproduct(L,v),(1,2))

def new_sc(v,w):
    return A.scalar_product(map_lin(v),map_lin(w))
so2.scalar_product = new_sc

def ricdiff(a,b):
    vv = Array(a)
    ww = Array(b)
    return so2.curvature_ricci(vv,ww) - A.curvature_ricci(map_lin(vv),map_lin(ww))

coso = B.curvature_sectional(map_lin(v),map_lin(w)) - A.curvature_sectional(v,w)
coso2 = B.curvature_ricci(map_lin(v),map_lin(w)) - A.curvature_ricci(v,w)

m = Matrix(MatrixSymbol('m',4,4))
m[0,0] = m[3,3] = 1 
m[0,2] = m[2,0] = 0
for i in range(4):
    for j in range(i,4):
        m[i,j] = m[j,i]
so2.define_scalar_product(m)

sd = B.scalar_product(map_lin(v),map_lin(w)) - A.scalar_product(v,w)
def sd_fun(a,b,c,d,aa,bb,cc,dd):
    return sd.subs({v[0]:a, v[1]:b , v[2]:c, v[3]:d, w[0]:aa, w[1]:bb , w[2]:cc, w[3]:dd})

xxx = ricdiff([0,1,0,0],[0,0,0,1])
len(str(xxx))
    """
    print(testo)


