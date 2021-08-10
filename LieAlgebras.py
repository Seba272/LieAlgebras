from sympy import *
import time as time

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
        [x,y,z]
        >>> monomials[2]
        [x*x,x*y,y*x,y*y]
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
                monomials += [ a*b for a in monomials[-1] for b in variables ]
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
    x*y - z
    >>> isubs(y*x*x*x*x*x*x*x*x*x*x*x*x*x,rules,100)
    -13*x**12*z + x**13*y
    >>> isubs(y*x*x*x*x*x*x*x*x*x*x*x*x*x,rules,5)
    max iter reached!
    -5*x**4*z*x**8 + x**5*y*x**8

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
    print('Warning: Max iterations reached!')
    return expr_simplified, iterazione

class LieAlgebra:
    dimension = None
    basis_symbols = None
    is_nilpotent = None
    step = None
    is_graded = None
    is_stratified = None
    growth_vector = None
    graded_basis_symbols = None
    basis_LI_diff_operators = None
    rules_vector_fields = None
    basis_tensors_symbols = None
    def __init__(self,dim,struct_const='Gamma'):
        self._dimension = dim
        self.structure_constants_build(struct_const)
    def structure_constants(self):
        try:
            self._structure_constants
        except AttributeError:
            self.structure_constants_build(self)
        return self._structure_constants
    def structure_constants_build(self,rules=None):
        """\
A.define_brackets(rules)
where ``rules`` is a dictionary with entries (index1,index2):[(coeff, index_vect),...]
{(i,j):[(a,2),(b,3)]}
means that bracket(bi,bj) = a*b2 + b*b3
Always i<j !!
Some examples:
Heisenberg Lie algebra (dim = 3): heis = {(0,1) : [(1,2)]}
so(2) (dim = 3) : so2 = { (0,1):[(1,2)] , (1,2):[(2,1)] , (0,2):[(-2,0)] }
so(2)xR (dim = 4) : so2R = so2 
A_{4,3} (dim = 4) : a43 = { (1,3) : [(-1,1)] , (2,3):[(-1,0)] }
        """
        if rules == None or type(rules) == str :
            self._structure_constants_build_abstract(rules)
        else:
            self._structure_constants_build_from_rules(rules)
    def _structure_constants_build_abstract(self,struct_const):
        """\
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
    def brackets(self,v,w):
        """\
brackets(v,w)[^k] = Gamma[_i,_j,^k]v[^i]w[^j]
        """
        return tensorcontraction(tensorproduct(self._structure_constants,v,w),(0,3),(1,4))
    def check_jacobi(self):
        """\
 G[_a,_d,^e] G[_b,_c,^d] + G[_b,_d,^e] G[_c,_a,_d] + G[_c,_d,^e] G[_a,_b,^d]
 but it is easier to check it on symbolic vectors.
        """
#        G = self.structure_constants()
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
    def a_vector(self,v):
        return Array(symbols(v+'[:%d]'%self._dimension))
    def _multbra_u(self,r,v,s,w,u):
        """\
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
        """\
Compute
\[
[v,[v,...,[v,[w,...[w,[v,...[v,w...
\]
r[1] volte v, s[1] volte w, r[2] volte v, s[2] volte w, .... , r[n] volte v, s[n] volte w.
Attenzioen che gli ultimi sono delicati, perché [w,w]=0, etc...
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
        """\
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
    def _prod_list(self,ll):
        p = 1
        for x in ll:
            p=p*x
        return p
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
                prodotto = self._prod_list(rs_fact)
                r = [rs[2*x] for x in range(n)]
                s = [rs[2*x+1] for x in range(n)]
                RS_sum = RS_sum + (somma*prodotto)**(-1) * self._multbra(r,v,s,w)
            res = res + coef * RS_sum
        return res
    def bch(self,v,w):
        """\
Baker–Campbell–Hausdorff formula
https://en.wikipedia.org/wiki/Baker–Campbell–Hausdorff_formula
with sum up to the step of self,
which needs to be nilpotent.
Otherwise, use bch_trnc(v,w,N) with level of precision N.
        """
        if self.is_nilpotent == True and self.step!=None :
            return self.bch_trnc(v,w,self.step)
        else:
            print('Error: This algebra is not nilpotent. Use bch_trnc(v,w,N) with level of precision N.')
            return None
    def declare_nilpotent(self,isit=True,step=None):
        self.is_nilpotent = isit
        self.step = step
    def check_nilpotent(self):
        print('Is nilpotent? ',self.is_nilpotent,' step: ',self.step)
        return None
    def declare_graded(self,isit=True,step=None,growth_vector=None):
        self.is_nilpotent = isit
        self.is_graded = isit
        self.step = step
        self.growth_vector = growth_vector
    def declare_stratified(self,isit=True,step=None,growth_vector=None):
        self.is_nilpotent = isit
        self.is_graded = isit
        self.is_stratified = isit
        self.step = step
        self.growth_vector = growth_vector
    def set_step(self,step=None):
        if self.step != None:
            print("At the moment, the step is ",self.step)
        if step==None:
            self.step = int(input("What is the step?"))
        else:
            self.step = step
        print("step set to ",self.step," (You should check that the Lie algebra is nilpotent!)")
    def build_graded_basis_symbols(self):
        """
Per Heisenberg:
x,y,z = symbols('x y z',commutative=False)
self.graded_basis_symbols = [[x,y],[z]]
self.basis_symbols = flatten(self.graded_basis_symbols)
xd,yd,zd = symbols('x^+ y^+ z^+',commutative=False) # dual elements
self.dual_basis_symbols = [xd,yd,zd]
        """
        if self.is_graded == None or self.is_graded:
            print('The algebra must be graded')
            return None
        self.graded_basis_symbols = []
        for j in range(len(self.growth_vector)):
            base_j_layer = []
            for k in range(growth_vector[j]):
                base_j_layer.append(symbols('b^'+str(j)+'_'+str(k),commutative=False))
            self.graded_basis_symbols.append(base_j_layer)
        self.basis_symbols = flatten(self.graded_basis_symbols)
    def build_dual_basis_symbols(self,symbol_for_dual='@'):
        if self.basis_symbols == None:
            print('Error: first you need to basis_symbols')
            return None
        self.dual_basis_symbols = [ \
                symbols(b_vect.name+symbol_for_dual,commutative=False) \
                for b_vect in self.basis_symbols ]
    def build_basis_tensors(self,order):
        """
Build the list basi = self.basis_tensors_symbols, 
where basi[k] is the basis of tensors of order k,
for k from 0 to 'order':
basi[0] = [1]
basi[1] = self.basis_symbols
basi[2] = [a*b for a in basi[1] for b in basi[1]]
etc...
basi[order] = ...
Notice that len(basi) = order + 1 
        """
        if order >= 0:
            self.basis_tensors_symbols = [[1]]
        if order >= 1:
            self.basis_tensors_symbols += [self.basis_symbols]
        if order > 1:
            for j in range(order-1):
                self.basis_tensors_symbols = [ \
                        a*b for a in self.basis_tensors_symbols[-1] for b in self.basis_symbols \
                        ]
    def build_rules_vector_fields(self):
        """
Example in the Heisenberg group:
rules[y*x] = x*y - z
i.e.,
rules = { y*x : x*y - z }
        """
        if self.basis_symbols == None:
            print('Error: first you need to build basis_symbols')
            return None
        basis = self.basis_symbols
        dim = self.dimension
        self.rules_vector_fields = {}
        for i in range(self.dimension):
            for j in range(i):
                bi = Array(flatten(eye(dim)[i,:]))
                bj = Array(flatten(eye(dim)[j,:]))
                bk = list(self.brackets(bi,bj))
                res = basis[j]*basis[i]
                for k in range(dim):
                    res += bk[k]*basis[k]
                self.rules_vector_fields[basis[i]*basis[j]] = res












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

