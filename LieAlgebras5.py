from sympy import *
from LieAlgebrasTools import *
from copy import copy




# Design principles:
#     - Vectors can be represented and dealt with in three ways:
#         1.  As linear combinations of a family of symbols that form a basis:
#             >>> b1 , b2 = symbols('b1 b2', commutative = False) # -> basis
#             >>> a1 , a2 = symbols('a1 a2') # -> scalars
#             >>> v_s = 3*a1*b1 + a2*b2
#             This basis will be: basis_symbolic
#         2.  As iterables, that can be: arrays, lists or tuples. 
#             Among these optinos, arrays are better because they can be added together.
#             >>> v_i = Array([3*a1, a2])
#             => we will have a function that transforms vectors in bad forms into arrays.
#             This basis will be: basis_coord
#         3.  A third form, which is the matter the vectors are made of. 
#             For example, we could be dealing with the vector space of polynomials of degree 1 in x, 
#             hence we will have that 
#             >>> v_p = 3*a1 x + a2
#             This basis will be: basis_outer
#   - We are not interested in working with specific vectors:
#             for this reason, we do not have a class for vectors, but only for vector space.
#             In the class of vector space there will be three basis and methods to transform an expression from one to the other.
#   - Every data in a class is organized as follows:
#        *   there is a variable _var, defined in the __init__ method
#        *   there is then a getter method
#        *   there is a setter method
#        For example (from the mooc course):
#            >>> class Wallet:
#            >>>     def __init__(self):
#            >>>         self.__money = 0
#            >>>     
#            >>>     # A getter method
#            >>>     @property
#            >>>     def money(self):
#            >>>        return self.__money
#            >>>
#            >>>     # A setter method
#            >>>     @money.setter
#            >>>     def money(self, money):
#            >>>         if money >= 0:
#            >>>             self.__money = money
#     - Every data is defined after the instance of the vector space is created.


class VectorSpace:
    def __init__(self):
        self.name = 'Vector Space'
        self._dimension = None
        self._basis_symbolic = None
        self._basis_coord = None
        self._basis_outer = None
        self.is_graded = False
        self._graded_basis_symbolic = None
        self._graded_basis_coord = None
        self._growth_vector = None
        self._step = None
        self._weights = None
        self._dil_matrix = None
        self._dual_basis_symbolic = None
        self._dual_vector_space = None
    
    @property
    def dimension(self):
        """
Dimension of the vector space `self`.
        """
        if self._dimension == None:
            if self._basis_symbolic != None:
                self._dimension = len(self._basis_symbolic)
            elif self._basis_coord != None:
                self._dimension = len(self._basis_coord)
            elif self._basis_outer != None:
                self._dimension = len(self._basis_outer)
            else:
                raise ValueError("Please, set the dimension.")
        return self._dimension

    @dimension.setter
    def dimension(self,dim):
        self._dimension = dim

    # Symbolic basis 
    @property
    def basis_symbolic(self):
        """
A basis of self made of (non-commutative) symbols.

As linear combinations of a family of symbols that form a basis:
>>> b1 , b2 = symbols('b1 b2', commutative = False) # -> basis
>>> a1 , a2 = symbols('a1 a2') # -> scalars
>>> v_s = 3*a1*b1 + a2*b2
        """
        # If basis_symbolic is not set yet, we build it with the default method
        if self._basis_symbolic == None:
            self._basis_symbolic_build()
        return self._basis_symbolic

    @basis_symbolic.setter
    def basis_symbolic(self, basis = None):
        self.basis_symbolic_set(basis)
        
    def basis_symbolic_set(self, basis = None, smbl = 'b'):
        # if basis is empty, it means we want to construct the default basis (with string smbl, if given).
        if basis == None:
            self._basis_symbolic_build(smbl)
        elif isinstance(basis, (list,tuple)):
            if self._dimension != None and self.dimension != len(basis):
                raise ValueError("The basis does not have the right length: check the dimension")
            # check if is is a list of strings or symbols:
            are_strings = True
            are_symbols = True
            for b in basis:
                are_strings *= isinstance(b,str)
                are_symbols *= isinstance(b,Symbol)
            # if *basis* is a list of strings, each string will be the name of an element of the basis.
            if are_strings:
                self._basis_symbolic = []
                for b in basis:
                    self._basis_symbolic.append(Symbol(b,commutative = False))
            # if basis is a list (or tuple) of symbols, this very list is the basis itself.
            if are_symbols:
                self._basis_symbolic = list(basis)
        # if none of the above happended, something went wrog.
        else:
            raise ValueError("Something went wrong.")

    def _basis_symbolic_build(self,smbl = 'b'):
        if self.is_graded :
            if self._graded_basis_symbolic == None:
                self._graded_basis_symbolic_build(smbl)
            self._basis_symbolic = flatten(self.graded_basis_symbolic)
        else:
            dim = self.dimension
            self._basis_symbolic = [symbols(smbl+'_'+str(j), commutative = False) for j in range(dim)]

    # Coordinate basis
    @property
    def basis_coord(self):
        """
The standard basis of self made of arrays.

As iterables, that can be: arrays, lists or tuples. 
Among these optinos, arrays are better because they can be added together.
So, vectors can be arrays.
>>> v_i = Array([3*a1, a2])
        """
        if self._basis_coord == None:
            self._basis_coord_build()
        return self._basis_coord

    def _basis_coord_build(self):
    # the standard basis is made of column vectors like: Array([1,0,0],(3,1))
        dim = self.dimension
        std_basis = eye(dim).rowspace()
        std_basis = [ Array(list(ec), (dim,1)) for ec in std_basis ]
        self._basis_coord = std_basis
     
    # Outer basis
    @property
    def basis_outer(self):
        """
A basis of self made of things.

A third form of a basis, which is the matter the vectors are made of. 
It is a list.
For example, we could be dealing with the vector space of polynomials of degree 1 in x, hence we will have that 
>>> vector_space.basis_outer
[x,1]
>>> v_p = 3*a1 x + a2
        """
        if self._basis_outer == None:
            self._basis_outer_build()
        return self._basis_outer

    @basis_outer.setter
    def basis_outer(self, basis:list):
        if not isinstance(basis,(list,tuple)):
            raise ValueError("The outer basis should be a list.")
        if self._dimension != None and self.dimension != len(basis):
            raise ValueError("The length of the outer basis should match the dimension")
        self._basis_outer = list(basis)
        self._dimension = len(self._basis_outer)

    def _basis_outer_build(self):
        # This function is meant to be overridden.
        raise ValueError("An outer basis is not set yet.")
    
    def from_symbols_to_array(self,v):
        """
Transforms a linear combination of symbols into an array.

>>> [b1, b2] = vector_space.basis_symbolic
>>> a1, a2 = symbols('a1 a2')
>>> v = 3*a1*b1 + a2*b2
>>> vector_space.from_symbols_to_array(v)
[[3*a1],[a2]]
        """
        basis = self.basis_symbolic
        dim = self.dimension
        v_list = from_symbols_to_list(v,basis)
        v_array = Array(v_list , (dim,1))
        return v_array

    def from_array_to_symbols(self,v):
        """
Transform an array into a linear combination of the symbolic basis.

It works also if v is a list, which is an handy way to construct symbolic vectors.

Example:
========
>>> a1 , a2 = symbols('a1 a2')
>>> vector_space.basis_symbolic = ['b1', 'b2']
>>> vector_space.basis_symbolic
[b1, b2]
>>> v = Array([3*a1,a2],(2,1))
>>> vector_space.from_array_to_symbols(v)
3*a1*b1 + a2*b2
>>> vector_space.from_array_to_symbols([3*a1,a2])
3*a1*b1 + a2*b2
>>> vector_space.from_array_to_symbols(['x','y'])
x*b1 + y*b1
        """
        dim = self.dimension
        basis = self.basis_symbolic
        if isinstance(v,Array):
            if v.shape != (dim,1):
                raise ValueError(f"The array should have the shape ({dim},1)")
            v_list = flatten(list(v))
        elif isinstance(v,list):
            if len(v) != dim:
                raise ValueError(f"The list should have legth {dim}")
            # We add a catchy method: if given a list of strings, we use them as names of symbols.
            are_strings = True
            for x in v:
                are_strings *= isinstance(x,str)
            if are_strings:
                v_list = [Symbol(name) for name in v]
            else:
                v_list = v
        else:
            raise ValueError(f"Something went wrong with {v}")
        return Add(*[xj*bj for xj,bj in zip(v_list,basis)])

    def from_symbols_to_outer(self,v):
        """
Transform a linear combination of symbols into a linear combination of the external objects
        """
        basis = self.basis_outer
        if isinstance(v,Array):
            v_array = v
        else:
            v_array = self.from_symbols_to_array(v)
        v_list = flatten(v_array)
        return Add(*[xj*bj for xj,bj in zip(v_list,basis)])

    def from_outer_to_symbols(self,v):
        """
Transform a linear combination of external objects into a linear combination of symbols.

This method needs to be set by the user, 
because it highly depends on the type of external objects.
        """
        pass
    
    def from_array_to_outer(self,v):
        """
Transform an array into a linear combination of the external objects.
        """
        # It might be silly to do this, 
        # because v_symbolic returns to array in the method
        # self.from_symbols_to_outer
        # but in this way we have all the facilitation from the method self.from_array_to_symbols .
        v_symbolic = self.from_array_to_symbols(v)
        return self.from_symbols_to_outer(v_symbolic)

    def from_outer_to_array(self,v):
        """
Transform a linear combination of external objects into an array.

This method needs to be set by the user, 
because it highly depends on the type of external objects.
        """
        pass
    def a_vector_array(self, smbl:str):
        """
        Returns a vector of self.

        *v* is a string that is used to define symbols v1,...,vn for the components of the output.
        The output is an Array.
        """
        dim = self.dimension
        return Array(symbols(smbl + '_:%d' %self._dimension), (dim,1))
    
    def a_vector_symbolic(self, smbl:str):
        """
        Returns a vector of self.

        *v* is a string that is used to define symbols v1,...,vn for the components of the output.
        The output is a linear combinatio of symbols.
        """
        return self.from_array_to_symbols( list( symbols(smbl + '_:%d'%self.dimension)))

    @property
    def dual_basis_symbolic(self):
        if self._dual_basis_symbolic == None:
            self._dual_basis_symbolic_build()
        return self._dual_basis_symbolic

    def _dual_basis_symbolic_build(self,pre_symbol_for_dual='@',post_symbol_for_dual=''):
        basis = self.basis_symbolic
        self._dual_basis_symbolic = [ \
                symbols(pre_symbol_for_dual + vect.name + post_symbol_for_dual, commutative=False) \
                for vect in basis ]
    
    @property
    def dual_space(self):
        if self._dual_vector_space == None:
            self._dual_vector_space = VectorSpace()
            self._dual_vector_space.basis_symbolic = self.dual_basis_symbolic
        return self._dual_vector_space        
    def from_dual_symbols_to_array(self,v):
        """
Transforms a linear combination of symbols into an array.
        """
        basis = self.dual_basis_symbolic
        dim = self.dimension
        v_list = from_symbols_to_list(v,basis)
        v_array = Array(v_list , (1,dim))
        return v_array

    def pairing_dualVSvect_symbolic(self,v,w):
        """
Given a covector v and a vector w (both symbolic), returns v(w).
        """
        v_dual = self.from_dual_symbols_to_array(v)
        w_vect = self.from_symbols_to_array(w)
        return my_contraction(tensorproduct( v_dual , w_vect ),[(0,1)])

    @property
    def growth_vector(self):
        """
The growth vector of vector_space, when it is graded.

The growth vector is a list of integers [a,b,...] 
that correspond to the dimensions of layers.
The sum a+b+... should match the dimension!
        """
        return self._growth_vector

    @growth_vector.setter
    def growth_vector(self,gr_vct:list):
        dim = sum(flatten(gr_vct))
        if self._dimension != None and self._dimension != dim:
            raise ValueError("This growth vector does not match the dimension of the space.")
        self._growth_vector = gr_vct
        self._dimension = dim
        self.is_graded = True

    @property
    def graded_basis_symbolic(self):
        """
When graded, it gives a symbolic graded basis.

A graded basis is a list of lists:
    [[b11,b12,...],[b21,b22,...],...]
    where the i-th list is the basis of the i-th layer.
        """
        if self._graded_basis_symbolic == None:
            self._graded_basis_symbolic_build()
        return self._graded_basis_symbolic

    @graded_basis_symbolic.setter
    def graded_basis_symbolic(self,basis:list):
        if self._dimension != None and self._dimension != len(flatten(basis)):
            raise ValueError("The total length of the basis does not match the dimension.")
        # If the growth vector is already set, we can give just a list of elements without separating them into layers.
        # NOTE: but, in this way, we could input a graded basis with a different growth vector than the one setted, and this method will not raise any error.
        if self.growth_vector != None:
            basis = flatten(basis)
            basis_graded = []
            for dim_j in self.growth_vector:
                basis_graded.append(basis[:dim_j])
                basis = basis[dim_j:]
            self._graded_basis_symbolic = basis_graded
        else: # if self.growth_vector == None:
            for basis_j in basis:
                if not isinstance(basis_j,list):
                    raise ValueError("It is not a list of lists")
            self._graded_basis_symbolic = basis
            self._dimension = sum([len(basis_j) for basis_j in basis])
            self.is_graded = True
        # If there is no basis already, this method will set it as the one that is graded.
        if self._basis_symbolic == None:
            self._basis_symbolic = flatten(self._graded_basis_symbolic)

    def _graded_basis_symbolic_build(self, smbl : str = 'b', force_build : bool = False):
        """
Builds a graded basis following the growth vector.
        """
        if not self.is_graded or self._growth_vector == None:
            raise ValueError("Error from graded_basis_symbols_build : The algebra must be graded")
        self._graded_basis_symbolic = []
        growth_vector = self.growth_vector
        # If there is not already a basis, or if we are forcing this method, we construct a new graded basis.
        if self._basis_symbolic == None or force_build:   
            for j in range(len(growth_vector)):
                base_j_layer = []
                for k in range(growth_vector[j]):
                    base_j_layer.append(symbols(\
                        smbl + '^' + str(j) + '_' +str(k),\
                        commutative=False))
                self._graded_basis_symbolic.append(base_j_layer)
            # if we arrived here by force (force_build == True), we probably do not want to overwrite the already defined symbolic basis.
            if not force_build:
                self._basis_symbolic = flatten(self._graded_basis_symbolic)
        # If there is already a symbolic basis, we use that to make the ordered one.
        else: # if self._basis_symbolic != None:
            basis = self.basis_symbolic.copy()
            for m in growth_vector:
                self._graded_basis_symbolic.append(basis[:m])
                basis = basis[m:]

    @property
    def graded_basis_coord(self):
        if self._graded_basis_coord == None:
            self._graded_basis_coord_build()
        return self._graded_basis_coord

    def _graded_basis_coord_build(self):
        basis = self.basis_coord.copy()
        gr_vct = self.growth_vector
        basis_gr = []
        for m in gr_vct:
            basis_gr.append(basis[:m])
            basis = basis[m:]
        self._graded_basis_coord = basis_gr

    @property
    def step(self):
        if self._step == None and self._growth_vector != None:
            self._step = len(self._growth_vector)
        return self._step
    
    @step.setter
    def step(self,s:int):
        if self._growth_vector != None and s != len(self._growth_vector):
            raise ValueError("The step does not match the growth vector")
        self._step = s

    @property
    def weights(self):
        if self._weights == None:
            self._weights_build()
        return self._weights
    
    def _weights_build(self):
        gr_vct = self.growth_vector
        self._weights = flatten( [ gr_vct[j] * [j+1] for j in range(len(gr_vct))])
    
    @property
    def dil_matrix(self):
        if self._dil_matrix == None and self._growth_vector!= None:
            self._dil_matrix = diag(*self.weights)
        return self._dil_matrix

    def dil(self, l:float, v:Array):
        dim = self.dimension
        if not isinstance(v,Array) or v.shape != (dim,1):
            raise ValueError("The vector is not array with the right shape")
        w = self.weights
        matrix = diag(*[ l**w[i] for i in range(dim)])
        return matrix @ v

    def subspace(self, basis:list, smbl:str):
        """
        Returns the subspace with basis.
        
        The given basis will be the outer basis of the subspace.
        The string smbl is used to make the symbolic basis of the subspace.
        
        Warning! The string smbl should be different from the string used in the ambient vector space V,
        otherwise the program will use the same symbols for the basis...
        Things may break.
        
        Example:
        ========
        
        >>> V = VectorSpace()
        >>> V.dimension = 3
        >>> B = V.basis_symbolic
        >>> W = V.subspace([B[0]+B[2],B[2]],'a')
        
        This method can be used also for a change of basis:
        
        >>> W = V.subspace([B[0],B[0]+B[1],B[2]],'a')
        
        
        """
        W = VectorSpace()
        W.basis_outer = basis
        W.basis_symbolic_set(None, smbl)
        A = LinMap()
        A.domain = W
        A.range = self
        A.rules = {a:W.from_symbols_to_outer(a) for a  in W.basis_symbolic}
        W.from_symbols_to_outer = A
        W.from_outer_to_symbols = A.pseudo_inverse
        return W

    def is_subalgebra_of(self, liealgebra:'LieAlgebra', force=False):
        """
        Returns true if self (a vector space) is a subalgebra of lieaglebra.

        """
        if not force and self.from_symbols_to_outer.range is not liealgebra:
            raise ValueError("The subspace given does not seem to be a subspace of the given Lie algebra. If you know better, you can force the method adding 'force = True'.")
        subspace = self
        subbasis = subspace.basis_outer
        subdim = subspace.dimension
        for j in range(subdim):
            for i in range(j):
                a = subbasis[i]
                b = subbasis[j]
                ab = liealgebra(a,b)
                abc = ab - subspace.from_symbols_to_outer(subspace.from_outer_to_symbols(ab))
                abc = simplify(abc)
                # Now, abc is [a,b] - pr([a,b]) , where pr:liealgebra -> subspace is a projection.
                if abc != 0:
                    print(i,a)
                    print(j,b)
                    print(abc)
                    return False
        return True
class MultLinMap():
    def __init__(self):
        self._domains = []
        self._range = None
        self._tensor_repr = None
        self._rules = None
        self.is_antisymmetric = None
        self.is_symmetric = None
    
    def __call__(self,*vectors):
        return self.apply(*vectors)

    @property
    def num_domains(self):
        return len(self._domains)

    @property
    def dimensions(self):
        return [vs.dimension for vs in self.domains]

    @property
    def domains(self):
        """
List of vector spaces.

If there is only one, that is, if this is a linear map, then returns the domain.
        """
        return self._domains
    
    @domains.setter
    def domains(self,*doms):
        if isinstance(doms[0], list):
            self._domains = doms[0]
        elif isinstance(doms[0],tuple):
            self._domains = list(doms[0])
        else:
            self._domains = list(doms)

    @property
    def range(self):
        """
One vector space that is the range of the map.
        """
        return self._range

    @range.setter
    def range(self, vct_sp : VectorSpace):
        self._range = vct_sp

    @property
    def rules(self):
        """
A dictionary that defines the map,

that is, dictionary {(b1,b2,...):w, ... }, so that L(b1,b2,...) = w .
If the map is just linear, then the dictionary is of the form
{(b1,):w, ... } .
        """
        return self._rules

    @rules.setter
    def rules(self,ruls : dict):
        # TODO: check type (dict), form (as described above).
        # Maybe, it could be useful to have a method that checks that the map is well defined?
        self._rules_set(ruls)

    def _rules_set(self, ruls:dict):
        # This method is needed becasue it will be different for the subclass LinMap.
        self._rules = ruls
        if self.is_antisymmetric:
            self.make_antisymmetric()
        if self.is_symmetric:
            self.make_symmetric()

    def make_antisymmetric(self):
        # NOTE: this works only if the it is a bilinear map
        diz = self._rules
        new_diz = diz.copy()
        for b1,b2 in diz.keys():
            new_diz[(b2,b1)] = - diz[(b1,b2)]
        self._rules = new_diz
    
    def make_symmetric(self):
        # NOTE: this works only if the it is a bilinear map
        diz = self._rules
        new_diz = diz.copy()
        for b1,b2 in diz.keys():
            new_diz[(b2,b1)] = diz[(b1,b2)]
        self._rules = new_diz

    @property
    def as_tensor(self):
        """
A tensor representation of the map.

This is handy when dealing with arrays.
It should be an Array.
        """
        if self._tensor_repr == None and self._rules != None:
            self.make_tensor_from_rules()
        return self._tensor_repr

    @as_tensor.setter
    def as_tensor(self,tns: Array):
        # TODO: check type (Array), dimensions,
        self._tensor_repr = tns
    
    def apply(self,*vectors):
        """
The application of the map to vectors.
        """
        # In the case this is a linear map, we can accept just a vector and not a list of vectors.
        # OR: it could be a tuple of vectors, like in L(v1,v2)
        if not isinstance(vectors,list):
            vectors = list(vectors)
        # If they are all arrays, we will apply the tensor product method
        are_arrays = prod([isinstance(v,Array) for v in vectors])
        if are_arrays:
            return self._apply_as_tensor(vectors)
        # Otherwise, we apply a more symbolic method
        return self._apply_as_symbolic(vectors)

    def _apply_as_tensor(self,vector_list : list):
        """
Application of the map as tensor product.

If L[_i1,_i2,...] is the linear map and [v^i1,v^i2,...] is the list of vectors,
then
L[v0,v1,....]
= L[_i0,_i1,...] * v^i0 * v^i1 * ...
        """
        n_doms = self.num_domains
        if len(vector_list) != n_doms:
            raise ValueError("The number of vectors is not the same as the number of domains")
        # construction of the list of pairs of axis we will contract.
        # If n_doms = 2, we have
        # [(0,2),(1,3)]
        pairs_for_contraction = [(1+i,1+i+n_doms) for i in range(n_doms)]
        # the method is in two stpes:
        # first, tensorize everything, obtaining
        # L[_i0,_i1,...,_i(n-1)] * v^in * v^i(n+1) * ... * v^i(2n-1)
        all_tensorized = tensorproduct(self.as_tensor , *vector_list)
        # and then contracts the indices ( _ij,^i(j+n) )
        return my_contraction( all_tensorized , pairs_for_contraction )
    
    def _apply_as_symbolic(self, vector_list : list):
        """
Application of the map in a programmatic way
        """
        n_doms = self.num_domains
        if len(vector_list) != n_doms:
            raise ValueError("The number of vectors is not the same as the number of domains")
        for j in range(n_doms):
            # if the j-th vector is a sum, we return the sum of the values.
            if isinstance(vector_list[j],Add):
                # make a list A of lists, so that A[j][i] is vector_list[i] for i≠j, and A[j][j] is the j-th element of the sum
                new_vector_lists = []
                for v in vector_list[j].args:
                    new_vector_lists.append(\
                            [vector_list[i] if i != j else v for i in range(n_doms) ]\
                            )
                return Add(*[self._apply_as_symbolic(vectors) for vectors in new_vector_lists])
            # if the j-th element is the product of a scalar by a vector, the scalar comes out.
            # a scalar is a number or an commutative symbol
            if isinstance(vector_list[j],Mul):
                # The product can have commutative and non-commutative parts.
                # for instance:
                # a = symbols('a') # a scalar
                # x = symbols('x',commutative = False) # a vector
                # v = 3*a*x
                # b.args_cnc()
                # -> [[3,a],[x]]
                comm, non_comm = vector_list[j].args_cnc()
                # if len(comm) == 0, then we must go on.
                # if len(comm) != 0, then we take it out (in the next iteration of the function, at this stage we will have len(comm)==0 ).
                if len(comm) != 0:
                    new_vector_list = [vector_list[i] if i!= j else prod(non_comm) for i in range(n_doms)]
                    return Mul(*comm) * self._apply_as_symbolic(new_vector_list)
        # finally, if vector_list is one of those appearing in the rules, then we apply them. Otherwise it is assumed to be zero.
        if tuple(vector_list) in self.rules.keys():
            return self.rules[tuple(vector_list)]
        else:
            return 0

    def make_tensor_from_rules(self):
        dims = self.dimensions
        doms = self.domains
        ran = self.range
        dim_r = ran.dimension
        shape = [dim_r,1] + flatten([ [1,n] for n in dims ])
        
        bases_doms_dual = [[Array(v) for v in eye(n).rowspace()] for n in dims]
        bases_doms_dirt = [vc_sp.basis_symbolic for vc_sp in doms]
        
        combs = combinations([list(range(n)) for n in dims])
        res = Array(prod(shape)*[0], shape)
        
        for cc in combs:
            vs_dual = []
            vs_dirt = []
            for j in range(len(cc)):
                vs_dual.append(bases_doms_dual[j][cc[j]])
                vs_dirt.append(bases_doms_dirt[j][cc[j]])
            vs = [ ran.from_symbols_to_array( self.apply(*vs_dirt))  ] + vs_dual
            res += tensorproduct(*vs)
        self._tensor_repr = res

class LinMap(MultLinMap):
    """
Is not much more than a multilinear map (MultLinMap).
    """
    def __init__(self):
        super().__init__()
        self._pseudo_inverse = None
        self._as_matrix = None

    def _rules_set(self, ruls : dict):
        if not isinstance(list(ruls.keys())[0],tuple):
            ruls_adapted = {}
            for b in ruls.keys():
                ruls_adapted[(b,)] = ruls[b]
        else:
            ruls_adapted = ruls
        self._rules = ruls_adapted
     
    @property
    def domain(self):
         return self.domains[0]

    @domain.setter
    def domain(self, dom: VectorSpace):
        self.domains = [dom]
    
    @property
    def as_matrix(self):
        if self._as_matrix == None:
            self._as_matrix_build()
        return self._as_matrix

    @as_matrix.setter
    def as_matrix(self, mtrx: Matrix):
        self._as_matrix = mtrx
        self._make_rules_from_matrix()

    def _make_rules_from_matrix(self):
        mtrx = self.as_matrix
        dom = self.domain
        ran = self.range
        basis_dom = self.domain.basis_symbolic
        diz = {}
        for b in basis_dom:
            diz[b] = ran.from_array_to_symbols( Array( mtrx * Matrix(dom.from_symbols_to_array(b)) ) )
        self.rules = diz

    def _as_matrix_build(self):
        dim_dom = self.domain.dimension
        dim_ran = self.range.dimension
        basis = self.domain.basis_coord
        M = zeros(dim_ran, dim_dom)
        for i in range(dim_dom):
            for j in range(dim_ran):
                M[j,i] = self.apply(basis[i])[j,0]
        self._as_matrix = M
    
    @property
    def pseudo_inverse(self):
        if self._pseudo_inverse == None:
            self._pseudo_inverse_build()
        return self._pseudo_inverse
    
    def _pseudo_inverse_build(self):
        pinv = LinMap()
        pinv.domain = self.range
        pinv.range = self.domain
        pinv.as_matrix = self.as_matrix.pinv()
        self._pseudo_inverse = pinv


class LieAlgebra(VectorSpace):
    def __init__(self):
        super().__init__()
        self.name = 'Lie Algebra'

        self.brackets = MultLinMap()
        self.brackets.domains = [self, self]
        self.brackets.range = self
        self.brackets.is_antisymmetric = True
        
        self._structure_constants = None

        self._rules_vector_fields = None
        
        self.is_nilpotent = None
        self.is_graded = None
        self.is_stratified = None

        self._bch_precision = 3

        self._a_basis_of_brackets = None
        self._a_basis_of_brackets_graded = None
        self._a_basis_of_brackets_dict = None
        self._a_basis_of_brackets_matrix = None

        self._a_basis_symbolic_of_brackets = None
        self._a_basis_symbolic_of_brackets_graded = None
        self._a_basis_symbolic_of_brackets_dict = None
        self._a_basis_symbolic_of_brackets_dict_V1 = None

        self._a_basis_symbolic_of_brackets_dict_V1_anti = None

        self._generic_derivation = None
        self._generic_derivation_graded = None
    
    def __call__(self,v,w):
        return self.brackets.apply(v,w)
    
    @property
    def structure_constants(self):
        """
Returns the structure constants of the Lie algebra,

that is a tensor Gamma of shape (dim,dim,dim) so that
self(b_i,b_j) = sum_k Gamma[^k,_i,_j] b_k
        """
        if self._structure_constants == None:
            self._structure_constants_build()
        return self._structure_constants
    
    def _structure_constants_build(self):
        if self.brackets.as_tensor != None:
            dim = self.dimension
            self._structure_constants = self.brackets.as_tensor.reshape(dim,dim,dim)

    def check_jacobi(self, verbose = False):
        """
Checks the Jacobi identity using the symbolic basis

The Jacobi identity is
$$
[a,[b,c]] + [b,[c,a]] + [c,[a,b]] == 0
$$
If *verbose* is *True*, then it prints the triples a, b, c where Jacobi fails,
together with the indices i, j, k of these vectors in the basis,
and the non-zero result of the Jacobi sum.

        """
        basis = self.basis_symbolic
        dim = self.dimension
        itis = True
        for i in range(dim):
            for j in range(i):
                for k in range(j):
                    a = basis[i]
                    b = basis[j]
                    c = basis[k]
                    jacobi = simplify( self(a,self(b,c)) + self(b,self(c,a)) + self(c,self(a,b)) ) 
                    if jacobi != 0:
                        itis = False
                        if verbose:
                            print('Jacobi identity [a,[b,c]] + [b,[c,a]] + [c,[a,b]] fails with')
                            print('i = ', i)
                            print('a = ', a)
                            print('j = ', j)
                            print('b = ', b)
                            print('k = ', k)
                            print('c = ', c)
                            print('res = ', jacobi)
                            print()
        return itis

    @property
    def rules_vector_fields(self):
        """
        Example in the Heisenberg group:
        rules[y*x] = x*y - z
        i.e.,
        rules = { y*x : x*y - z }
        """
        if self._rules_vector_fields == None:
            self.rules_vector_fields_build()
        return self._rules_vector_fields

    def rules_vector_fields_build(self):
        basis = self.basis_symbolic
        dim = self.dimension
        diz = {}
        for i in range(dim):
            for j in range(i):
                diz[basis[i]*basis[j]] = basis[j] * basis[i] + self.brackets(basis[i],basis[j])
        self._rules_vector_fields = diz
    def declare_nilpotent(self, step : int = None, isit : bool = True):
        self.is_nilpotent = isit
        self._step = step

    def declare_graded(self, growth_vector : list = None, step : int = None, isit : bool = True):
        if step == None and growth_vector != None:
            step = len(growth_vector)
        self.declare_nilpotent(step,isit)
        self.growth_vector = growth_vector
        self.is_graded = isit

    def declare_stratified(self, growth_vector : list = None, step : int = None, isit : bool = True):
        self.declare_graded(growth_vector,step,isit)
        self.is_stratified = isit
    def bch(self,v,w):
        """
        Baker–Campbell–Hausdorff formula
        https://en.wikipedia.org/wiki/Baker–Campbell–Hausdorff_formula
        with sum up to the step of self,
        which needs to be nilpotent.
        Otherwise, use bch_trnc(v,w,N) with level of precision N.
        """
        if self.step == None:
            print(f"BCH formula trucated at {self._bch_precision} layer. If you want to change this, change LieAlgebra._bch_precision.")
            return self.bch_trnc(v,w,self._bch_precision)
        else:
            return self.bch_trnc(v,w,self.step)

    def bch_trnc(self,v,w,N):
        """\
        Baker–Campbell–Hausdorff formula
        https://en.wikipedia.org/wiki/Baker–Campbell–Hausdorff_formula
        with sum up to N
        """
        res = 0*v
        for n in range(1,N+1):
            coef = Rational( Integer( (-1)**(n-1) ) , Integer(n) )
            RS = self._list_rs(n,N)
            RS_sum = 0*v
            for rs in RS:
                somma = Integer(sum(rs))
                rs_fact = [factorial(x) for x in rs]
                prodotto = Integer(prod(rs_fact))
                r = [rs[2*x] for x in range(n)]
                s = [rs[2*x+1] for x in range(n)]
                RS_sum = RS_sum + Rational(1, somma*prodotto ) * self._multbra(r,v,s,w)
            res = res + coef * RS_sum
        return res

    def _multbra_u(self,r,v,s,w,u):
        """
        Method for the BCH formula.

        Compute
        \[
        [v,[v,...,[v,[w,[w,...,[w,u]]]]]]]
        \]
        where v appears r times, and w appears s times.
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

    @property
    def a_basis_of_brackets(self):
        """
        A basis of brackets, not organized into layers.

        Returns a basis in coordinates for *self* out of the set of all brackets of vectors of the first strata.

        Works only for stratified Lie algebras!
        """
        if not self._a_basis_of_brackets:
            self._a_basis_of_brackets_build()
        return self._a_basis_of_brackets
    
    @property
    def a_basis_of_brackets_graded(self):
        """
        A basis of brackets, organized into layers.

        Returns a basis in coordinates for *self* out of the set of all brackets of vectors of the first strata.

        Works only for stratified Lie algebras!
        """
        if not self.is_stratified:
            raise ValueError('Error 61137d91: self must be stratified. Method cannot work.')
        if not self._a_basis_of_brackets_graded:
            self._a_basis_of_brackets_build()
        return self._a_basis_of_brackets_graded 

    @property
    def a_basis_of_brackets_dict(self):
        """
        A dictionary of parents for vectors in LieAlgebra.a_basis_of_brackets.

        A dictionary to see for each vector in self.a_basis_of_brackets who are his parents.
        One parent is in the first layer, the second parent is in the previous layer.
        """
        if not self._a_basis_of_brackets_dict:
            self._a_basis_of_brackets_build()
        return self._a_basis_of_brackets_dict

    def _a_basis_of_brackets_build(self):
        """
        Builds self._a_basis_of_brackets_graded , self._a_basis_of_brackets , and self._a_basis_of_brackets_dict

        that is, 
        a basis in coordinates for *self*
        out of the set of all brackets of the first layer,
        both organized in layers and flattened,
        and a dictionary that tells you for each element of the basis which are the two vectors it is a bracket of.

        Works only for stratified Lie algebras.
        """
        if not self.is_stratified:
            print('Error from _a_basis_of_brackets_build_: self must be stratified. Method cannot work.')
            return 
        dim = self.dimension
        dim_1 = self.growth_vector[0]
        basis_1 = self.basis_coord[:dim_1]
        basis = [basis_1]
        basis_decoupling = {}
        while True:
            vects_trpl = [[v,w,self.brackets(v,w)] for v in basis_1 for w in basis[-1]]
            vects = [VV[2] for VV in vects_trpl]
            # Since this method starts from the last element, 
            # for the subsequent lines it is better to reverse vects
            vects.reverse()
            vects = [Array(v) for v in maximal_lin_ind(vects)]
            if vects == []:
                break
            for bb in vects:
                for VV in vects_trpl:
                    if bb == VV[2]:
                        basis_decoupling[bb] = VV[:2]
            basis.append(vects)
        self._a_basis_of_brackets_graded = basis
        self._a_basis_of_brackets = flatten(basis,levels=1)
        self._a_basis_of_brackets_dict = basis_decoupling
    
    @property
    def a_basis_of_brackets_matrix(self):
        """
        Returns a matrix M whose colomuns are the coordinates of the standard basis in the new basis.
        """
        if self._a_basis_of_brackets_matrix == None:
            self._a_basis_of_brackets_matrix_build()
        return self._a_basis_of_brackets_matrix

    def _a_basis_of_brackets_matrix_build(self):
        basis = self._a_basis_of_brackets
        self._a_basis_of_brackets_matrix = Matrix(basis).transpose().inv()

    @property
    def a_basis_symbolic_of_brackets(self):
        if self._a_basis_symbolic_of_brackets == None:
            self._a_basis_symbolic_of_brackets_build()
        return self._a_basis_symbolic_of_brackets

    @property
    def a_basis_symbolic_of_brackets_graded(self):
        if self._a_basis_symbolic_of_brackets_graded == None:
             self._a_basis_symbolic_of_brackets_build()
        return  self._a_basis_symbolic_of_brackets_graded

    def _a_basis_symbolic_of_brackets_build(self):
        # First we use the one made of arrays:
        bas_bra = self.a_basis_of_brackets
        bas_bra_s = [ self.from_array_to_symbols(v) for v in bas_bra ]
        self._a_basis_symbolic_of_brackets = bas_bra_s
        # And then we graded it:
        bas_bra_s_g = []
        gr_vect = self.growth_vector
        for m in gr_vect:
            bas_bra_s_g.append(bas_bra_s[:m])
            bas_bra_s = bas_bra_s[m:]
        self._a_basis_symbolic_of_brackets_graded = bas_bra_s_g

    @property
    def a_basis_symbolic_of_brackets_dict(self):
        """
Like self.a_basis_of_brackets_dict(), but using symbolic basis.
        """
        if not self._a_basis_symbolic_of_brackets_dict:
            self._a_basis_symbolic_of_brackets_dict_build()
        return self._a_basis_symbolic_of_brackets_dict

    def _a_basis_symbolic_of_brackets_dict_build(self):
        diz = self.a_basis_of_brackets_dict
        diz_s = {}
        for v in diz.keys():
            [a, b] = diz[v]
            v = self.from_array_to_symbols(v) 
            a = self.from_array_to_symbols(a)
            b = self.from_array_to_symbols(b)
            diz_s[v] = a*b - b*a
        self._a_basis_symbolic_of_brackets_dict = diz_s
    
    @property
    def a_basis_symbolic_of_brackets_dict_V1(self):
        """
Returns a dictionary {z: x*y-y*x}

so that z is the bracket x and y, and both x and y are in the first layer only.
So, it is like self.a_basis_symbolic_of_brackets_dict(),
but where only elements of the first layer appear
        """
        if not self._a_basis_symbolic_of_brackets_dict_V1:
            self._a_basis_symbolic_of_brackets_dict_V1_build()
        return  self._a_basis_symbolic_of_brackets_dict_V1
    
    def _a_basis_symbolic_of_brackets_dict_V1_build(self):
        diz_s = self.a_basis_symbolic_of_brackets_dict
        for j in range(self.step - 1):
             for v in diz_s.keys():
                diz_s[v] = diz_s[v].subs(diz_s)
        self._a_basis_symbolic_of_brackets_dict_V1 = diz_s
    
    def boil_to_V1(self,X):
        """
Takes a symbolic expression X and returns it
with only elements in the first layer.

Works only in stratified Lie algebras.

The result is a priori not unique: 
the algorithm chooses one just as it is built.
        """
        try:
            return X.subs(self.a_basis_symbolic_of_brackets_dict_V1 )
        except:
            return X
    
    def boil_to_V1_anti(self,X):
        """
Returns a symbolic expression as anti-lie brackets of elements in the first layer

Works only in stratified Lie algebras.

The result is a priori not unique: 
the algorithm chooses one just as it is built.

This method is meant to be used to contruct an lie-algebra anti-morphism given a linear map on the first layer.
A Lie algebra anti-morphism is a linear map L between Lie algebras such that
$$
L([x,y]) = - [Lx,Ly]
$$
        """
        try:
            return X.subs(self.a_basis_symbolic_of_brackets_dict_V1_anti )
        except:
            return X

    @property
    def a_basis_symbolic_of_brackets_dict_V1_anti(self):
        """
Returns a dictionary {z: -(x*y-y*x)}

so that z is the bracket x and y, and both x and y are in the first layer only.
So, it is like self.a_basis_symbolic_of_brackets_dict(),
but where only elements of the first layer appear
        """
        if self._a_basis_symbolic_of_brackets_dict_V1_anti == None:
            self._a_basis_symbolic_of_brackets_dict_V1_anti_build()
        return  self._a_basis_symbolic_of_brackets_dict_V1_anti
    
    def _a_basis_symbolic_of_brackets_dict_V1_anti_build(self):
        """
Builds self.a_basis_symbolic_of_brackets_dict_V1_anti .
        """
        diz_s = self.a_basis_symbolic_of_brackets_dict
        diz_anti = {}
        # If z = [x,y], then we find z:xy-yx and we change it into z:-(xy-yx)
        for key in diz_s.keys():
            diz_anti[key] = - diz_s[key]
        for j in range(self.step - 1):
             for v in diz_anti.keys():
                diz_anti[v] = diz_anti[v].subs(diz_anti)
        self._a_basis_symbolic_of_brackets_dict_V1_anti = diz_anti
    
    @property
    def generic_derivation(self):
        """
A matrix D that represent all derivations of the Lie algebra.
        """
        if self._generic_derivation == None:
            self._generic_derivation_build()
        return self._generic_derivation

    def _generic_derivation_build(self):
        la = self
        dim = la.dimension
        # Construct a generic derivation
        der = LinMap()
        der.domain = la
        der.range = la
        DD = Matrix(MatrixSymbol('D',dim,dim))
        der.as_matrix = DD
        
        # build list of linear conditions
        conditions = []
        for i in range(dim):
            for j in range(i):
                a = la.basis_symbolic[i]
                b = la.basis_symbolic[j]
                conditions.append( la(der(a),b) + la(a,der(b)) - der(la(a,b)) )
        # flatten out the list of conditions
        conditions_list = [ list(la.from_symbols_to_array(c)) for c in conditions ]
        conditions_list = flatten(conditions_list)
        
        # Solve the conditions (they are linear equations)
        conditions_list = list(set(conditions_list))
        solutions = linsolve(conditions_list,list(DD)) # <-- This may take time!
        solutions = list(*solutions)
        
        # Make the solution into a dictionary {der[i,j]: ... }
        solutions_diz = {}
        DD_list = list(DD)
        for dd in DD_list:
            solutions_diz[dd] = solutions[DD_list.index(dd)]
        
        # Apply the rules found as solutions to the generic linear map:
        der.as_matrix = DD.subs(solutions_diz)
        self._generic_derivation = der

    @property
    def generic_derivation_graded(self):
        """
A matrix D that represent all strata-preserving derivations of the Lie algebra.
        """
        if not self.is_stratified:
            raise ValueError("The lie algebra must be stratified.")
        if self._generic_derivation_graded == None:
            self._generic_derivation_graded_build()
        return self._generic_derivation_graded

    def _generic_derivation_graded_build(self):
        la = self
        dim = la.dimension
        # We could start with the generic derivation, but maybe not.
        # der = la.generic_derivation
        # Construct a generic derivation
        der = LinMap()
        der.domain = la
        der.range = la
        DD = Matrix(MatrixSymbol('D',dim,dim))
        der.as_matrix = DD
        
        # build list of linear conditions
        conditions = []
        for i in range(dim):
            for j in range(i):
                a = la.basis_symbolic[i]
                b = la.basis_symbolic[j]
                conditions.append( la(der(a),b) + la(a,der(b)) - der(la(a,b)) )
        # flatten out the list of conditions
        conditions_list = [ list(la.from_symbols_to_array(c)) for c in conditions ]
        conditions_list = flatten(conditions_list)

        # conditions of mapping V_1 to V_1 (i.e., of being strata-preserving)
        dimV1 = la.growth_vector[0]
        for b in la.basis_symbolic[:dimV1]:
            b_der = la.from_symbols_to_array(der(b))
            b_der = flatten(list(b_der))
            conditions_list.extend(b_der[dimV1:])
        
        # Solve the conditions (they are linear equations)
        conditions_list = list(set(conditions_list))
        solutions = linsolve(conditions_list,list(DD)) # <-- This may take time!
        solutions = list(*solutions)
        
        # Make the solution into a dictionary {der[i,j]: ... }
        solutions_diz = {}
        DD_list = list(DD)
        for dd in DD_list:
            solutions_diz[dd] = solutions[DD_list.index(dd)]
        
        # Apply the rules found as solutions to the generic linear map:
        der.as_matrix = DD.subs(solutions_diz)
        self._generic_derivation_graded = der

    def lie_subalgebra(self, basis:list, smbl:str):
        """Returns the Lie subalgebra of self with basis 'basis'.


        The given basis will be the outer basis of the subspace.
        The string smbl is used to make the symbolic basis of the subspace.
        
        Warning! The string smbl should be different from the string used in the ambient vector space V,
        otherwise the program will use the same symbols for the basis...
        Things may break.


        """
        W = LieAlgebra()
        W.basis_outer = basis
        W.basis_symbolic_set(None, smbl)
        A = LinMap()
        A.domain = W
        A.range = self
        A.rules = {a:W.from_symbols_to_outer(a) for a  in W.basis_symbolic}
        W.from_symbols_to_outer = A
        W.from_outer_to_symbols = A.pseudo_inverse
        if not W.is_subalgebra_of(self):
            raise ValueError('The given basis does not form a subalgebra!')

        # Construct the Lie brackets of the subalgebra
        rules = {}
        dimW = W.dimension
        basisW = W.basis_symbolic
        basisW_outer = W.basis_outer
        for j in range(dimW):
            for i in range(j):
                a = basisW[i]
                ao = basisW_outer[i]
                b = basisW[j]
                bo = basisW_outer[j]
                rules[(a,b)] = W.from_outer_to_symbols(self(ao,bo))
        W.brackets.rules = rules
        return W

    def conditions_for_morphisms_to(self, la , smbl='L'):
        """Returns a linear map L from self to la and a list of conditions on L
        """
        L = LinMap()
        dim1 = self.dimension
        dim2 = la.dimension
        L.domain = self
        L.range = la
        L.as_matrix = Matrix(symarray(smbl,(dim2,dim1)))
        conditions = []
        for j in range(dim1):
            for i in range(j):
                a = self.basis_symbolic[i]
                b = self.basis_symbolic[j]
                conditions.append( la.from_symbols_to_array( L( self(a,b) ) - la( L(a),L(b) ) ) )
        conditions = flatten(conditions)
        return L, conditions

    def morphisms_to(self, la, smbl='L'):
        """Returns all Lie Algebra morphisms from self to la.


Example:
--------
>>> heis = LieAlgebra()
>>> heis.dimension = 3
>>> [X,Y,Z] = heis.basis_symbolic
>>> heis.brackets.rules = {(X,Y):Z}
>>> 
>>> heis2 = LieAlgebra()
>>> heis2.dimension = 5
>>> [X1,Y1,X2,Y2,Z] = heis2.basis_symbolic
>>> heis2.brackets.rules = {(X1,Y1):Z, (X2,Y2):Z}
>>> 
>>> W1 = heis2.lie_subalgebra([X1,Y1,Z],'a')
>>> 
>>> W2 = heis2.lie_subalgebra([X1,X2,Z],'aa')
>>> 
>>> L, sol = heis.morphisms_to(W1)
>>> L.as_matrix = L.as_matrix.subs(sol[0]) 
>>> print(L.as_matrix)
Matrix([[L_0_0, L_0_1, 0], [L_1_0, L_1_1, 0], [L_2_0, L_2_1, L_0_0*L_1_1 - L_0_1*L_1_0]])
>>> 
>>> M, sol = heis.morphisms_to(W2,smbl='M') # One can choose the letter for the unknown morphism
>>> M.as_matrix = M.as_matrix.subs(sol) # One needs to check if to sol is a list or not
>>> print(M.as_matrix)
Matrix([[M_0_0, M_0_1, 0], [M_1_0, M_1_1, 0], [M_2_0, M_2_1, 0]])
>>> 
>>> K, sol = heis.morphisms_to(heis2,smbl='K')
>>> K.as_matrix = K.as_matrix.subs(sol[0])
>>> print(K.as_matrix)
Matrix([[K_0_0, K_0_1, 0], [K_1_0, K_1_1, 0], [K_2_0, K_2_1, 0], [K_3_0, K_3_1, 0], [K_4_0, K_4_1, K_0_0*K_1_1 - K_0_1*K_1_0 + K_2_0*K_3_1 - K_2_1*K_3_0]])


        """
        L, conditions = self.conditions_for_morphisms_to(la,smbl)
        sol = solve(conditions, list(L.as_matrix), manual=1, dict=True)
        #sol = solve(conditions, manual=1, dict=True)
        return L, sol
    
class JetAlgebra(LieAlgebra):
    def __init__(self):
        super().__init__()
        self.name = "Jet Lie Algebra"
        self._lie_algebra_domain = None
        self._target_vector_space = LineVectorSpace
        self._order = None
        self._indices_graded = None
        self._indices = None
        self._basis_symbolic_dict = None
        self._basis_HD_dict = None
        self._basis_HD_graded_dict = None
        self._basis_outer_dict = None

    def build_me(self):
        tot = 7
        def header(a,b):
            return f"Step {a} of {b}:"
        print(header(1,tot),"Construct set of indices:")
        print(self.indices)
        print()
        print(header(2,tot),"Construct HD basis:")
        print(self.basis_HD_dict)
        print()
        print(header(3,tot),"and outer basis:")
        print(self.basis_outer)
        print()
        print(header(4,tot),"Construct sybolic basis:")
        print(self.basis_symbolic)
        print()
        print(header(5,tot),"Construct growth vector")
        gr_vct = [len(idxs) for idxs in self.indices_graded.values()]
        self.declare_stratified(gr_vct)
        print(self.growth_vector)
        print()
        # We construct the transformation rules 
        # THIS OPERATION CAN USE A LOT OF TIME!
        print(header(6,tot),"Construct functions from outer basis to the others.")
        self.from_outer_to_others_build()
        print()
        print(header(7,tot),"Construct Lie bracket operation")
        self.lie_brackets_build()
        print()
        print("Done")

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self,m : int):
        self._order = m

    @property
    def lie_algebra_domain(self):
        return self._lie_algebra_domain

    @lie_algebra_domain.setter
    def lie_algebra_domain(self,lad : LieAlgebra):
        self._lie_algebra_domain = lad

    @property
    def target_vector_space(self):
        return self._target_vector_space

    @target_vector_space.setter
    def target_vector_space(self,tvs : VectorSpace):
        self._target_vector_space = tvs

    @property
    def step(self):
        if self._step == None:
            self._step = max(self.order + 1, self.lie_algebra_domain.step)
        return self._step

    @property
    def indices_graded(self):
        """
Dictionary that gives all indices for each layer.
        """
        if self._indices_graded == None:
            self._indices_graded_build()
        return self._indices_graded

    @property
    def indices(self):
        if self._indices == None:
            self._indices = flatten(self.indices_graded.values(), levels = 1)
        return self._indices

    def _indices_graded_build(self):
        lad = self.lie_algebra_domain
        order = self.order
        # First, we construct the indices related to the basis of lad.
        indices_neg = []
        basis_std = lad.graded_basis_coord
        for layer_j in basis_std:
            indices_j = []
            for v in layer_j:
                indices_j.append((tuple(flatten(-v)),0))
            indices_neg.append(indices_j)
        # Second, we construct the indices related to the spaces HD
        indices_pos = []
        indices_pos_0 = build_graded_indices_dict(lad.growth_vector, order)
        for j in range(order+1):
            new_indices_j = []
            for a in indices_pos_0[j]:
                for b in self.target_vector_space.basis_symbolic:
                    new_indices_j.append((a,b))
            indices_pos.append(new_indices_j)
        indices_pos.reverse()
        # Third, we construct the indices for the last layer of the jet space:
        zrs = tuple(lad.dimension * [0])
        indices_T = [(zrs,b) for b in self.target_vector_space.basis_symbolic ]
        # Fourth, we put them together, ordered by layer of the JetAlgebra, as a dictionary
        indices = {}
        for j in range(1,self.step+1):
            indices[j] = []
            if j <= lad.step:
                indices[j].extend(indices_neg[j-1])
            if j<= self.order:
                indices[j].extend(indices_pos[j-1])
        indices[self.step].extend(indices_T)
        # Finally, we save the result in the right variable.
        self._indices_graded = indices

    @property
    def basis_symbolic_dict(self):
        if self._basis_symbolic_dict == None:
            self._basis_symbolic_dict_build()
        return self._basis_symbolic_dict
    
    def _basis_symbolic_dict_build(self,smbl = 'A'):
        symbols_basis = {}
        for idx in self.indices:
            symbols_basis[idx] = Symbol(smbl + '^' + str(idx[1]) + '_' + str(idx[0]), commutative = False)
        self._basis_symbolic_dict = symbols_basis
    
    def _basis_symbolic_build(self,smbl = 'b'):
        # This is an override from vector space
        basis_dict = self.basis_symbolic_dict
        indices = self.indices
        # at this point, one coul just define 
        # basis = flatten(basis_dict.values)
        # but we want that the order is the same as in indices.
        basis = []
        for idx in indices:
            basis.append(basis_dict[idx])
        self._basis_symbolic = basis
    
    @property
    def basis_coord_dict(self):
        pass
    
    @property
    def basis_HD_dict(self):
        if self._basis_HD_dict == None :
            self._basis_HD_build()
        return self._basis_HD_dict

    @property
    def basis_HD_graded_dict(self):
        if self._basis_HD_graded_dict == None :
            self._basis_HD_build()
        return self._basis_HD_graded_dict

    def _basis_HD_build(self):
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
        lad = self.lie_algebra_domain
        basis = lad.basis_symbolic
        dual_basis = lad.dual_basis_symbolic
        basis_V1 = lad.graded_basis_symbolic[0]
        rules = lad.rules_vector_fields
        order = self.order
        # Let's start the construction: this will be a dictionary
        self._basis_HD_graded_dict = {}
        self._basis_HD_dict = {}
        for order_a in range(order + 1):
            # self.indices_graded is layered along with the stratification of JetAlgebra, which is inverse to the order of derivations.
            indices_a_hom = self.indices_graded[order + 1 - order_a].copy()
            # from this list we remove the indices related to lad:
            indices_a_hom = [idx for idx in indices_a_hom if idx[-1]!=0 ]
            basis_HD_a = {}
            # If order_a == 0, the construction is easy : { ((0,0,0),wj) : wj }
            if order_a == 0:
                for idx in indices_a_hom:
                    basis_HD_a[idx] = idx[1]
            else:
                basis_T_a_V1 = build_monomials_nc(basis_V1,order_a)[order_a]
                for idx in indices_a_hom:
                    A = 0
                    idx_mon = idx[0]
                    idx_vect = idx[1]
                    mon = monomial_ordered(basis,idx_mon)
                    for xi in basis_T_a_V1:
                        tau_bb = noncomm_pol_dict(isubs_force(xi,rules))
                        tau_bb_I = tau_bb.get(mon,0)
                        A += tau_bb_I * dualize(reverse_order(xi),basis,dual_basis) * idx_vect
                    basis_HD_a[idx] = A
            self._basis_HD_graded_dict[order_a] = basis_HD_a
            # This merges the two dictionaries
            self._basis_HD_dict = {**self._basis_HD_dict , **basis_HD_a}

    @property
    def basis_outer_dict(self):
        if self._basis_outer_dict == None:
            self._basis_outer_dict_build()
        return self._basis_outer_dict

    def _basis_outer_dict_build(self):
        indices = self.indices
        lad_basis = self.lie_algebra_domain.basis_symbolic
        HD_dict = self.basis_HD_dict
        diz = {}
        for idx in indices:
            if idx[1] == 0:
                # idx = ((0,...,-1,...),0), where -1 is at the j-th position
                j = idx[0].index(-1)
                diz[idx] = (lad_basis[j])
            else:
                diz[idx] = HD_dict[idx]
        self._basis_outer_dict = diz

    def _basis_outer_build(self):
        diz = self.basis_outer_dict
        indices = self.indices
        basis = []
        # Using the dictionary to build the outer basis allows us to use the standard ordering given by the indices.
        for idx in indices:
            basis.append(diz[idx])
        self.basis_outer = basis
    
    def from_outer_to_others_build(self):
        """
A complicated operation.
        """
        print("Building transformation operations: this can take a lot of time. You may want to change the source code.")
        order = self.order
        monomials = []
        for vv in self.basis_outer:
            monomials.extend(noncomm_pol_dict(vv).keys())
        monomials = list(set(monomials))
        pol_space = VectorSpace()
        pol_space.basis_outer = monomials
        def fun(v):
            v_coeff = noncomm_pol_dict(v)
            v_list = [v_coeff.get(b,0) for b in monomials]
            return pol_space.from_array_to_symbols(v_list)

        L = LinMap()
        L.domain = self
        L.range = pol_space
        diz = {}
        for b in self.basis_symbolic:
            diz[b] = fun( self.from_symbols_to_outer(b) )
        L.rules = diz
        Linv = L.pseudo_inverse
        self.from_outer_to_symbols = lambda v:  Linv(fun(v))
        self.from_outer_to_array = lambda v: self.from_symbols_to_array( Linv(fun(v)) )

    def rcontr(self,X,v):
        """
returns v rcontr X, with both X and v written in the outer basis.
        """
        v = self.lie_algebra_domain.boil_to_V1_anti(v)
        v = expand(v)
        X = expand(X)
        return self._rcontr_1(X,v)
    
    def _rcontr_1(self,X,v):
        if isinstance(v,Add):
            return sum([self._rcontr_1(X,z) for z in v.args])
        if isinstance(v,Pow):
            return self._rcontr_1( self._rcontr_2(X, v.base ) , Pow(v.base,v.exp-1) )
        if isinstance(v,Mul):
            comm , non_comm = v.args_cnc()
            if len(non_comm) == 1:
                return Mul(*comm) * self._rcontr_2(X, non_comm[-1])
            else:
                return Mul(*comm) * self._rcontr_1(self._rcontr_2(X, non_comm[-1]) , Mul(*non_comm[:-1] ) )
        #            print(0, comm,non_comm)
        #            res =  Mul(*comm) * self._rcontr_1(self._rcontr_2(X, non_comm[-1]) , Mul(*non_comm[:-1] ) )
        #            print(1, Mul(*comm), self._rcontr_1(self._rcontr_2(X, non_comm[-1]) , Mul(*non_comm[:-1] ) ))
        #            print(2, self._rcontr_2(X, non_comm[-1]) , Mul(*non_comm[:-1] ) )
        #            return res
        return self._rcontr_2(X,v)
        
    def _rcontr_2(self,X,v):
        if isinstance(X,(int,float)) or X.is_number:
            return 0
        if isinstance(X,Add):
            return sum([self._rcontr_2(z,v) for z in X.args])
        if isinstance(X,Mul):
            comm , non_comm = X.args_cnc()
            X_comm = Mul(*comm)
            X_target = non_comm[-1]
            if X_target not in self.target_vector_space.basis_symbolic:
                return 0
            X_domain = non_comm[:-1]
            X_last = X_domain[-1]
            X_rest = X_domain[:-1]
            return X_comm * Mul(*X_rest) * self._rcontr_3(X_last,v) * X_target
        return self._rcontr_3(X,v)
    
    def _rcontr_3(self,X,v):
        if isinstance(X,Pow):
            return Pow(X.base, X.exp - 1) * self._rcontr_3(X.base, v) 
        else:
            return self.lie_algebra_domain.pairing_dualVSvect_symbolic(X,v)

    def rcontr_symbolic(self,X,v):
        """
returns v rcontr X, with both X and v written in the symbolic basis.
        """
        return self.from_outer_to_symbols( self.rcontr( self.from_symbols_to_outer(X), self.from_symbols_to_outer(v) ) )

    def bracketJet(self,v,w):
        """
Returns the brackets [v,w] in jet, in the outer basis.

lad(v,w) + rcontr(v,w) - rcontr(w,v)
        """
        return  self.lie_algebra_domain(v,w) + self.rcontr(v,w) - self.rcontr(w,v)

    def lie_brackets_build(self):
        """
lad(v,w) + rcontr(v,w) - rcontr(w,v)
        """
        def is_in_domain(v):
            return v[1] == 0
        lad = self.lie_algebra_domain
        indices = self.indices
        basis_o = self.basis_outer_dict
        basis_s = self.basis_symbolic_dict
        dim = self.dimension
        diz = {}
        for i1 in range(dim):
            idx1 = indices[i1]
            for i2 in range(i1):
                idx2 = indices[i2]
                b1_s = basis_s[idx1]
                b1_o = basis_o[idx1]
                b2_s = basis_s[idx2]
                b2_o = basis_o[idx2]
                if is_in_domain(idx1) and is_in_domain(idx2): # then also is_in_domain(idx2)
                    diz[( b1_s, b2_s )] = self.from_outer_to_symbols( lad(b1_o,b2_o) )
                elif is_in_domain(idx1): # but now not is_in_domain(idx2), thanks to the 'elif'
                    diz[( b1_s, b2_s )] = - self.from_outer_to_symbols( self.rcontr(b2_o,b1_o) )
                elif is_in_domain(idx2): # but now not is_in_domain(idx1), thanks to the 'elif'
                    diz[( b1_s, b2_s )] = self.from_outer_to_symbols( self.rcontr(b1_o,b2_o) )
                else: # neither are in domain
                    diz[( b1_s, b2_s )] = 0
        self.brackets.rules = diz
                    
    def _sd_prod_build(self):
        """
Build the semidirect product.

(exp(x),A)(exp(y),B) = (exp(x)exp(y) , B + e^{y rcontr } A
        """
        HD = VectorSpace()
        indices = self.indices
        basis = self.basis_symbolic_dict
        HD.basis_symbolic = [basis[idx] for idx in indices if idx[1]!=0]
        self.HD = HD

        rcontr_w = LinMap()
        rcontr_w.domain = HD
        rcontr_w.range = HD
        w = self.lie_algebra_domain.a_vector_symbolic('w')
        w = self.from_outer_to_symbols(w)
        rcontr_w.rules = { b: self.rcontr_symbolic(b,w) for b in HD.basis_symbolic}
        self.rcontr_w = rcontr_w

        exp_rcontr_w = LinMap()
        exp_rcontr_w.domain = HD
        exp_rcontr_w.range = HD
        exp_rcontr_w.as_matrix = exp(self.rcontr_w.as_matrix)
        self.exp_rcontr_w = exp_rcontr_w

    def lad_part_symbolic(self,v):
        """
Return the part of v that belongs to the lie_algebra_domain.
        """
        v_out = self.from_symbols_to_outer(v)
        v_out = expand(v_out)
        v_dict = noncomm_pol_dict(v_out)
        basis_lad = self.lie_algebra_domain.basis_symbolic
        v_lad = 0
        for b in basis_lad:
            v_lad += v_dict.get(b,0)*b
        return v_lad

    def sd_prod(self,v,w):
        """
(exp(x),A)(exp(y),B) = (exp(x)exp(y) , B + e^{y rcontr } A
        """
        x = self.lad_part_symbolic(v)
        A = v - self.from_outer_to_symbols(x)
        y = self.lad_part_symbolic(w)
        B = w - self.from_outer_to_symbols(y)
        lad = self.lie_algebra_domain
        w = lad.a_vector_symbolic('w')
        exp_rcontr_y = copy(self.exp_rcontr_w)
        exp_rcontr_y.as_matrix = vect_subs(exp_rcontr_y.as_matrix , w, y)
        return self.from_outer_to_symbols( lad.bch(x,y) ) + B + exp_rcontr_y(A)
    
    @property
    def one(self):
        return self.target_vector_space.basis_symbolic[0]

    def A(self,*idx):
        idx = tuple(idx)
        if sum(list(idx)) < 0:
            return self.basis_symbolic_dict[(idx,0)]
        else:
            return self.basis_symbolic_dict[(idx,self.one)]
        
# Useful instances:
LineVectorSpace = VectorSpace()
LineVectorSpace.basis_symbolic = ['@1']
