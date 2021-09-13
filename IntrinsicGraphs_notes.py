#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sympy import *
from LieAlgebras import *
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
s, t = symbols('s t', real=True, positive=True)
x, y, z = symbols('x y z')


# In[63]:


# A note about symbols:
etahat = symbols('\hat{\eta}')
display(etahat)
print(etahat)
print(latex(etahat))
display(latex(etahat))


# In[108]:





# In[2]:





# In[3]:





# In[88]:


heis = LieAlgebra()
heis.declare_stratified(growth_vector=[2,1])
heis.structure_constants_build({(0,1) : [(1,2)]})

eta, tau = symbols('eta tau')
eta0, tau0 = symbols('eta0 tau0')
a0, a10, a01, a20, a11, a02 = symbols('a0, a10, a01, a20, a11, a02')

pol = a0 + a10*(eta-eta0) + a01*(tau-tau0) + a20*(eta-eta0)**2 + a11*(eta-eta0)*(tau-tau0) + a02 *(tau-tau0)**2
def f(v):
    f_pol = pol.subs({eta:v[1],tau:v[2]})
    return Array([f_pol,0,0])
v = [0,y,z]
w0 = Array([0,eta0,tau0])

coso = ff(heis,f,t,w0,v)
dd1 = [simplify(limit(cc,t,'oo')) for cc in coso]


# In[163]:


heis2 = LieAlgebra()
heis2.declare_stratified(growth_vector=[4,1])
heis2.structure_constants_build({(0,1) : [(1,4)],(2,3) : [(1,4)]})

# Splitting:
# (0,eta2,xi1,eta1,tau)(xi2,0,0,0,0)

x2, y2, x1, y1, z = symbols('x2, y2, x1, y1, z', real=True)
xi1, eta1, eta2, tau = symbols('xi1, eta1, eta2, tau', real=True)
xi10, eta10, eta20, tau0 = symbols('xi10, eta10, eta20, tau0', real=True)

p0, p1000, p0100, p0010, p0001, p2000, p1100, p1010, p1001, p0200, p0110, p0101, p0020, p0011, p0002 = symbols('p0, p1000, p0100, p0010, p0001, p2000, p1100, p1010, p1001, p0200, p0110, p0101, p0020, p0011, p0002')
pol = p0 + p1000*(eta2-eta20) + p0100*(xi1-xi10) + p0010*(eta1-eta10) + p0001*(tau-tau0) + p2000*(eta2-eta20)*(eta2-eta20) + p1100*(eta2-eta20)*(xi1-xi10) + p1010*(eta2-eta20)*(eta1-eta10) + p1001*(eta2-eta20)*(tau-tau0) + p0200*(xi1-xi10)*(xi1-xi10) + p0110*(xi1-xi10)*(eta1-eta10) + p0101*(xi1-xi10)*(tau-tau0) + p0020*(eta1-eta10)*(eta1-eta10) + p0011*(eta1-eta10)*(tau-tau0) + p0002*(tau-tau0)*(tau-tau0)

def f(v):
    v = list(v)
    xi2, eta2, xi1, eta1, tau = v
    f_pol = p0 + p1000*(eta2-eta20) + p0100*(xi1-xi10) + p0010*(eta1-eta10) + p0001*(tau-tau0) + p2000*(eta2-eta20)*(eta2-eta20) + p1100*(eta2-eta20)*(xi1-xi10) + p1010*(eta2-eta20)*(eta1-eta10) + p1001*(eta2-eta20)*(tau-tau0) + p0200*(xi1-xi10)*(xi1-xi10) + p0110*(xi1-xi10)*(eta1-eta10) + p0101*(xi1-xi10)*(tau-tau0) + p0020*(eta1-eta10)*(eta1-eta10) + p0011*(eta1-eta10)*(tau-tau0) + p0002*(tau-tau0)*(tau-tau0)
#    f_pol = pol.subs({eta2:v[1],xi1:v[2],eta1:v[3],tau:v[4]})
    return Array([f_pol,0,0,0,0])
v = [0,eta2, xi1, eta1, tau]
w0 = [0, eta20, xi10, eta10, tau0]

coso = ff(heis2,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]
#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
for cc in diff:
    display(cc)


# In[164]:


dd = diff[0]
display(simplify(dd.subs({eta2:1,xi1:0,eta1:0,tau:0})))
display(simplify(dd.subs({eta2:0,xi1:1,eta1:0,tau:0})))
display(simplify(dd.subs({eta2:0,xi1:0,eta1:1,tau:0})))
display(simplify(dd.subs({eta2:0,xi1:0,eta1:0,tau:1})))


# In[9]:


heis2 = LieAlgebra()
heis2.declare_stratified(growth_vector=[4,1])
heis2.structure_constants_build({(0,1) : [(1,4)],(2,3) : [(1,4)]})

# Splitting:
# (0,eta1,0,eta2,tau)(xi1,0,xi2,0,0)

xi1, eta1, xi2, eta2, tau = symbols('xi1, eta1, xi2, eta2, tau', real=True)
xi10, eta10, xi20, eta20, tau0 = symbols('xi10, eta10, xi20, eta20, tau0', real=True)

v = [0,eta1, 0, eta2, tau]
w0 = [0, eta10, 0, eta20, tau0]

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('A',vv,1)
    f_pol_2 = polynomial_build('B',vv,1)
    return Array([f_pol_1,0,f_pol_2,0,0])


coso = ff(heis2,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]
#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
for cc in diff:
    display(cc)


# In[27]:


diff[2].subs({eta1:0}).collect(eta2)


# In[6]:


engel = LieAlgebra()
engel.declare_stratified(growth_vector=[2,1,1])
# 0 1 , 2 , 3
engel.structure_constants_build({(0,1) : [(1,2)],(0,2) : [(1,3)]})

# Splitting:
# (0,eta,tau,zeta)(xi,0,0,0)

xi, eta, tau, zeta = symbols('xi, eta, tau, zeta', real=True)
xi0, eta0, tau0, zeta0 = symbols('xi0, eta0, tau0, zeta0', real=True)

v = [0, eta, tau, zeta]
w0 = [0, eta0, tau0, zeta0]

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('f',vv,2)
    return Array([f_pol_1,0,0,0])


coso = ff(engel,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]
#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
for cc in diff:
    display(cc)


# In[11]:


engel = LieAlgebra()
engel.declare_stratified(growth_vector=[2,1,1])
# 0 1 , 2 , 3
engel.structure_constants_build({(0,1) : [(1,2)],(0,2) : [(1,3)]})

# Splitting:
# (xi,0,tau,zeta)(0,eta,0,0)

xi, eta, tau, zeta = symbols('xi, eta, tau, zeta', real=True)
xi0, eta0, tau0, zeta0 = symbols('xi0, eta0, tau0, zeta0', real=True)

v = [xi, 0, tau, zeta]
w0 = [xi0, 0, tau0, zeta0]

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('f',vv,2)
    return Array([0,f_pol_1,0,0])


coso = ff(engel,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]
#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
for cc in diff:
    display(cc)


# In[6]:


free32 = LieAlgebra()
free32.declare_stratified(growth_vector=[3,3])
# 0 1 2, 3 4 5
# 01 = 3
# 02 = 4
# 12 = 5
free32.structure_constants_build({(0,1) : [(1,3)],(0,2) : [(1,4)],(1,2) : [(1,5)]})

# Splitting:
# (0,xi2, xi3, tau1, tau2, tau3) (xi1, 0,0,0,0,0 )

xi1, xi2, xi3, tau1, tau2, tau3 = symbols('xi1, xi2, xi3, tau1, tau2, tau3', real=True)
xi10, xi20, xi30, tau10, tau20, tau30 = symbols('xi10, xi20, xi30, tau10, tau20, tau30', real=True)

v = [0, xi2, xi3, tau1, tau2, tau3]
#w0 = [0, eta0, tau0, 0]
w0 = [0, xi20, xi30, tau10, tau20, tau30]

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('A',vv,2)
#    f_pol_2 = polynomial_build('B',vv,1)
    return Array([f_pol_1,0,0,0,0,0])


coso = ff(free32,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]
#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
for cc in diff:
    display(cc)


# In[2]:


free32 = LieAlgebra()
free32.declare_stratified(growth_vector=[3,3])
# 0 1 2, 3 4 5
# 01 = 3
# 02 = 4
# 12 = 5
free32.structure_constants_build({(0,1) : [(1,3)],(0,2) : [(1,4)],(1,2) : [(1,5)]})

# Splitting:
# (0,0, xi3, 0, tau2, tau3) (xi1, xi2,0,tau1,0,0 )

xi1, xi2, xi3, tau1, tau2, tau3 = symbols('xi1, xi2, xi3, tau1, tau2, tau3', real=True)
xi10, xi20, xi30, tau10, tau20, tau30 = symbols('xi10, xi20, xi30, tau10, tau20, tau30', real=True)

v = [0, 0, xi3, 0, tau2, tau3]
#w0 = [0, 0, xi30, 0, tau20, tau30]
w0 = [0, 0, 0, 0, 0, 0]

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('A',vv,2)
    f_pol_2 = polynomial_build('B',vv,2)
    f_pol_3 = polynomial_build('C',vv,2)
    return Array([f_pol_1,f_pol_2,0,f_pol_3,0,0])

coso = ff(free32,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]

# SURPRISE SURPRISE!
# In this case, the component of 'f' of order 2 
# cannot be dependent on coordinates of degree 1
# What happens is that: what is a Lipschitz function from R to sqrt(R)?
# that is, sqrt(|f(x)-f(y)|) < |x-y| :
# Only constant functions!
# We see this because in coso[3] there are terms with 1/s : 
# we take them away, and 'f' should do it.

# ... and then, higher order derivatives remain exposed!

condition = simplify(s*coso[3]).expand().subs({s:0})
print('Condition to be satisfied:')
display(condition)

coso[3] = simplify(coso[3] - condition/s )


#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
print('The intrinsic differential:')
for cc in diff:
    display(cc)


# In[10]:


basis = free32.basis_symbols()
basis_W = [basis[i] for i in range(1,6)]
display(basis_W)
basis_V = [basis[i] for i in [0]]
display(basis_V)


# In[4]:


def _check_lin_ind(la,vectors):
    vcts = [la.from_symbols_to_array(v) for v in vectors]
    m = Matrix(vcts)
    r = m.rank()
    return r == len(vectors)


# In[5]:


def _check_is_subalg(la,vectors):
    vcts = [la.from_symbols_to_array(v) for v in vectors]
    dim = Matrix(vcts).rank()
    for i in range(len(vcts)):
        for j in range(i+1,len(vcts)):
            v = la(vcts[i],vcts[j])
            if Matrix([*vcts,v]).rank() > dim:
                return False
    return True


# In[7]:


def _check_is_ideal(la,vectors):
    vcts = [la.from_symbols_to_array(v) for v in vectors]
    if not _check_is_subalg(la,vectors):
        print('It is not a subalgebra.')
        return False
    dim = Matrix(vcts).rank()
    basis = la.std_basis()
    for v in vcts:
        for b in basis:
            vb = la(v,b)
            if Matrix([*vcts,vb]).rank() > dim :
                print('It is a subalgebra, but not an ideal.')
                return False
    return True


# In[14]:


def _variables_build(smbl,number):
    var = [Symbol(smbl+'_'+str(i)) for i in range(number)]
    varhat = [Symbol('\hat{'+smbl+'}_'+str(i)) for i in range(number)]
    return var, varhat


# In[15]:


_variables_build('xi',4)


# In[19]:


m = Matrix([[1,2],[3,4]])
m*m.inv()


# In[119]:


def _basis_change_lists(la,vector,basis_from,basis_to):
    """
    *vector* is a list of coordinates in *basis_from*, 
    it returns a list of coordinates in *basis_to*.
    
    In other words, think at basis_from as a matrix whose columns are the vectors,
    and similarly basis_to.
    Then: basis_from * vectors = basis_to*output
    that is, output = basis_to^{-1}*basis_from*vector
    
    if *vector* is a symbolic expression, then it still works, but you should be careful!
    See also _basis_change_symbols
    """
    vector = Matrix(la.from_symbols_to_array(vector))
    b_from = [la.from_symbols_to_array(b) for b in basis_from]
    b_to = [la.from_symbols_to_array(b) for b in basis_to]
    m_from = Matrix(b_from).transpose()
    m_to_inv = Matrix(b_to).transpose().inv()
    res = m_to_inv * m_from * vector
    return res

def _basis_change_symbols(la,vector,basis_from,basis_to):
    """
    Applies _basis_change_lists, but then uses the result to add up in *basis_to*.
    """
    res_list = _basis_change_lists(la,vector,basis_from,basis_to)
    res_zip = [z for z in zip(res_list , basis_to)]
    res = res_zip.pop()
    res = res[0]*res[1]
    for r in res_zip:
        res += r[0]*r[1]
    return res


# In[120]:


_basis_change_symbols(free32,basis[5]+basis[0],basis,free32.std_basis())


# In[66]:


std_b = free32.std_basis()
3*std_b[0]


# In[93]:


sum(term[0]*term[1] for term in res_zip)


# In[106]:


res = []
for term in res_zip:
    res.append( term[0]*term[1])
out = res.pop()
for r in res:
    out += r
out


# In[105]:


0 + res[0]+res[1]


# In[ ]:





# In[ ]:





# In[7]:


# Splitting:
# (0,*,*,*,*,*)(*,0,0,0,0,0) 

basis = free32.basis_symbols()
basis_W = [basis[i] for i in range(1,6)]
basis_V = [basis[i] for i in [0]]

# checks ...

# variables
w, what =  _variables_build('w',len(basis_W))

def f(v_):
    v_ = list(v_)
    vv = [v_[j]-w0[j] for j in range(len(v_))]
    f_pol_1 = polynomial_build('A',vv,2)
    f_pol_2 = polynomial_build('B',vv,2)
    f_pol_3 = polynomial_build('C',vv,2)
    return Array([f_pol_1,f_pol_2,0,f_pol_3,0,0])

coso = ff(free32,f,1/s,w0,v)
coso = [simplify(cc) for cc in coso]

condition = simplify(s*coso[3]).expand().subs({s:0})
print('Condition to be satisfied:')
display(condition)

coso[3] = simplify(coso[3] - condition/s )


#diff = [simplify(limit(cc,s,0)) for cc in coso]
diff = [simplify(cc.subs({s:0})) for cc in coso]
print('The intrinsic differential:')
for cc in diff:
    display(cc)


# In[43]:


aa = free32.dil(1/s,f(free32.dil(s,v)))
aa = simplify(aa)
for a in aa:
    display(a)


# In[73]:


bb = free32.graded_basis_symbols()
bb

