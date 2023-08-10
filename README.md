# LieAlgebras and their jet spaces
The goal is the construction Jet spaces as described in the paper
https://arxiv.org/abs/2201.04534 .

The main files are `LieAlgebras5.py` with the companion `LieAlgebrasTools.py`.
A presentation/guide is the Jupyter Notebook `LieAlgebras5_presentation.ipynb`.

The main classes defined are:
- VectorSpace
- MultLinMap
- LinMap(MultLinMap)
- LieAlgebra(VectorSpace)
- JetAlgebra(LieAlgebra)

It also defines at the the instance `LineVectorSpace` of the class `VectorSpace`
with symbolic basis `LineVectorSpace.basis_symbolic = ['@1']`.

The other files are not important and possibly broken.
