## Formulation:

Max-cut problem can be easily formulated as Binary Integer Programming as follows:

$max \sum_{(i,j) \in E} w_{ij} e_{ij}$

$\text{Subject to:}$

$e_{ij} \le v_i + v_j, \quad \forall (i,j) \in E$

$e_{ij} \le 2 - (v_i + v_j), \quad \forall (i,j) \in E$

$e_{ij}, v_i, v_j \in \{0, 1\}, \quad \forall (i,j) \in E$

It is solved in python by `python-MIP` [CBC solver](https://github.com/coin-or/Cbc) interface by default
And if possible - by Gurobi