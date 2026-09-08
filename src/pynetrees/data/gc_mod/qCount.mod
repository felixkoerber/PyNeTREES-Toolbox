UNITS {
(mA) = (milliamp)
(nC) = (nanocoulomb)
(um) = (micron)
(S) = (siemens)
}

NEURON {
SUFFIX qCount
USEION na READ ina
USEION ca READ ica
RANGE ina, ica, q, tracker
}

PARAMETER {
tracker=0
}

STATE {
qNode (nC)
}

ASSIGNED {
ina (mA/cm2)
ica (mA/cm2)
area (um2)
q (nC)
}

BREAKPOINT {
SOLVE states METHOD cnexp
q = qNode
}

INITIAL {
qNode = 0
}

DERIVATIVE states {
qNode' = -(1e-5)*area*ina/3-(1e-5)*area*ica/2
}