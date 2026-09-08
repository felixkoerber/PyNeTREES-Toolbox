TITLE passive membrane channel

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX pasq
	USEION na READ ena WRITE ina
	USEION k READ ek
	NONSPECIFIC_CURRENT i
	RANGE g, e, ina
}

PARAMETER {
	g = .001	(S/cm2)	<0,1e9>
	e = -70	(mV)
}

ASSIGNED {v (mV)  
		i (mA/cm2)
		ina (mA/cm2)
		ena (mV)
		ek (mV)
}

BREAKPOINT {
 	ina = g*(ek-e)/(ek - ena)*(v - ena)
	i = - ina
}
