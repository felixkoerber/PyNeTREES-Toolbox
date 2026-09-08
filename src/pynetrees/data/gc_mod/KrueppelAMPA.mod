: AMPA synapse firing at onset time
: biexponential timecourse
: including
: Including NET_RECEIVE statement
: 2009 Roland Krueppel

COMMENT
synaptic current with exponential rise and decay conductance defined by
        i = g * (v - e)      i(nanoamps), g(micromhos);
        where
         g = 0 for t < onset and
         g = amp * (exp(-(t-onset)/tau1)-exp(-(t-onset)/tau2))
          for t > onset
ENDCOMMENT
					       
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS KrueppelAMPA
	RANGE R, R0, onset, tau1, tau2, gmax, e, Fire
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(umho) = (micromho)
}

PARAMETER {
	tau2 = 40.0 (ms)
	tau1 = 0.33 (ms)
	gmax = 1 	 (uS)
	
	e=0	 (mV)
}

ASSIGNED {
	v (mV)
	i (nA)
	R
	R0
	Fire		: 0 precell did not fire, 1 fire in da hood
	lastrelease (ms)	: time of last spike
}

INITIAL {
	lastrelease = -10
	R = 0
	R0 = 0
	Fire = 0
}

NET_RECEIVE ( weight (microsiemens) ) {
	Fire = 1
}

BREAKPOINT {
	SOLVE release
	i = gmax * R *(v - e)
}

PROCEDURE release() {
	if ( Fire ) {
		lastrelease = t
		R0 = R
		Fire = 0
	}
	if ( lastrelease > 0 ) {
		R = exptable(-(t-lastrelease)/tau2) - exptable(-(t-lastrelease)/tau1) + R0 * exptable(-(t-lastrelease)/tau1)
	}
	
	VERBATIM
	return 0;
	ENDVERBATIM
}

FUNCTION exptable(x) {
	TABLE  FROM -100 TO 0 WITH 2000
	if ( (x < 0) && (x > -100) ) {
		exptable = exp(x)
	} else {
		exptable = 0
	}
}

