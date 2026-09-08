COMMENT
Implementation of a hybrid biophysical synaptic plasticity rule with three types of plasticity
See Ebner et al. (2018)
ENDCOMMENT

NEURON {
	POINT_PROCESS SynHybrid
	: Parameters & variables of the original Exp2Syn
	RANGE tau_a, tau_b, e, i
	NONSPECIFIC_CURRENT i
	: Parameters & variables of the plasticity rule
	RANGE A_LTD_pre, A_LTD_post, A_LTP_post
	RANGE tau_G2_a, tau_G2_b, tau_u, tau_alpha, tau_gamma
	RANGE theta_minus, theta_plus, theta_u1, theta_u2
	RANGE m_C1, m_G2, m_alpha, m_beta, s_beta
	RANGE u_bar, G1, C1, E, G2, C2, P, K_alpha, K_alpha_bar, Rho, K_beta, K_gamma, K
	RANGE g_max
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	: Parameters of the original Exp2Syn
	tau_a = 0.5 (ms) <1e-9,1e9>		: time constant of EPSP rise
	tau_b = 5 (ms) <1e-9,1e9>		: time constant of EPSP decay
	e = 0 (mV)						: reversal potential
	g_max = 0.001 (uS)				: (initial) maximal conductance

	: Parameters of the plasticity rule
	A_LTD_pre = 8.5e-7				: amplitude of pre-LTD
	A_LTD_post = 3.6e-7				: amplitude of post-LTD
	A_LTP_post = 5.5e-5				: amplitude of post-LTP
	tau_G2_a = 2 (ms) <1e-9,1e9>	: time constant of the presynaptic event (rise)
	tau_G2_b = 50 (ms) <1e-9,1e9>	: time constant of the presynaptic event (decay)
	tau_u = 10 (ms) <1e-9,1e9>		: time constant for filtering u
	tau_alpha = 15 (ms) <1e-9,1e9>	: time constant for filtering K_alpha to calculate K_alpha_bar
	tau_gamma = 20 (ms) <1e-9,1e9>	: time constant for filtering K_beta to calculate K_gamma
	theta_u1 = -60					: voltage threshold 1
	theta_u2 = -68					: voltage threshold 2
	theta_minus = 15				: threshold of post-LTD
	theta_plus = 35					: threshold of post-LTP
	m_C1 = 1.7						: slope of the saturation function for C1
	m_G2 = 10						: slope of the saturation function for G2
	m_alpha = 1.5					: slope of the saturation function for K_alpha
	m_beta = 1.7					: slope of the saturation function for K_beta
	s_beta = 100					: scaling factor for calculation of K_beta
}

ASSIGNED {
	v (mV)
	i (nA)
	factor
	factor_G2
	C1
	C2
	G2
	E
	P
	K_alpha
	Rho
	K_beta
	K
	flag_g
}

STATE {
	g (uS)
	A (uS)
	B (uS)
	G2_a
	G2_b
	u_bar
	K_alpha_bar
	K_gamma
}

INITIAL {
	LOCAL tp, tp_G2
	g = 0
	u_bar = 0
	K_alpha_bar = 0
	K_gamma = 0
	flag_g = -1
	
	: Calculations taken from the original Exp2Syn
	: Actual EPSP
	if (tau_a/tau_b > .9999) {
		tau_a = .9999*tau_b
	}
	A = 0
	B = 0
	tp = (tau_a*tau_b)/(tau_b - tau_a) * log(tau_b/tau_a)
	factor = -exp(-tp/tau_a) + exp(-tp/tau_b)
	factor = 1/factor
	
	: Glutamate activation of NMDARs (G2)
	if (tau_G2_a/tau_G2_b > .9999) {
		tau_G2_a = .9999*tau_G2_b
	}
	G2_a = 0
	G2_b = 0
	tp_G2 = (tau_G2_a*tau_G2_b)/(tau_G2_b - tau_G2_a) * log(tau_G2_b/tau_G2_a)
	factor_G2 = -exp(-tp_G2/tau_G2_a) + exp(-tp_G2/tau_G2_b)
	factor_G2 = 1/factor_G2
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	G2 = sigmoid_sat(m_G2, G2_b - G2_a)		: representing glutamate activation of NMDARs
	g = g_max * (B - A)						: difference of exponentials for EPSP
	i = g * (v - e)							: At present, synaptic current only includes AMPAR-based EPSP
}

DERIVATIVE state {
	LOCAL G1, u, Eta, LTD_pre, LTD_post, LTP_post, g_update
	
	if(flag_g == 1) {	: Check if there is a presynaptic event
		G1 = 1
	    flag_g = -1
	}
	else {
	    G1 = 0
	}
	
	UNITSOFF
	u = v		: read local voltage
	Eta = dt	: learning rate
	UNITSON
	
	u_bar' = (- u_bar + positive(u - theta_u1)) / tau_u		: filtered corrected voltage
	C1 = sigmoid_sat(m_C1, u_bar)							: representing PLC-Ca2+
	E = G1 * C1												: representing eCBs
	
	C2 = G2 * positive(u - theta_u2)						: representing NMDAR coincidence signal
	
	P = positive(C2 - theta_minus) * positive(theta_plus - C2) / pow((theta_plus - theta_minus) / 2, 2) : representing phosphatase activation
	
	K_alpha = sigmoid_sat(m_alpha, positive(C2 - theta_plus)) * Rho	: representing initial kinase activation
	K_alpha_bar' = (- K_alpha_bar + K_alpha) / tau_alpha
	K_beta = sigmoid_sat(m_beta, (K_alpha_bar * s_beta))			: representing slow kinase process #1
	Rho = 1.0 - K_beta												: representing CaM limitation
	K_gamma' = (- K_gamma + K_beta) / tau_gamma						: representing slow kinase process #2
	K = K_alpha * K_beta * K_gamma									: representing kinase reaching LTP-state
	
	LTD_pre = - A_LTD_pre * E
	LTD_post = - A_LTD_post * P * Eta								: apply Eta to make values invariant to changes in dt
	LTP_post = A_LTP_post * K * Eta
	
	g_update = LTD_pre + LTD_post + LTP_post	: Weight update
	g_max = g_max + g_update
	
	A' = -A / tau_a
	B' = -B / tau_b
	G2_a' = -G2_a / tau_G2_a
	G2_b' = -G2_b / tau_G2_b
}

NET_RECEIVE(weight (uS)) {
	: printf("Received weight = %f at t = %f\n", weight, t)
	A = A + weight * factor
	B = B + weight * factor
	G2_a = G2_a + factor_G2
	G2_b = G2_b + factor_G2
	flag_g = 1
}

FUNCTION positive(value) {	: rectification function
	if(value < 0) {
		positive = 0
	}
	else {
		positive = value
	}
}

FUNCTION sigmoid_sat(slope, value) {	: sigmoidal saturation
	sigmoid_sat = 2.0 / (1.0 + pow(slope, -value)) - 1.0
}