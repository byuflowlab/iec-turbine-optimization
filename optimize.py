import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from ccblade import BlendedCCAirfoil, CCBlade as CCBlade_PY


class CCBlade(ExplicitComponent):

    def setup(self):

        self.add_input('r', shape=n)
        self.add_input('chord', shape=n)
        self.add_input('theta', shape=n)
        self.add_input('blend', shape=n)
        self.add_input('Rhub')
        self.add_input('Rtip')
        self.add_input('hubHt')
        self.add_input('precone')
        self.add_input('tilt')
        self.add_input('yaw')

        # self.add_input("af", shape=n)
        self.add_input("B")
        self.add_input("rho")
        self.add_input("mu")
        self.add_input("shearExp")
        self.add_input("nSector")

        self.add_input("Uhub", shape=m)
        self.add_input("Omega", shape=m)
        self.add_input("pitch", shape=m)

        self.add_output("T", shape=m)
        self.add_output("Q", shape=m)
        self.add_output("P", shape=m)

        self.declare_partials('*', '*', method='fd')  # FIXME


    def compute(self, inputs, outputs):

        # setup ccblade model
        # TODO: no hub info was provided.  hub loss and tip loss turned off for now.
        ccblade = CCBlade_PY(inputs["r"], inputs["chord"], inputs["theta"], af,
            inputs["blend"], inputs["Rhub"], inputs["Rtip"], inputs["B"], inputs["rho"], inputs["mu"],
            inputs["precone"], inputs["tilt"], inputs["yaw"], inputs["shearExp"], inputs["hubHt"],
            inputs["nSector"], tiploss=True, hubloss=True, wakerotation=True,
            usecd=True, derivatives=True)

        # evaluate power, thrust
        outputs["P"], outputs["T"], outputs["Q"], dP, dT, dQ = ccblade.evaluate(
            inputs["Uhub"], inputs["Omega"], inputs["pitch"], coefficient=False)


class UnregulatedPowerCurve(ExplicitComponent):

    def setup(self):

        self.add_input('V', shape=m)
        self.add_input('Omega_min')  # RPM
        self.add_input('Omega_max')
        self.add_input('tsropt')
        self.add_input('Rtip')

        self.add_output('Omega', shape=m)  # RPM

        # no derivatives needed

    def compute(self, inputs, outputs):

        V = inputs['V']
        tsr = inputs['tsropt']
        Rtip = inputs['Rtip']

        RS2RPM = 30.0/np.pi
        Omega_d = tsr*V/Rtip*RS2RPM
        Omega = np.minimum(Omega_d, inputs['Omega_max'])
        Omega = np.maximum(Omega_d, inputs['Omega_min'])

        outputs['Omega'] = Omega




class WeibullCDF(ExplicitComponent):
    """Weibull cumulative distribution function"""

    def setup(self):
        self.add_input('V', shape=m)
        self.add_input('A', desc='scale factor')
        self.add_input('k', desc='shape or form factor')

        self.add_output('F', shape=m)

        # self.declare_partials('F', 'V')

    def compute(self, inputs, outputs):

        outputs['F'] = 1.0 - np.exp(-(inputs['V']/inputs['A'])**inputs['k'])



def trapz_deriv(y, x):
    """trapezoidal integration and derivatives with respect to integrand or variable."""

    dI_dy = np.gradient(x)
    dI_dy[0] /= 2
    dI_dy[-1] /= 2

    dI_dx = -np.gradient(y)
    dI_dx[0] = -0.5*(y[0] + y[1])
    dI_dx[-1] = 0.5*(y[-1] + y[-2])

    return dI_dy, dI_dx


class AEP(ExplicitComponent):
    """integrate to find annual energy production"""

    def setup(self):
        # inputs
        self.add_input('CDF_V', shape=m)
        self.add_input('P', shape=m, desc='power curve (power)')  # units='W',
        # lossFactor = Float(iotype='in', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', desc='annual energy production')  # units='kW*h',

        self.declare_partials('AEP', 'P')

    def compute(self, inputs, outputs):

        outputs['AEP'] = np.trapz(inputs['P'], inputs['CDF_V'])/1e3*365.0*24.0  # in kWh

    def compute_partials(self, inputs, J):

        factor = 1.0/1e3*365.0*24.0

        dAEP_dP, dAEP_dCDF = trapz_deriv(inputs['P'], inputs['CDF_V'])
        dAEP_dP *= factor

        J['AEP', 'P'] = dAEP_dP




if __name__ == '__main__':
    import yaml
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.core.indepvarcomp import IndepVarComp

    # load yaml file
    stream = file("initial_design.yaml", "r")
    initial = yaml.load(stream)

    # number of stations on blade
    n = len(initial['planform']['SParam'])

    # airfoil names
    afnames = ['Cylinder', 'FFA_W3_600', 'FFA_W3_480', 'FFA_W3_360', 'FFA_W3_301', 'FFA_W3_241']
    naftypes = len(afnames)

    # load all the airfoil aerodynamic data
    # afdata = naftypes*[0]
    # for i in range(naftypes):
    #     a = initial['airfoils'][afnames[i]]
    #     afdata[i] = CCAirfoil(a['Angle_Of_Attack'], [1.0], a['Cl'], a['Cd'])  # no Re dependency

    # check that all angles of attack are the same: (they are)
    # for i in range(naftypes-1):
    #     print np.array(initial['airfoils'][afnames[i+1]]['Angle_Of_Attack']) - np.array(initial['airfoils'][afnames[i]]['Angle_Of_Attack'])

    # load all blending coefficients
    blendingall = np.zeros((naftypes, n))
    for i in range(naftypes):
        blendingall[i, :] = initial['planform']['Airfoil_Coefficient_Blending_Weights'][afnames[i]]

    # reparameterize blending data in terms of 0-1
    afidx = np.zeros(n, dtype='int_')  # idx of starting airfoil (blend of two airfoils)
    for i in range(n):
        for j in range(naftypes):
            if blendingall[j, i] != 0.0:
                if j == naftypes-1:  # last one
                    afidx[i] = j-1
                else:
                    afidx[i] = j
                break

    # compute initial blending weights (will be a design var)
    # and set airfoils
    blendingweight = np.zeros(n)
    af = [0]*n
    for i in range(n):
        blendingweight[i] = blendingall[afidx[i], i]
        af1 = initial['airfoils'][afnames[afidx[i]]]
        af2 = initial['airfoils'][afnames[afidx[i]+1]]
        # no Re dependency
        af[i] = BlendedCCAirfoil(af1['Angle_Of_Attack'], [1.0], af1['Cl'], af1['Cd'],
            af2['Angle_Of_Attack'], [1.0], af2['Cl'], af2['Cd'])



    # create a group
    model = Group()
    ivc = IndepVarComp()
    ivc.add_output('r', initial['planform']['Z'][1:-1])  # chop off hub/tip
    ivc.add_output('chord', initial['planform']['Chord'][1:-1])
    ivc.add_output('theta', initial['planform']['Twist'][1:-1])
    ivc.add_output('Rhub', 0.0)
    ivc.add_output('Rtip', initial['planform']['Z'][-1])
    ivc.add_output('hubHt', 0.0)  # irrelevant if no shear
    ivc.add_output('precone', 0.0)
    ivc.add_output('tilt', 0.0)
    ivc.add_output('yaw', 0.0)
    ivc.add_output('B', 3)  # never explicitly stated
    ivc.add_output('rho', initial['environment']['Air_Density'])
    ivc.add_output('mu', 0.0000181206)  # irrelevant if no Re dependency
    ivc.add_output('shearExp', 0.0)
    ivc.add_output('nSector', 1)
    ivc.add_output('blend', blendingweight[1:-1])
    af = af[1:-1]
    n -= 2  # removed hub/tip

    tsr = np.linspace(1.0, 15.0, 30)
    Omega = 10.0*np.ones_like(tsr)
    rho = initial['environment']['Air_Density']
    Rtip = initial['planform']['Z'][-1]
    Uhub = Omega*np.pi/30.0*Rtip/tsr
    pitch = np.zeros_like(tsr)

    # Uhub = [7.0, 8.0]
    # Omega = [10.0, 10.0]
    # pitch = [0.0, 0.0]
    m = len(Uhub)

    ivc.add_output('Uhub', Uhub)
    ivc.add_output('Omega', Omega)
    ivc.add_output('pitch', pitch)


    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('ccblade', CCBlade(), promotes=['*'])

    ivc2 = IndepVarComp()
    V = np.arange(4, 25)
    m = len(V)
    A = 8.0
    k = 2.0
    print V
    ivc2.add_output('V', V)
    ivc2.add_output('A', A)
    ivc2.add_output('k', k)
    model2 = Group()
    model2.add_subsystem('inputs', ivc2, promotes=['*'])
    model2.add_subsystem('weibull', WeibullCDF(), promotes=['*'])

    prob = Problem(model2)
    prob.setup()
    prob.run_model()

    print prob['F']

    # print prob['T']
    # print prob['Q']
    # print prob['P']

    # CP = prob['P']/(0.5*rho*Uhub**3 * np.pi*Rtip**2)

    # import matplotlib.pyplot as plt
    # plt.plot(tsr, CP)
    # plt.show()
