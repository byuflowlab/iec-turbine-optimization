import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from ccblade import BlendedCCAirfoil, BlendedThickness, CCBlade as CCBlade_PY


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




# class WeibullCDF(ExplicitComponent):
#     """Weibull cumulative distribution function"""

#     def setup(self):
#         self.add_input('V', shape=m)
#         self.add_input('A', desc='scale factor')
#         self.add_input('k', desc='shape or form factor')

#         self.add_output('F', shape=m)

#         # self.declare_partials('F', 'V')

#     def compute(self, inputs, outputs):

#         outputs['F'] = 1.0 - np.exp(-(inputs['V']/inputs['A'])**inputs['k'])




class Thickness(ExplicitComponent):

    def setup(self):
        self.add_input('chord', shape=n)
        self.add_input('blend', shape=n)

        self.add_output('t', shape=n)

        self.declare_partials('t', ['chord', 'blend'])

    def compute(self, inputs, outputs):

        for i in range(n):
            outputs['t'][i] = tau[i].thickness(inputs['chord'][i], inputs['blend'][i])


    def compute_partials(self, inputs, J):

        dt_dchord = np.zeros(n)
        dt_dblend = np.zeros(n)
        for i in range(n):
            dt_dchord[i], dt_dblend[i] = tau[i].derivatives(inputs['chord'][i], inputs['blend'][i])

        J['t', 'chord'] = np.diag(dt_dchord)
        J['t', 'blend'] = np.diag(dt_dblend)




class AEP(ExplicitComponent):
    """integrate to find annual energy production"""

    def setup(self):
        # inputs
        self.add_input('PDF', shape=m)
        self.add_input('P', shape=m, desc='power curve (power)')  # units='W',
        # lossFactor = Float(iotype='in', desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', desc='annual energy production')  # units='kW*h',

        self.declare_partials('AEP', 'P')

    def compute(self, inputs, outputs):

        # outputs['AEP'] = np.trapz(inputs['P'], inputs['CDF_V'])/1e3*365.0*24.0  # in kWh
        outputs['AEP'] = np.sum(inputs['P']*inputs['PDF'])/1e3*365.0*24.0  # in kWh

    def compute_partials(self, inputs, J):

        J['AEP', 'P'] = inputs['PDF']/1e3*365.0*24.0




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
    afnames = ['Cylinder', 'FFA_W3_600', 'FFA_W3_480', 'FFA_W3_360', 'FFA_W3_301', 'FFA_W3_241']  # NOTE: add to yaml
    naftypes = len(afnames)

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
    blendingweight = blendingweight[1:-1]
    af = af[1:-1]

    # grab t/c for each airfoil
    tau_each = np.zeros(naftypes)
    for i in range(naftypes):
        tau_each[i] = initial['airfoils'][afnames[i]]['Relative_Thickness']

    tau = [0]*n
    for i in range(n):
        tau[i] = BlendedThickness(tau_each[afidx[i]], tau_each[afidx[i]+1])
    tau = tau[1:-1]

    # tmax
    tmax = initial['optimization']['Constraints']['Absolute_Thickness']['Lower_Limit']['Limit']
    tmax = tmax[1:-1]  # remove hub/tip

    # hub radius
    Rhub = initial['planform']['hub_radius']
    z = np.array(initial['planform']['Z'])  # no x,y for now so not worrying about coordinate transformation.
    r = Rhub + z[1:-1]  # chop off hub/tip (loads are zero)
    Rtip = Rhub + z[-1]

    # create a group
    model = Group()
    ivc = IndepVarComp()
    ivc.add_output('r', r)
    ivc.add_output('chord', initial['planform']['Chord'][1:-1])  # remove hub/tip
    ivc.add_output('theta', initial['planform']['Twist'][1:-1])
    ivc.add_output('Rhub', Rhub)
    ivc.add_output('Rtip', Rtip)
    ivc.add_output('hubHt', 0.0)  # irrelevant if no shear
    ivc.add_output('precone', 0.0)
    ivc.add_output('tilt', 0.0)
    ivc.add_output('yaw', 0.0)
    ivc.add_output('B', 3)  # never explicitly stated
    ivc.add_output('rho', initial['environment']['Air_Density'])
    ivc.add_output('mu', 0.0000181206)  # irrelevant if no Re dependency
    ivc.add_output('shearExp', 0.0)
    ivc.add_output('nSector', 1)
    ivc.add_output('blend', blendingweight)
    n -= 2  # removed hub/tip

    # setup operational parameters
    # NOTE: add to yaml (currently in Table 16):
    V = np.array([4, 5, 6, 7, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 18, 20, 25])
    pdf = np.array([0.09703371, 0.105419021, 0.106573726, 0.101533754, 0.091850061, 0.060731811, 0.036232231, 0.032753105, 0.029304348, 0.025955836, 0.022764024, 0.028607044, 0.029058315, 0.020545317, 0.014008677, 0.012489291, 0.007386119, 0.003183651, 0.000328316])
    m = len(V)
    tsr = initial['control']['Design_Tip_Speed_Ratio']
    Omegamin = initial['control']['Minimum_Rotational_Rate']  # RPM
    Omegamax = initial['control']['Maximum_Rotational_Rate']  # RPM
    RS2RPM = 30.0/np.pi
    Omega = tsr*V/Rtip*RS2RPM
    Omega = np.minimum(Omega, Omegamax)
    Omega = np.maximum(Omega, Omegamin)
    Prated = initial['control']['Maximum_Mechanical_Power']

    pitch = np.zeros_like(V)  # some (but not all) of these need to be optimization variables

    ivc.add_output('Uhub', V)
    ivc.add_output('Omega', Omega)
    ivc.add_output('pitch', pitch)
    ivc.add_output('PDF', pdf)

    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('ccblade', CCBlade(), promotes=['*'])
    model.add_subsystem('thickness', Thickness(), promotes=['*'])
    model.add_subsystem('aep', AEP(), promotes=['*'])

    prob = Problem(model)
    prob.setup()
    prob.run_model()

    print prob['AEP']
    print prob['T']

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(r, prob['chord'])
    plt.figure()
    plt.plot(r, prob['t'])
    plt.plot(r, tmax)
    plt.figure()
    plt.plot(prob['Uhub'], prob['P'])
    plt.plot([4.0, 25.0], [Prated, Prated])
    plt.figure()
    plt.plot(V, Omega)
    plt.show()
