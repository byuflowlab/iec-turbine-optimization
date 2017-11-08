import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.api import pyOptSparseDriver
from ccblade import BlendedCCAirfoil, BlendedThickness, CCBlade as CCBlade_PY
from akima import akima_interp_with_derivs


class Spline(ExplicitComponent):

    def setup(self):
        # self.add_input('Rhub')
        # self.add_input('Rtip')
        # self.add_input('r2')
        self.add_input('rspline', shape=nspline)
        self.add_input('chordspline', shape=nspline)
        self.add_input('thetaspline', shape=nspline)
        self.add_input('r', shape=n)

        self.add_output('chord', shape=n)
        self.add_output('theta', shape=n)

        self.declare_partials('chord', 'chordspline')
        self.declare_partials('theta', 'thetaspline')

    def compute(self, inputs, outputs):

        # rspline = np.linspace(inputs['r2']*inputs['Rtip'], inputs['Rtip'], nspline-1)
        # rspline = np.concatenate([inputs['Rhub'], rspline])

        outputs['chord'], dchorddr, dchorddrpt, self.dchorddchordspline = akima_interp_with_derivs(
            inputs['rspline'], inputs['chordspline'], inputs['r'])

        outputs['theta'], dthetadr, dthetadrpt, self.dthetadthetaspline = akima_interp_with_derivs(
            inputs['rspline'], inputs['thetaspline'], inputs['r'])

    def compute_partials(self, inputs, J):
        J['chord', 'chordspline'] = self.dchorddchordspline
        J['theta', 'thetaspline'] = self.dthetadthetaspline
        # J['chord', 'r2'] = self.dchorddrpt[:, 1]*inputs['Rtip']


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
        # self.add_output("Q", shape=m)
        self.add_output("P", shape=m)
        self.add_output("Bfw", shape=m)

        self.declare_partials('T', ['chord', 'theta', 'blend', 'pitch'])
        self.declare_partials('P', ['chord', 'theta', 'blend', 'pitch'])
        self.declare_partials('Bfw', ['chord', 'theta', 'blend', 'pitch'])


    def compute(self, inputs, outputs):

        # setup ccblade model
        ccblade = CCBlade_PY(inputs["r"], inputs["chord"], inputs["theta"], af,
            inputs["blend"], inputs["Rhub"], inputs["Rtip"], inputs["B"], inputs["rho"], inputs["mu"],
            inputs["precone"], inputs["tilt"], inputs["yaw"], inputs["shearExp"], inputs["hubHt"],
            inputs["nSector"], tiploss=True, hubloss=True, wakerotation=True,
            usecd=True, derivatives=True)

        # evaluate power, thrust
        outputs["P"], outputs["T"], Q, outputs["Bfw"], self.dP, self.dT, self.dQ, self.dBfw = ccblade.evaluate(
            inputs["Uhub"], inputs["Omega"], inputs["pitch"], coefficient=False)


    def compute_partials(self, inputs, J):

        J['T', 'chord'] = self.dT['dchord']
        J['T', 'theta'] = self.dT['dtheta']
        J['T', 'blend'] = self.dT['dblend']
        J['T', 'pitch'] = self.dT['dpitch']

        J['P', 'chord'] = self.dP['dchord']
        J['P', 'theta'] = self.dP['dtheta']
        J['P', 'blend'] = self.dP['dblend']
        J['P', 'pitch'] = self.dP['dpitch']

        J['Bfw', 'chord'] = self.dBfw['dchord']
        J['Bfw', 'theta'] = self.dBfw['dtheta']
        J['Bfw', 'blend'] = self.dBfw['dblend']
        J['Bfw', 'pitch'] = self.dBfw['dpitch']



class ThicknessMargin(ExplicitComponent):

    def setup(self):
        self.add_input('chord', shape=n)
        self.add_input('blend', shape=n)
        self.add_input('tmin', shape=n)

        self.add_output('tmargin', shape=n)

        self.declare_partials('tmargin', ['chord', 'blend'])

    def compute(self, inputs, outputs):

        for i in range(n):
            outputs['tmargin'][i] = tau[i].thickness(inputs['chord'][i], inputs['blend'][i]) - inputs['tmin'][i]


    def compute_partials(self, inputs, J):

        dt_dchord = np.zeros(n)
        dt_dblend = np.zeros(n)
        for i in range(n):
            dt_dchord[i], dt_dblend[i] = tau[i].derivatives(inputs['chord'][i], inputs['blend'][i])

        J['tmargin', 'chord'] = np.diag(dt_dchord)
        J['tmargin', 'blend'] = np.diag(dt_dblend)




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

    # tmin
    tmin = initial['optimization']['Constraints']['Absolute_Thickness']['Lower_Limit']['Limit']
    tmin = tmin[1:-1]  # remove hub/tip

    # hub radius
    Rhub = initial['planform']['hub_radius']
    z = np.array(initial['planform']['Z'])  # no x,y for now so not worrying about coordinate transformation.
    r = Rhub + z[1:-1]  # chop off hub/tip (loads are zero)
    Rtip = Rhub + z[-1]

    # load other constraints
    Tfactor = initial['optimization']['Constraints']['Rotor_Thrust']['Upper_Limit']
    Bfwfactor = initial['optimization']['Constraints']['Root_Flap_Wise_Bending_Moment']['Upper_Limit']

    # create an input component
    ivc = IndepVarComp()
    ivc.add_output('r', r)
    # ivc.add_output('chord', initial['planform']['Chord'][1:-1])  # remove hub/tip
    # ivc.add_output('theta', initial['planform']['Twist'][1:-1])
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
    ivc.add_output('tmin', tmin)
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

    # pitch = np.zeros_like(V)  # some (but not all) of these need to be optimization variables
    pitch = 20*np.ones_like(V)  # some (but not all) of these need to be optimization variables

    ivc.add_output('Uhub', V)
    ivc.add_output('Omega', Omega)
    ivc.add_output('pitch', pitch)
    ivc.add_output('PDF', pdf)


    model = Group()
    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('ccblade', CCBlade(), promotes=['*'])
    model.add_subsystem('thickness', ThicknessMargin(), promotes=['*'])
    model.add_subsystem('aep', AEP(), promotes=['*'])

    prob = Problem(model)

    # # -------- power regulation optimization (just to set inital values) -----

    # # run optimization to determine pitch for base model
    # prob.driver = pyOptSparseDriver()
    # prob.driver.options['optimizer'] = "SNOPT"

    # # prob.driver.opt_settings['Major optimality tolerance'] = 1e-6

    # prob.model.add_design_var('pitch', lower=0.0, upper=20.0)
    # prob.model.add_objective('AEP', scaler=-1.0/1e7)  # maximize

    # # to add the constraint to the model
    # prob.model.add_constraint('P', lower=0.0, scaler=1.0/Prated, upper=Prated)

    # prob.setup()
    # prob.run_driver()


    # prob.setup()
    # # prob['pitch'] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.32391674, 3.14215928, 5.17330117, 6.88514161, 8.41286035, 11.13157035, 13.56485978, 18.92794969]
    # # # prob.check_partials(compact_print=True)
    # prob.run_model()

    # print prob['pitch']
    # print prob['AEP']
    # print repr(prob['P'])
    # print np.amax(prob['T'])
    # print np.amax(prob['Bfw'])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(r, prob['chord'])
    # plt.figure()
    # plt.plot(r, prob['theta'])
    # plt.figure()
    # plt.plot(r, prob['t'])
    # plt.plot(r, tmin)
    # plt.figure()
    # plt.plot(prob['Uhub'], prob['P'])
    # plt.plot([4.0, 25.0], [Prated, Prated])
    # plt.figure()
    # plt.plot(V, Omega)
    # plt.show()

    # set values from initial power regulation optimization

    pitch0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.32391674, 3.14215928, 5.17330117, 6.88514161, 8.41286035, 11.13157035, 13.56485978, 18.92794969]
    AEP0 = 29285836.91244333
    Tmax0 = 1257095.74514
    Bfw0 = 24506700.5989

    # spline parameterization
    nspline = 6
    rspline = np.linspace(Rhub, Rtip, nspline)
    ivc.add_output('rspline', rspline)  # starting point
    chordspline = np.interp(rspline, r, initial['planform']['Chord'][1:-1])
    thetaspline = np.interp(rspline, r, initial['planform']['Twist'][1:-1])
    ivc.add_output('chordspline', chordspline)  # starting point
    ivc.add_output('thetaspline', thetaspline)  # starting point

    model = Group()
    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('spline', Spline(), promotes=['*'])
    model.add_subsystem('ccblade', CCBlade(), promotes=['*'])
    model.add_subsystem('thickness', ThicknessMargin(), promotes=['*'])
    model.add_subsystem('aep', AEP(), promotes=['*'])

    prob = Problem(model)

    # # prob.setup()
    # # prob.check_partials(compact_print=True)

    # # ------------ run actual optmization -----

    # prob.driver = pyOptSparseDriver()
    # prob.driver.options['optimizer'] = "SNOPT"

    # # prob.driver.opt_settings['Major optimality tolerance'] = 1e-6

    # # prob.model.add_design_var('chord', lower=0.0, upper=10.0)
    # # prob.model.add_design_var('theta', lower=-10, upper=30.0)
    # prob.model.add_design_var('chordspline', lower=0.0, upper=10.0)
    # prob.model.add_design_var('thetaspline', lower=0.0, upper=30.0)
    # prob.model.add_design_var('blend', lower=0.0, upper=1.0)
    # prob.model.add_design_var('pitch', lower=0.0, upper=30.0)

    # prob.model.add_objective('AEP', scaler=-1.0/AEP0)  # maximize

    # # to add the constraint to the model
    # prob.model.add_constraint('P', lower=0.0, scaler=1.0/Prated, upper=Prated)
    # prob.model.add_constraint('T', lower=0.0, scaler=1.0/Tmax0, upper=Tmax0*Tfactor)
    # prob.model.add_constraint('Bfw', lower=0.0, scaler=1.0/Bfw0, upper=Bfw0*Bfwfactor)
    # prob.model.add_constraint('tmargin', lower=0.0, upper=10.0)

    # prob.setup()
    # prob['pitch'] = pitch0
    # # prob.run_model()
    # # prob.check_totals()
    # prob.run_driver()

    # print repr(prob['chordspline'])
    # print repr(prob['thetaspline'])
    # # print repr(prob['chord'])
    # # print repr(prob['theta'])
    # print repr(prob['blend'])
    # print repr(prob['pitch'])
    # print repr(prob['AEP'])

    # # ------ print results ----

    chordspline = np.array([ 7.72904009,  9.66650332,  4.642612  ,  3.75945005,  2.8982931 ,
        1.87423212])
    thetaspline = np.array([ 16.85396088,  12.10532789,   5.84643576,   1.60971854,
         0.        ,   0.        ])
    blend = np.array([ 0.07844398,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.03376377,  0.        ,  0.        ,  0.35197787,  0.06817927,
        0.        ,  0.1009294 ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ])
    pitch = np.array([  2.59660269,   1.9575017 ,   1.0509937 ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.87327895,   2.45057508,   4.35385999,
         6.81823811,   8.71234371,  10.34547414,  11.82074766,
        14.47373294,  16.87127535,  22.21422761])
    AEPopt = 32247377.56358137

    # # # chord = np.array([ 8.93833333,  8.54816667,  7.81966667,  6.938     ,  5.97333333,
    # # #     5.09516667,  8.8548126 ,  7.99550941,  5.95666667,  5.42055556,
    # # #     4.96194444,  4.55583333,  4.15444444,  4.49634551,  4.1354921 ,
    # # #     4.29336588,  4.34184261,  4.31958399,  4.28652301,  4.24613446,
    # # #     4.20851552,  4.15745326,  4.10082724,  4.03664055,  3.96594352,
    # # #     3.88826966,  3.78597026,  3.67576357,  3.54947714,  3.40284257,
    # # #     3.23473786,  3.03996166,  2.80107282,  2.54414293,  2.26877459,
    # # #     1.93628802,  1.54887319,  1.87872384])
    # # # theta = np.array([ 29.59961895,  20.60267425,  14.07574485,   8.39071291,
    # # #      4.59898245,   1.82387254,  12.88868913,  10.93396311,
    # # #      9.82700858,   8.61454412,   7.57391329,   6.6666137 ,
    # # #      5.79337362,   4.13698913,   3.43315284,   3.63611537,
    # # #      3.55495721,   3.47124253,   3.3807371 ,   3.28433599,
    # # #      3.19708406,   3.09954296,   3.00175352,   2.89824626,
    # # #      2.79710415,   2.69646859,   2.57279114,   2.4546052 ,
    # # #      2.32859262,   2.1963466 ,   2.05580362,   1.89992937,
    # # #      1.68740727,   1.48524427,   1.30470774,   1.0491293 ,
    # # #      0.73660942,  -2.16956379])
    # # # blend = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    # # #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    # # #     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    # # # pitch = np.array([  0.95132649,   0.67035572,   0.12989032,   0.        ,
    # # #      0.        ,   0.        ,   0.        ,   0.        ,
    # # #      0.        ,   0.37071199,   1.49137906,   3.22475018,
    # # #      5.50216933,   7.26500237,   8.78991032,  10.17124843,
    # # #     12.6620359 ,  14.92132383,  20.        ])
    # # # AEPopt = 32022268.73061096

    prob.setup()
    # prob['chord'] = chord
    prob['chordspline'] = chordspline
    prob['thetaspline'] = thetaspline
    prob['blend'] = blend
    prob['pitch'] = pitch
    prob.run_model()

    # ---- plot -----

    print prob['AEP']/AEP0
    print prob['AEP']
    print np.amax(prob['T'])/Tmax0
    print np.amax(prob['Bfw'])/Bfw0
    # print repr(Omega)
    print np.amax(prob['T'])
    print np.amax(prob['Bfw'])
    print "chord =", repr(prob['chord'])
    print "theta =", repr(prob['theta'])
    print "P = ", repr(prob['P'])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(r, prob['chord'])
    plt.figure()
    plt.plot(r, prob['theta'])
    plt.figure()
    plt.plot(r, prob['tmargin'] + prob['tmin'])
    plt.plot(r, tmin)
    plt.figure()
    plt.plot(prob['Uhub'], prob['P'])
    plt.plot([4.0, 25.0], [Prated, Prated])
    plt.show()
