import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

from ccblade import CCAirfoil, CCBlade as CCBlade_PY


class CCBlade(ExplicitComponent):

    def setup(self):

        self.add_input('r', shape=n)
        self.add_input('chord', shape=n)
        self.add_input('theta', shape=n)
        self.add_input('Rhub')
        self.add_input('Rtip')
        self.add_input('hubHt')
        self.add_input('precone')
        self.add_input('tilt')
        self.add_input('yaw')

        self.add_input("airfoil_files", shape=n)
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

        # airfoil files
        n = len(inputs["airfoil_files"])
        af = [0]*n
        for i in range(n):
            af[i] = CCAirfoil.initFromAerodynFile(inputs["airfoil_files"][i])

        # setup ccblade model
        # TODO: no hub info was provided
        ccblade = CCBlade_PY(inputs["r"], inputs["chord"], inputs["theta"], af,
            inputs["Rhub"], inputs["Rtip"], inputs["B"], inputs["rho"], inputs["mu"],
            inputs["precone"], inputs["tilt"], inputs["yaw"], inputs["shearExp"], inputs["hubHt"],
            inputs["nSector"], tiploss=True, hubloss=False, wakerotation=True,
            usecd=True, derivatives=True)

        # evaluate power, thrust
        outputs["P"], outputs["T"], outputs["Q"], dP, dT, dQ = ccblade.evaluate(
            inputs["Uhub"], inputs["Omega"], inputs["pitch"], coefficient=False)



if __name__ == '__main__':
    import yaml
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.core.indepvarcomp import IndepVarComp

    # load yaml file
    stream = file("initial_design.yaml", "r")
    initial = yaml.load(stream)

    n = len(initial['planform']['SParam'])

    afnames = ['Cylinder', 'FFA_W3_600', 'FFA_W3_480', 'FFA_W3_360', 'FFA_W3_301', 'FFA_W3_241']

    # load all the airfoil aerodynamic data
    naftypes = len(afnames)
    afdata = naftypes*[0]
    for i in range(naftypes):
        a = initial['airfoils'][afnames[i]]
        afdata[i] = CCAirfoil(a['Angle_Of_Attack'], [1.0], a['Cl'], a['Cd'])  # no Re dependency

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
    blendingweight = np.zeros(n)
    for i in range(n):
        blendingweight[i] = blendingall[afidx[i], i]



    # # create a group
    # model = Group()
    # ivc = IndepVarComp()
    # ivc.add_output('r', initial['planform']['Z'][:-1])
    # ivc.add_output('chord', initial['planform']['Chord'][:-1])
    # ivc.add_output('theta', initial['planform']['Twist'][:-1])
    # ivc.add_output('Rhub', 0.0)
    # ivc.add_output('Rtip', initial['planform']['Z'][-1])
    # ivc.add_output('hubHt', 0.0)  # irrelevant if no shear
    # ivc.add_output('precone', 0.0)
    # ivc.add_output('tilt', 0.0)
    # ivc.add_output('yaw', 0.0)
    # ivc.add_output('B', 3)
    # ivc.add_output('rho', initial['environment']['Air_Density'])
    # ivc.add_output('mu', 0.0000181206)  # irrelevant if no Re dependency
    # ivc.add_output('shearExp', 0.0)
    # ivc.add_output('nSector', 1)
    # ivc.add_output('airfoil_files', )


    # model.add_subsystem('inputs', ivc)
    # model.add_subsystem('ccblade', CCBlade())

    # prob = Problem(model)
    # prob.setup()
    # prob.run_model()