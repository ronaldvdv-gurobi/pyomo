# Based on https://docs.gurobi.com/projects/examples/en/current/examples/python/acopf_4buses.html

from math import pi
import pyomo.common.unittest as unittest

from pyomo.core.base.constraint import Constraint
from pyomo.environ import *

gurobi_direct = SolverFactory('gurobi_direct')

class TestBasicNLModel(unittest.TestCase):
    @unittest.skipUnless(
        gurobi_direct.available(exception_flag=False) and gurobi_direct.license_is_valid(),
        "needs Gurobi Direct interface",
    )
    def test_nl_basic_gurobi(self):
        
        # Number of Buses (Nodes)
        N = 4

        # Conductance/susceptance components
        G = [
                [1.7647, -0.5882, 0.0, -1.1765],
                [-0.5882, 1.5611, -0.3846, -0.5882],
                [0.0, -0.3846, 1.5611, -1.1765],
                [-1.1765, -0.5882, -1.1765, 2.9412],
            ]
        B = [
                [-7.0588, 2.3529, 0.0, 4.7059],
                [2.3529, -6.629, 1.9231, 2.3529],
                [0.0, 1.9231, -6.629, 4.7059],
                [4.7059, 2.3529, 4.7059, -11.7647],
            ]        

        # Assign bounds where fixings are needed
        v_lb = [1.0, 0.0, 1.0, 0.0]
        v_ub = [1.0, 1.5, 1.0, 1.5]
        P_lb = [-3.0, -0.3, 0.3, -0.2]
        P_ub = [3.0, -0.3, 0.3, -0.2]
        Q_lb = [-3.0, -0.2, -3.0, -0.15]
        Q_ub = [3.0, -0.2, 3.0, -0.15]
        theta_lb = [0.0, -pi / 2, -pi / 2, -pi / 2]
        theta_ub = [0.0, pi / 2, pi / 2, pi / 2]

        exp_v = [1.0, 0.949, 1.0, 0.973]
        exp_theta = [0.0, -2.176, 1.046, -0.768]
        exp_P = [0.2083, -0.3, 0.3, -0.2]
        exp_Q = [0.212, -0.2, 0.173, -0.15]

        m = ConcreteModel(name="acopf")
        
        m.P = VarList()
        m.Q = VarList()
        m.v = VarList()
        m.theta = VarList()

        for i in range(N):
            p = m.P.add()
            p.lb = P_lb[i]
            p.ub = P_ub[i]
            
            q = m.Q.add()
            q.lb = Q_lb[i]
            q.ub = Q_ub[i]

            v = m.v.add()
            v.lb = v_lb[i]
            v.ub = v_ub[i]

            theta = m.theta.add()
            theta.lb = theta_lb[i]
            theta.ub = theta_ub[i]

        m.obj = Objective(expr=m.Q[1] + m.Q[3])

        m.define_P = ConstraintList()
        m.define_Q = ConstraintList()
        for i in range(N):
            m.define_P.add(m.P[i+1] == m.v[i+1] * sum(m.v[j+1] * (G[i][j] * cos(m.theta[i+1] - m.theta[j+1]) + B[i][j] * sin(m.theta[i+1] - m.theta[j+1])) for j in range(N)))
            m.define_Q.add(m.Q[i+1] == m.v[i+1] * sum(m.v[j+1] * (G[i][j] * sin(m.theta[i+1] - m.theta[j+1]) - B[i][j] * cos(m.theta[i+1] - m.theta[j+1])) for j in range(N)))
        
        results = gurobi_direct.solve(m, tee=True)
        self.assertEqual(SolverStatus.ok, results.solver.status)
        for i in range(N):
            self.assertAlmostEqual(exp_P[i], m.P[i+1].value, delta=1e-3, msg=f'P[{i}]')
            self.assertAlmostEqual(exp_Q[i], m.Q[i+1].value, delta=1e-3, msg=f'Q[{i}]')
            self.assertAlmostEqual(exp_v[i], m.v[i+1].value, delta=1e-3, msg=f'v[{i}]')
            self.assertAlmostEqual(exp_theta[i], m.theta[i+1].value * 180 / pi, delta=1e-3, msg=f'theta[{i}]')