import pyomo.common.unittest as unittest

from pyomo.core.base.constraint import Constraint
from pyomo.environ import *

gurobi_direct = SolverFactory('gurobi_direct')

class TestGurobiSigmoid(unittest.TestCase):
    def _basic_model(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(0, 4))
        m.y = Var(bounds=(0, 4))
        m.o = Objective(expr=m.y-m.x)
        m.c = Constraint(expr=1/(1+exp(-m.x)) <= m.y)
        return m

    @unittest.skipUnless(
        gurobi_direct.available(exception_flag=False) and gurobi_direct.license_is_valid(),
        "needs Gurobi Direct interface",
    )
    def test_gurobi_sigmoid(self):
        m = self._basic_model()
        gurobi_direct.solve(m, tee=True, options={'logfile':'gurobi.log'})
        self.assertAlmostEqual(m.o(), -3.017, delta=1e-3)