import pyomo.common.unittest as unittest

from pyomo.core.base.constraint import Constraint
from pyomo.environ import *

gurobi_direct = SolverFactory('gurobi_direct')

class TestBasicNLModel(unittest.TestCase):
    def _basic_model(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(-1, 4))
        m.o = Objective(expr=sin(m.x) + cos(2*m.x) + 1)
        m.c = Constraint(expr=0.25 * exp(m.x) - m.x <= 0)
        return m

    @unittest.skipUnless(
        gurobi_direct.available(exception_flag=False) and gurobi_direct.license_is_valid(),
        "needs Gurobi Direct interface",
    )
    def test_nl_basic_gurobi(self):
        m = self._basic_model()
        results = gurobi_direct.solve(m)
        self.assertEqual(m.o(), results['Problem'][0]['Lower bound'])