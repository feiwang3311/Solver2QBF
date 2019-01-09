from typing import Dict, NamedTuple, List, Tuple, NewType, Optional, Set
from itertools import groupby
from satparser import *
import glob
import subprocess

class SharpSAT(object):
    def __init__(self):
        self.models = []
        self.nVars = 0
        self.iclauses = []

    def new_var(self):
        self.nVars += 1

    def nvars(self):
        return self.nVars

    def add_clause(self, clause):
        self.iclauses.append(Clause(clause))

    def apply_backtrack(self, s: State) -> State:
        v, f, asn = s.cont[0]
        return State(f.assign(v, False), (-v,)+asn, s.cont[1:])

    def apply_unit(self, s: State) -> State:
        f, asn, cont = s
        new_f, new_asn = f.elimUnit()
        return State(new_f, new_asn + asn, cont)

    def apply_pure(self, s: State) -> State:
        f, asn, cont = s
        return State(f.addUnit(f.pureVars[0]), asn, cont)

    def dpll_step(self, s: State) -> State:
        f, asn, cont = s
        if f.hasUnsat():  return self.apply_backtrack(s)
        elif f.hasUnit(): return self.apply_unit(s)
        #elif f.hasPure(): return apply_pure(s)
        else:
            v = f.pick()
            return State(f.assign(v, True), (v,)+asn, (Cont(v, f, asn),)+cont)

    def drive(self, s: State) -> Optional[Asn]:
        f, asn, cont = s
        if f.isEmpty():
            self.models.append(asn)
            if len(cont) == 0 or len(self.models) > 3: return None
            return self.drive(self.apply_backtrack(s))
        elif len(cont) == 0 and f.hasUnsat(): return None
        else: return self.drive(self.dpll_step(s))

    def inject(self, f: Formula) -> State: return State(f, (), ())

    def solve(self, f: Formula = None) -> State:
        self.models = []
        if f is not None:
            nVars = f.nVars
            self.drive(self.inject(f))
        else:
            nVars = self.nVars
            ff = Formula(self.iclauses)
            self.drive(self.inject(ff))
        # process self.models
        nums = [2**(nVars-len(m)) for m in self.models]
        return sum(nums) > 0

    def get_model(self, end=-1):
        # this function only works if solve was done with None parameter
        m = sorted(list(self.models[0]), key=abs)
        model = []
        index = 0
        for i in m:
            while abs(i) > index + 1:
                model.append(0)
                index += 1
            model.append(int(i > 0))
            index += 1
        while self.nVars > index:
            model.append(0)
            index += 1
        if end < 0:
            return model
        else:
            return model[:end]


def run_sharp_SAT(filename, verbose=False):
    result = subprocess.run(['/homes/wang603/sharpSAT/build/Release/sharpSAT', filename], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8').split('\n')
    try:
        idx1 = result.index('# solutions ')
        idx2 = result.index('# END')
        assert idx2 == idx1 + 2
        return int(result[idx1 + 1])
    except ValueError:
        print(result)

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    
    sats = glob.glob('/homes/wang603/sharpSAT/build/Release/sats/*.cnf')
    for filename in sats[:1]:
        print(filename)
        solver = SharpSAT()
        formula = parse_dimacs(filename)
        print(solver.solve(formula))
        print(solver.get_model())
        # verify the correctness of number of solutions
        print('sharpSolver gives {}'.format(run_sharp_SAT(filename)))

