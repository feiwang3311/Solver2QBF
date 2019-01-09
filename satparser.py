from typing import Dict, NamedTuple, List, Tuple, NewType, Optional, Set
from itertools import groupby
import sys
import random

Lit = NewType('Lit', int)
Asn = Tuple[Lit, ...]

class Clause():
    def __init__(self, xs: List[Lit]):
        self.xs = xs
    def __len__(self): return len(self.xs)
    def __str__(self): return str(self.xs)
    __repr__ = __str__
    def __getitem__(self, i): return self.xs[i]
#    def shuffle(self):
#        xs = self.xs.copy()
#        random.shuffle(xs)
#        return Clause(xs)
    def contains(self, x): return x in self.xs
    def remove(self, x) -> 'Clause': return Clause([y for y in self.xs if x != y])
    def assign(self, v: Lit, b: bool) -> Optional['Clause']:
        new_xs = []
        for x in self.xs:
            if abs(x) == abs(v):
                if (x > 0) == b: return None
            else: new_xs.append(x)
        return Clause(new_xs)

class Formula():
    def __init__(self, cs: List[Clause]):
        self.cs = cs
        self.allVars = list(set([item for sub in [c.xs for c in cs] for item in sub]))
        pureVarsGrp = [list(g) for k, g in groupby(sorted(self.allVars, key=abs), abs)]
        self.nVars = len(pureVarsGrp)
        self.pureVars = [g[0] for g in pureVarsGrp if len(g) == 1]
        self.unitVars = [c[0] for c in cs if len(c) == 1]
    def __str__(self): return str(self.cs)
    __repr__ = __str__
    def pick(self): return self.cs[0][0]
    def isEmpty(self) -> bool: return len(self.cs) == 0
    def assign(self, v: Lit, b: bool) -> 'Formula':
        assigned = [c.assign(v, b) for c in self.cs]
        assigned = [c for c in assigned if c != None]
        return Formula(assigned)
    def elimUnit(self, v = None) -> Tuple['Formula', Asn]:
        if v == None: v = self.unitVars[0]
        return (Formula([c.remove(-v) for c in self.cs if not c.contains(v)]), (v,))
    def addUnit(self, v: Lit) -> 'Formula':
        return Formula([Clause([v])] + self.cs.copy())
    def hasUnsat(self) -> bool:
        for c in self.cs: 
            if len(c) == 0: return True
        return False
    def hasUnit(self) -> bool: return len(self.unitVars) != 0
    def hasPure(self) -> bool: return len(self.pureVars) != 0
#    def shuffle(self) -> 'Formula':
#        cs = [c.shuffle() for c in self.cs]
#        random.shuffle(cs)
#        return Formula(cs)

Cont = NamedTuple('Cont', [('var', Lit),
                           ('formula', Formula),
                           ('assignment', Asn)])
State = NamedTuple('State', [('formula', Formula),
                             ('assignment', Asn),
                             ('cont', Tuple[Cont, ...])])

# def shuffle_state(self) -> 'State': return State(self.formula.shuffle(), self.assignment, self.cont)

def parse_line(line) -> Clause: return Clause([int(x) for x in line.split(" ")][:-1])

def parse_dimacs(filename: str) -> Formula:
    def valid(l): return not (len(l)==0 or l[0]=='c' or l[0]=='p' or l[0]=='0' or l[0]=='%')
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [parse_line(l) for l in lines if valid(l)]
        return Formula(lines)

if __name__ == '__main__':
    print(parse_dimacs(sys.argv[1]))
