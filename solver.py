import sys
sys.path.insert(0, '/homes/wang603/')
from PyMiniSolvers import minisolvers
import os

import math
import numpy as np
import random
import datetime
import subprocess
import pickle
import sys
import argparse
from options import add_neurosat_options
from neurosat import NeuroSAT
from neurosat0 import NeuroSAT0
from dimacs_to_data import one_problem
from dpll import SharpSAT

def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1

    # the following line should be the "p cnf" line
    header = lines[i].strip().split(" ")
    assert(header[0] == "p")
    n_vars = int(header[2])

    # qdimacs file has 2 more lines (for 2QBF)
    # a 1 2 3 ... 0 << the forall variables
    # e 9 10 11 ... 0 << the exist variables
    i += 1
    while (lines[i].strip().split(" ")[0] == 'a' or lines[i].strip().split(" ")[0] == 'e'):
        i += 1
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i:]]
    return n_vars, iclauses

class QBFSolver(object):
    # this is actually a 2QBF solver for generated random problem with specific format
    def __init__(self, specs, sizes, filename):
        # specs: list of size 2. For example: [2, 3] means each clause has 2 forall vars first, then 3 exists vars.
        # sizes: list of size 2. For example: [8, 10] means the problem has 8 forall vars and 10 exists vars.
        # filename: the file name of the 2QBF problem
        self.specs = specs
        self.sizes = sizes
        _, self.iclauses = parse_dimacs(filename)
        self.temp_filename = self.get_temp_filename(filename)
        self.steps = 0

        # set up the SAT solver for counter examples
        self.omega = minisolvers.MinisatSolver()
        # self.omega = SharpSAT()
        # add all forall vars
        for _ in range(sizes[0]):
            self.omega.new_var()
        # add all clause vars (Cz) and the clauses to constraint that (Cz == false of clause c)
        self.Cz_list = []
        for c in self.iclauses:
            self.omega.new_var()
            Cz = self.omega.nvars()
            self.Cz_list.append(Cz)
            for Cc in c[:specs[0]]:
                self.omega.add_clause([-Cz, -Cc])

    def get_temp_filename(self, filename): 
        paths = filename.split('/') 
        index = paths.index('QBF') 
        self.temp_filename = '/'.join(paths[:index+1] + ['temp'] + paths[index+1:]) 
        temp_path = '/'.join(paths[:index+1] + ['temp'] + paths[index+1:-1]) 
        if not os.path.exists(temp_path): 
            os.makedirs(temp_path) 

    def solveRecStart(self, start):
        has_counter, counter = self.check_candidate(list(start))
        if not has_counter:
            self.steps = 1
            return 'has candidate', list(start)
        self.refine_abs(counter)
        return self.solve(start_step=1)

    def solve(self, start_step=0):
        self.steps = start_step
        while True:
            self.steps += 1
            if not self.omega.solve():
                return 'no candidate', None
            candidate = list(self.omega.get_model(end=self.sizes[0]))

            has_counter, counter = self.check_candidate(candidate)
            if not has_counter:
                return 'has candidate', candidate

            self.refine_abs(counter)

    def refine_abs(self, counter):
        Zc = [Cz for (c, Cz) in zip(self.iclauses, self.Cz_list) if not self.trued_by_exists(c, counter)]
        self.omega.add_clause(Zc)        

    def check_candidate(self, candidate):
        assert (len(candidate) == self.sizes[0]), 'candidate should have all forall vars'
        sat = []
        for c in self.iclauses:
            if self.trued_by_forall(c, candidate):
                continue
            else:
                sat.append(c[self.specs[0]:])
        if True:
            return self.run_new_sat(sat)
        else:
            # write sat to file
            fn = self.write_sat_to_file(sat)
            # run sharpSAT and collect result
            n_sat, models = self.run_sharp_SAT(fn)
            return 'TO HERE'

    def run_new_sat(self, iclauses):
        S = minisolvers.MinisatSolver()
        for _ in range(self.sizes[0] + self.sizes[1]):
            S.new_var()
        for c in iclauses:
            S.add_clause(c)
        if S.solve():
            return True, list(S.get_model())
        else:
            return False, None
   
    def trued_by_exists(self, c, t):
        return any([t[abs(c[i])-1] == int(c[i]>0) for i in range(self.specs[0], self.specs[0]+self.specs[1])])

    def trued_by_forall(self, c, t):
        return any([t[abs(c[i])-1] == int(c[i]>0) for i in range(self.specs[0])])

    def shift(self, s):
        return map(lambda x: (x + self.sizes[0]) if x < 0 else (x - self.sizes[0]), s)
    
    def write_sat_to_file(self, sat):
        filename = self.temp_filename
        with open(filename, 'w') as f:
            f.write('p cnf {} {}\n'.format(self.sizes[1], len(sat)))
            for s in sat:
                s = self.shift(s)
                f.write('{} 0\n'.format(' '.join(map(str, s))))
        return filename

    def run_sharp_SAT(self, filename):
        result = subprocess.run(['/homes/wang603/sharpSAT/build/Release/sharpSAT', filename], stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8').split('\n')
        try:
            idx1 = result.index('# solutions ')
            idx2 = result.index('# END')
            assert idx2 == idx1 + 2
            return int(result[idx1 + 1])
        except ValueError:
            print(result)
        # TODO: how to get models from sharpSAT??

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    
    parser = argparse.ArgumentParser()
    add_neurosat_options(parser)

    parser.add_argument('--dimacs_dir', action='store', dest='dimacs_dir', type=str, help='directory of dimacs files')
    parser.add_argument('--restore_id', action='store', dest='restore_id', type=int, default=None)
    parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=None)
    parser.add_argument('--model_id', action='store', dest='model_id', type=int, default=0)
    parser.add_argument('--n_quantifiers', action='store', dest='n_quantifiers', type=int, help='<Required> provide the number of quantifier', required=True)
    parser.add_argument('-a', action='append', dest='specification', type=int, help='<Required> provide the specs of random QBF', required=True)

    opts = parser.parse_args()
    specs = opts.specification[:opts.n_quantifiers]
    sizes = opts.specification[opts.n_quantifiers:]

    setattr(opts, 'run_id', None)
    setattr(opts, 'n_saves_to_keep', 1)
    print(opts)

    # dimacs_dir = '/homes/wang603/QBF/test10_sat/'
    filenames = sorted(os.listdir(opts.dimacs_dir))
    filenames = [os.path.join(opts.dimacs_dir, filename) for filename in filenames]
    results = []

    if opts.restore_id is None or opts.restore_epoch is None:
        g = None
    elif opts.model_id == 0:
        g = NeuroSAT0(opts)
    else:
        g = NeuralSAT(opts)

    for fn in filenames:
        S = QBFSolver([2,3], [8,10], fn)
        if opts.restore_id is None or opts.restore_epoch is None:
            result = S.solve()
        else:
            problem = one_problem(fn, specs, sizes)
            witness = g.reference_one(problem)
            result = S.solveRecStart(tuple(witness))
        results.append((fn, result, S.steps))
        print('file {} result {} in steps {}'.format(fn, result, S.steps))
        exit()
    steps = [r[2] for r in results]
    print('mean steps is {}'.format(sum(steps) / len(steps)))
