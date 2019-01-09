# Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import math

def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

class Problem(object):
    def __init__(self, specs, sizes, n_vars_AL, iclauses, labels, n_A_cells_per_batch, n_L_cells_per_batch, all_dimacs):
        self.specs = specs
        self.sizes = sizes
        self.n_vars_AL = n_vars_AL
        self.n_lits_AL = [n_vars_AL[0] * 2, n_vars_AL[1] * 2]
        self.n_clauses = len(iclauses)

        self.n_cells_A = sum(n_A_cells_per_batch)
        self.n_cells_L = sum(n_L_cells_per_batch)
        self.n_A_cells_per_batch = n_A_cells_per_batch
        self.n_L_cells_per_batch = n_L_cells_per_batch

        # self.is_sat = is_sat
        self.labels = labels
        self.compute_AL_unpack(iclauses)

        # will be a list of None for training problems
        self.dimacs = all_dimacs

    def compute_AL_unpack(self, iclauses):
        self.A_unpack_indices = np.zeros([self.n_cells_A, 2], dtype=np.int)
        self.L_unpack_indices = np.zeros([self.n_cells_L, 2], dtype=np.int)
        cell_A = 0
        cell_L = 0
        for clause_idx, iclause in enumerate(iclauses):
            assert(len(iclause) == sum(self.specs))
            for i in range(self.specs[0]):
                vlit = ilit_to_vlit(iclause[i], self.n_vars_AL[0])
                self.A_unpack_indices[cell_A, :] = [vlit, clause_idx]
                cell_A += 1
            for i in range(self.specs[0], sum(self.specs)):
                vlit = ilit_to_vlit(iclause[i], self.n_vars_AL[1])
                self.L_unpack_indices[cell_L, :] = [vlit, clause_idx]
                cell_L += 1
            # vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            # for vlit in vlits:
            #     self.L_unpack_indices[cell, :] = [vlit, clause_idx]
            #     cell += 1

        assert(cell_A == self.n_cells_A)
        assert(cell_L == self.n_cells_L)

def shift_ilit(x, offsets, idx, sizes):
    # x is the actual lit in the clause
    # offsets is the number of collected vars, i.e. (80, 100) means that the batch as collected 80 forall vars and 100 exist vars
    # idx is the category of x (0 or 1)
    # sizes is the number of vars for this problem, i.e. (8, 10) means that the current problem has 8 forall vars and 10 exist vars
    assert (x != 0)
    offset_in_problem = []
    for i in range(len(sizes)):
        offset_in_problem.append(0 if i == 0 else (offset_in_problem[i-1] + sizes[i-1]))
    if x > 0: return x + (offsets[idx] - offset_in_problem[idx])
    else:     return x - (offsets[idx] - offset_in_problem[idx])

#def shift_ilit(x, offset):
#    assert(x != 0)
#    if x > 0: return x + offset
#    else:     return x - offset

def shift_iclauses(iclauses, specs, sizes, offsets):
    # specs is a tuple of 2, i.e. (2, 3), meaning that there are 2 foralls and 3 exists per clause
    # sizes is a tuple of 2, i.e. (8, 10), meaning that this problem has 8 foralls and 10 exists
    # offsets is a tuple of 2, i.e.(80, 100), meaning that the batch has collected 80 forall vars and 100 exist vars
    assert (len(specs) == 2), "only handle 2QBFs"
    assert (len(offsets) == 2), "only handle 2QBFs"
    for iclause in iclauses:
        assert(len(iclause) == sum(specs)), "num of vars should fits the specs"
        for i in range(specs[0]):
            iclause[i] = shift_ilit(iclause[i], offsets, 0, sizes)
        for i in range(specs[0], sum(specs)):
            iclause[i] = shift_ilit(iclause[i], offsets, 1, sizes)
    return iclauses
    # return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]

# this function only applies to 2QBF
def mk_batch_problem_2QBF(problems):
    all_iclauses = []
    # all_is_sat = []
    all_labels = []
    all_n_cells_L = []
    all_n_cells_A = []
    all_dimacs = []
    offsets = [0, 0]

    prev_specs = None
    prev_sizes = None
    # specs is about how many of each variables in each quantifier block
    # for example (2, 3) means each clause has 5 vars, first 2 from forall, last 3 from exists
    # sizes is about how many of total variables in each quantifier block
    # for example (8, 10) means the problem has total 8 vars for forall, total 10 vars for exists 
    for dimacs, specs, sizes, iclauses, labels in problems:
        assert(len(specs) == 2)
        assert(len(sizes) == 2)
        assert(prev_specs is None or specs == prev_specs)
        assert(prev_sizes is None or sizes == prev_sizes)
        prev_specs = specs
        prev_sizes = sizes

        # concatenate clauses of a batch of problems together requires the vars to be shifted
        all_iclauses.extend(shift_iclauses(iclauses, specs, sizes, offsets))
        # all_is_sat.append(is_sat)
        all_labels.append(labels)
        all_n_cells_A.append(len(iclauses) * specs[0])
        all_n_cells_L.append(len(iclauses) * specs[1])
        # all_n_cells_L.append(sum([len(iclause) for iclause in iclauses]))
        all_dimacs.append(dimacs)
        for i in range(len(sizes)):
            offsets[i] += sizes[i]

    return Problem(prev_specs, prev_sizes, offsets, all_iclauses, all_labels, all_n_cells_A, all_n_cells_L, all_dimacs)
