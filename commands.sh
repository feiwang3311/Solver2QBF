CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 # 33.138

# Predict witness model 1
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 10302 --restore_epoch 41676 # 33.05
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 10302 --restore_epoch 41676 # 19.5725 
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 10302 --restore_epoch 41676 # 20.31 
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 10302 --restore_epoch 41676 # 34.354 

# Predict witness model 2
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 6502 --restore_epoch 33897 # 33.125
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 6502 --restore_epoch 33897 # 19.6725
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 6502 --restore_epoch 33897 # 14.084 
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 6502 --restore_epoch 33897 # 33.972

# Predict witness model 3
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 20301 --restore_epoch 15339 # 32.8225 
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/test10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 20301 --restore_epoch 15339 # 18.9275
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_unsat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 20301 --restore_epoch 15339 # 19.213
CUDA_VISIBLE_DEVICES=0 python3 solver.py --dimacs_dir /homes/wang603/QBF/train10_sat/ --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10 --restore_id 20301 --restore_epoch 15339 # 34.019 
