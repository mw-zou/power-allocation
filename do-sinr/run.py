
import pickle
import os
from examples_blotto import *
# Create directoty for saving
dir_name = "Results"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Function for finding an equilibrium and saving the results
def run_game(game, method="DO", maxitr=2, epsilon=1e-10, file_name=None):
    xs_full, p_full, ys_full, q_full, lower_bounds, upper_bounds = double_oracle(game, maxitr, method=method, epsilon=epsilon)
    xs, p, ys, q = reduce_strategies(xs_full, p_full, ys_full, q_full, epsilon=1e-10)
    
    if not file_name is None:
        with open(file_name + ".pkl", 'wb') as f:###
            pickle.dump([game, xs, p, ys, q, xs_full, p_full, ys_full, q_full, lower_bounds, upper_bounds], f)

    return xs, p, ys, q, lower_bounds, upper_bounds


#####################################
#  COLONEL BLOTTO GAME
#####################################

def run_blotto(n_all, c_all, iter_type_all, a_type_all):#
    comb_all = np.array(np.meshgrid(n_all, c_all, iter_type_all, a_type_all)).T.reshape(-1,4)#
    for i in range(len(comb_all)):
    
        n         = int(comb_all[i,0])#
        c         = float(comb_all[i,1])##
        init_type = comb_all[i,2]##iter_ty
        a_type    = comb_all[i,3]##a_type
        if a_type == "equal":###
            a = np.ones(n)
        elif a_type == "unequal":
            a = np.array(range(3, 3+n))
            
        game = Blotto( ProbBlock(n), ProbBlock(n), a, c, init_type=init_type)##
        file_name = os.path.join(dir_name, "Blotto" + "_" + str(n) + "_" + str(c) + "_" + init_type + "_" + a_type)#
        run_game(game, method="DO", file_name=file_name)


n_all = [3]##
c_all = [1/64]##
#iter_type_all = ["bounds"]
iter_type_all = ["uniform"]#
a_type_all    = ["equal"]###
run_blotto(n_all, c_all, iter_type_all, a_type_all)###



