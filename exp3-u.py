import numpy as np
import math
import scipy.sparse as sci
import networkx as nx
import time
import pandas as pd
import random
# INPUT of the game
n = 3  # number of battlefields
m = 3  # number of troops of LEARNER，，，100
p = m  # number of troops of ADVERSARY，，2
save = 'explore-exp3-u.csv'


values = np.array([0.25, 0.4, 0.35])
exploration = 'Uniform'  # Either 'Uniform' exploration Or 'Optimal' exploration
setting = 'Semi_bandit-OE'  # Choose setting = 'Full_info' or 'Semi_bandit' or 'Bandit'
setting1 = 'exploi'
for adv_distribution in ['Uniform_1']:  # Adversary plays 'Battlefield_wise' or 'Uniform_1' or 'Test_extreme'

    # GENERATE values of battlefields
    # BLOTTO rule
    def blotto(x, y):
        if (x >= y):
            return 1
        elif (x < y):
            return 0

    #
    N = 2 + (n - 1) * (m + 1)  # Number of nodes
    E = int(2 * (m + 1) + (n - 2) * (m + 1) * (m + 2) / 2)  # number of edges
    P = int(math.factorial(m + n - 1) / (math.factorial(m) * math.factorial(n - 1)))

    ###########################################################################
    ###########################################################################
    # DEFINE the graph and find all (binary) paths
    # Define the graph
    G = nx.DiGraph()
    G.add_nodes_from(range(0, N))  # add N nodes, indexed by 0 to N-1

    # Define a function to compute the layer of a node，，
    def Layer(u):
        if u == 0:
            return int(0)
        elif u == N - 1:
            return int(n)
        else:
            return int(np.floor((u - 1) / (m + 1)) + 1)


    def Vertical(u):
        if (u == 0):
            Vertical = 0
        elif (u == N - 1):
            Vertical = int(m)
        elif (np.remainder(u, m + 1) == 0):
            Vertical = m
        else:
            Vertical = np.remainder(u, m + 1) - 1
        # print("Vertical=",Vertical)
        return Vertical

    def Children(u):
        if u == 0:
            children = np.array(range(1, m + 2))
        elif (u >= N - 1 - (m + 1)):
            children = np.array([N - 1])
        else:
            temp = range(0, m + 1 - Vertical(u))
            children = (u + m + 1) * np.ones(m + 1 - Vertical(u))
            children = children + temp
        return children.astype(int)


    # Add edges from 0 to 1st-layer nodes，，
    for j in range(1, m + 2):
        # Trans_matrix[0,j] =1  `
        G.add_edge(0, j)
    # Add edges from (n-1)th-layer nodes to N-1
    for i in range(1 + (n - 2) * (m + 1), N - 1):
        # Trans_matrix[i,N-1]=1
        G.add_edge(i, N - 1)
    # Add edges for other nodes
    u = 1
    while u < N - 1 - (m + 1):
        for j in range(u + m + 1, int((Layer(u) + 1) * (m + 1) + 1)):
            # Trans_matrix[u,j]=1
            G.add_edge(u, j)
        u += 1

    # DRAW the graph
    nx.draw(G, pos=nx.shell_layout(G), with_labels=True)

    # The sets of nodes, edges, paths (series of nodes)
    nodes = list(G.nodes())
    edges = list(G.edges())
    paths_by_nodes = list(nx.all_simple_paths(G, source=0, target=N - 1))


    ##########  Translate Graph into Blotto         ############
    def bin_path(p):  # Translate a paths_by_nodes into a binary path \in {0,1}^E
        path_temp = np.zeros(shape=(n, 2))
        for i in range(0, n):
            path_temp[i] = [p[i], p[i + 1]]
        bin_paths = np.zeros(E)
        for j in range(0, E):
            if any(np.equal(path_temp, edges[j]).all(1)) == 1:  # if the j-th edge is in the path_temp (edgewise)
                bin_paths[j] = 1
        return bin_paths


    def bin_path1(p):  # Translate a paths_by_nodes into a binary path \in {0,1}^E，，，eg:p=[0 4 8 9]
        path_temp = np.zeros(shape=(n, 2))  ##n=3
        for i in range(0, n):
            path_temp[i] = [p[i], p[i + 1]]
        return path_temp  ###


    def allo(e):  # The allocation corresponding to an edge e
        alloc = np.zeros(2)
        alloc[1] = Layer(edges[e][1])  # The battlefield corresponds to edge e
        if (edges[e][0] == 0) or (edges[e][1] == N - 1):  # if the edge comes from source 0 or to root N-1
            alloc[0] = edges[e][1] - edges[e][0] - 1  # allocate alloc[0] to battlefield alloc[1]
        else:
            alloc[0] = edges[e][1] - edges[e][0] - (m + 1)
        return alloc


    # print([allo(0),allo(11),allo(5)])
    ########################################################################
    ##################   STRATEGY of ADVERSARY       ########################
    #
    ### ADversary strategy
    def adversary(option):
        adver_stra = np.zeros(n)
        if (option == 'Uniform_1'):
            battle = np.array(range(1, n + 1))
            resource = p
            while (resource > 0 and battle.size > 1):
                b = np.random.choice(battle, size=1, replace=False)  # draw uniformly a battlefield
                allo = random.randint(0, resource)  # Draw uniformly an allocation
                resource = resource - allo
                adver_stra[b - 1] = allo
                battle = np.delete(battle, np.where(battle == np.array(b)))
                # print([b, allo],battle,resource,'\n')

            adver_stra[battle - 1] = resource
        elif (option == 'Battlefield_wise'):
            resource = p
            while (resource > 0):
                b = np.random.choice(np.array(range(1, n + 1)), size=1,
                                     p=values / sum(values))  # draw a battlefield based on values
                adver_stra[b - 1] += 1
                resource -= 1
        return adver_stra

    ##############################################################################
    #    LOSS generated by the Adversary
    def Loss(adver_stra):  # A E-dim vector that is the loss of each egde comparing with adver_stra
        L = np.zeros(E)
        for e in range(0, E):
            L[e] = values[int(allo(e)[1]) - 1] - values[int(allo(e)[1]) - 1] * blotto(allo(e)[0], adver_stra[
                int(allo(e)[1]) - 1])  # allo[0]=allocation to allo[1] = battlefield
        return L#



    ###########   MAIN PART        #########################


    def explore(option):
        if option == 'Uniform':
            item = np.random.choice(range(P), size=1, p=None)  # Uniformly uniform a path
            path = paths_by_nodes[int(item)]
        elif option == 'Optimal':

            item = np.random.choice(range(P), size=1, p=[0.3, 0.4, 0.3])
            path = paths_by_nodes[int(item)]
        return path


    #########    Dynamic computing     #########################

    ####### this can be reduced by usinhg special structure of Blotto graph
    def update_H(w):
        H = np.identity(N)
        for j in np.flip(nodes, axis=0):
            for i in np.flip(list(nx.ancestors(G, j)), axis=0):###8,7,6,5
                for k in nx.dfs_successors(G, i)[i]:
                    H[i, j] = H[i, j] + w[edges.index((i, k))] * H[k, j]
        return H

    # w_test = np.array(range(1,E+1))
    # H=update_H(w_test)


    ######### exploit using w_edges and H = Sample a path by w_edges and H #########
    def exploit(H, w):
        node_k_1 = 0
        chosen_path = np.array([0])
        while (len(chosen_path) <= n):
            prob = np.array([])

            for k in nx.dfs_successors(G, node_k_1)[node_k_1]:
                prob = np.append(prob, [w[edges.index((node_k_1, k))] * H[k, N - 1] * (1 - eta) / H[node_k_1, N - 1] + eta * (1/ (m + 1))])  ###
            prob /= prob.sum()
            node_k = np.ndarray.item(np.random.choice(nx.dfs_successors(G, node_k_1)[node_k_1], size=1, p=prob))
            chosen_path = np.append(chosen_path, node_k)
            node_k_1 = node_k

        return chosen_path


    # chosen_path=exploit(H,w_test)
    # print(chosen_path)

    def single_prob(e, w, H):  # probability that an edge e is chosen by exploiting distribution
        single_prob = H[0, edges[e][0]] * w[e] * H[edges[e][1], N - 1] / H[0, N - 1]
        return single_prob


    def coocurence_mat(w, H):
        mat = np.zeros((E, E))
        for e_1 in range(E):
            mat[e_1, e_1] = single_prob(e_1, w, H)
            for e_2 in range(e_1 + 1, E):
                mat[e_1, e_2] = H[0, edges[e_1][0]] * w[e_1] * H[edges[e_1][1], edges[e_2][0]] * w[e_2] * H[
                    edges[e_2][1], N - 1] / H[0, N - 1]
                mat[e_2, e_1] = mat[e_1, e_2]
        return mat

    #####################################################################
    ################## LOST ESTIMATION ##############
    ######################################################################
    def est_loss(path, adver_stra, w, option, loss):
        estimate1 = [0]*E  # Estimate loss for each edge,,
        H = update_H(w)
        if option == 'Full_info':
            estimate = Loss(adver_stra)
        elif option == 'Bandit':
            #        bandit_loss = np.dot(Loss(adver_stra),bin_path(path))
            C = (1 - gamma) * coocurence_mat(w, H) + gamma * C_explore
            #        C = coocurence_mat(w,H)
            estimate = np.asarray(loss * (np.matmul(np.linalg.pinv(C), bin_path(path)))).flatten()
        #        for e in range(len(estimate)):
        #            if (estimate[e] < 0): estimate[e]=0
        #            elif (estimate[e]  >1): estimate[e] =1
        elif option == 'Semi_bandit-OE':

            celue1=[0,0,0]
            for i in range(1,N-1):
                if Layer(i) == 1:

                    if chosen_path[1] >= i and celue1[0]<adver_stra[0]:
                        w_edges1 = 1 * w_edges
                        w = 0
                        for j in range(i, m + 1+1):
                            w = w + w_edges1[edges.index((0, j))] * H[j, N - 1]/ H[0, N - 1]
                            w_edges1[edges.index((0, j))] = 0  #
                        estimate1[i - 1] = (celue1[0]-(i-1))*Loss(adver_stra)[edges.index(
                            (0, i))] / (w+beta)

                    if chosen_path[1] <= i and celue1[0] >= adver_stra[0]:
                        estimate1[i - 1] = 0
                #   estimate1[i + m] = Loss(adver_stra)[edges.index(edges[chosen_path[2] + N-1])]
                if Layer(i) == n - 1:
                    if chosen_path[n - 1] <= i and celue1[2] < adver_stra[2]:
                        estimate1[i - 1] = 0
                elif Layer(i) !=1 and Layer(i)!=n-1:
                    print(i, "is ok")

        else:
            print('error in the name of bandit setting')
        return np.array(estimate1)

    regret1=[]
    for Time in [30000]:
        T = Time  # Time horizon
        start = time.time()
        #######################################################################################################
        #############################          MAIN ALGORITHM          ########################################
        #######################################################################################################
        adv_store = np.zeros((T + 1, n))
        # Find the lambda_min corresponding to either uniform exploration or optimal exploration
        if exploration == 'Uniform':
            w_uniform = np.ones(E)
            H_uniform = update_H(w_uniform)
            C_explore = coocurence_mat(w_uniform, H_uniform)
            eigenval = np.sort(np.round(np.linalg.eig(C_explore)[0], 8))
            for i in range(E):
                if np.real(eigenval[i]) + 0 > 0:
                    lambda_min = np.real(eigenval[i])
                    break

        elif exploration == 'Optimal':
            #Input of the optimal exploration distribution
            Location = r'C:\Users\ai\mu.csv' #Change this for the location of data file
            data= pd.read_csv(Location, header=None, usecols=[0]) #columns resp. to the instance
            data = np.array(data.dropna())
            lambda_min = np.asscalar(data[-1])
            distri_mu = (data[0:P]).flatten()
            #### The follows should be computed a priori of this algorithm
            def f(p): ## define the function of covariance matrix
                return np.dot(p.T,p)
            C_explore = sci.lil_matrix((E, E))
            C_explore= distri_mu[0]*f(sci.lil_matrix(bin_path(paths_by_nodes[0])))
            for i in range(1,P):
                C_explore = C_explore +  distri_mu[i]*f(sci.lil_matrix(bin_path(paths_by_nodes[i])))
            C_explore = np.matrix(C_explore.todense())
        eta = 0.1
        beta = E**2 / (3+2*E)#
        gamma = 0##
        lambda_min=0#

        ###########################################################################

        # Initialization
        Cul_loss = 0
        w_edges = np.ones(E)
        H = np.identity(N)
        chosen_path = np.zeros(E)

        for t in range(1, T + 1):
            adver_stra = adversary(
                adv_distribution)  # Adversary plays an action: Option= 'Battlefield_wise' or 'Uniform_1'
            adv_store[t] = adver_stra
            #    print('gamma = ', gamma)
            Coin = np.random.choice([0, 1], size=1, p=[1 - gamma, gamma])  # Flip a coin, bias gamma
            #    Coin = 0
            if (setting1 == 'exploi'): Coin = 0
            if Coin == 0:  # EXPLOIT
                print('T=', t, 'exploit')
                H = update_H(w_edges)  # Update H according to current w_edges
                chosen_path = exploit(H, w_edges)  # Dr
                loss = np.dot(bin_path(chosen_path), Loss(adver_stra))  # A loss (scalar) generated by adversay
                Cul_loss += loss
                estimate = est_loss(chosen_path, adver_stra, w_edges, setting, loss)  # Unbiasly Estimate the loss according to setting

                w_edges = w_edges * np.exp(-eta * estimate)

            elif Coin == 1:  # EXPLORE
                print('t=', t, 'explore')
                chosen_path = explore(exploration)  ## Explore by either exploration='Uniform' or either ='Optimal'
                loss = np.dot(bin_path(chosen_path),
                              Loss(adver_stra))  # A loss generated by adversay, unobserved by player
                Cul_loss += loss
                estimate = est_loss(chosen_path, adver_stra, w_edges, setting,
                                    loss)  # Unbiasly Estimate the loss according to setting
                w_edges = w_edges * np.exp(-eta * estimate)
        ###############################################################################
        ############    COMPUTE BEST ACTION IN HIND-SIGHT   ##########################
        K = np.zeros((m + 1, n + 1))
        V = np.zeros((m + 1, n + 1))
        # compute the value of best action to GAIN THE MAXIMUM
        for i in range(m + 1):  # troops
            for j in range(1, n + 1):  # battlefield
                K[i, j] = blotto(i, adv_store[1, j - 1]) * values[j - 1]
                for t in range(2, T + 1):
                    K[i, j] = K[i, j] + blotto(i, adv_store[t, j - 1]) * values[j - 1]
                temp = 0
                for k in range(0, i + 1):
                    temp = max(temp, V[k, j - 1] + K[i - k, j])
                V[i, j] = temp
        regret = Cul_loss - (T - V[m, n])  # Cul loss minus the (min loss a.k.a. max gain)#
        regret1.append(regret)
        end = time.time()

        ####################################################################################
        ######################         SAVING THE RESULTS        #########################
        #
        df = pd.read_csv(save, header=None)
        df1 = pd.DataFrame(np.append([n, m, p, T],
                                     [gamma, eta, exploration, lambda_min, setting, adv_distribution,
                                      end - start, T - V[m, n], regret]))#

        output = pd.concat([df, df1], ignore_index=False, axis=1)
        output.to_csv(save, header=None, index=False)
    print("regret1=", regret1)