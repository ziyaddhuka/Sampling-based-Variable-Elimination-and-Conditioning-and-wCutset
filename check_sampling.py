import networkx as nx
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import sys
import warnings
from tqdm import tqdm
from scipy.special import logsumexp
warnings.filterwarnings('ignore')
import random
import time
from multiprocessing import Pool, cpu_count


class GraphicalModel:
    def read_file(self,model_file_path,evidence_file_path, flag):

        lines = []
        # read all the lines and remove empty lines from it
        for line in open(model_file_path,'r').read().split('\n'):
            if line.rstrip():
                lines.append(line)

        line_no = 0

        # inititalize the graph object
        self.graph = nx.Graph()

        # capturing the network type
        self.network_type = lines[line_no]
        line_no+=1

        # capturing the number of variables
        self.no_of_variables = lines[line_no]
        line_no+=1

        # capturing the cardinalities of the variables
        cardinality_line = lines[line_no].rsplit()
        # storing cardinalities a as list of int
        self.variable_cardinalities = list(map(int,cardinality_line))
        line_no+=1

        # capturing number of cliques
        self.no_of_cliques = int(lines[line_no])
        line_no+=1

        self.factors = []
        self.cpt = []

        # looping through all cliques and adding it them as nodes and vertices of the graph
        for i in range(self.no_of_cliques):
            cliques_input = lines[line_no+i]
            cliques = list(map(int,lines[line_no+i].rsplit()))[1:]
            # adding nodes to the graph
            self.graph.add_nodes_from(cliques)
            # check length of cliques if > 1 then add that edge
            if(len(cliques)>1):
                # if there are more than 2 nodes in the cliques then generate all combinations of pairs and add edge to the graph
                self.graph.add_edges_from(list(set(itertools.combinations(cliques, 2))))
            # append cliques to the factors list
            self.factors.append(np.array(cliques))

        line_no = line_no+i+1

        # looping and saving all the factor table
        # this time the product of factors was going out of limit and due to which I got inf value in numpy
        # hence converting factors to logspace
        for k in range(self.no_of_cliques):
            var = lines[line_no+k]
            self.cpt.append(np.log(np.array(list(map(float,lines[line_no+k+1].split(' '))))))
            line_no = line_no+1


        # looping through evidence file and storing the evidence variables
        if flag==0:
            lines = open(evidence_file_path,'r').read().split('\n')
            line_content = lines[0].rsplit(' ')
            total_evid_vars = int(line_content[0])
            evidence = {}

            if total_evid_vars > 0:
                for i in range(1,total_evid_vars*2,2):
                    evidence[int(line_content[i])] = int(line_content[i+1])
            self.evidence = evidence

        else:
            self.evidence = {}

        # combining the factors and factor tables together in a single array. Will refer to it as "COMBINED ARRAY"
        self.factors_and_cpt = np.array((self.factors,self.cpt),dtype=object).T
        return


def variable_elimination(factors_and_fact_table,reduced_evidence_factors, order):
    # making copy of the array
    cp_factors_and_fact_table = deepcopy(factors_and_fact_table)
    # initializing the bucket as per the min-degree order
    bucket = {l: [] for l in order}

    # looping through all the bucket values in order
    for key,value in bucket.items():
        # initializing empty arrays
        factor_index = []
        cpt_value = []
        idx = []

        # looping through all the factors
        for i in range(cp_factors_and_fact_table.shape[0]):
            # checking if bucket variable is inside the factor array
            if key in cp_factors_and_fact_table[i][0]:
                # if yes than append the factor table of that corresponding factor
                cpt_value.append(cp_factors_and_fact_table[i][1])
                # store the factor as well
                factor_index.append(cp_factors_and_fact_table[i][0])
                # store the index of the factor so as to delete it once processe
                idx.append(i)

        # add those factors to the bucket
        bucket_array = np.array((factor_index,cpt_value,idx),dtype=object).T
        # get unique variables inside the bucket array
        new_clique = get_unique_vars_factors(bucket_array)

        # number of variables is the dimension
        dimension = len(new_clique)
        # format to generate binary numbers
        s = '{0:0'+str(dimension)+'b}' # used to interpret address
        idxes = []
        fact = []

        # loop to all possible numbers from 0 to 2^n
        for i in range(2**dimension):
            # genrate binary string of the number
            binar = s.format(i)
            # convert all the characters to list
            bin_d = list(binar)

            # creating an empty dictionary
            binary_dist = {}
            j=0

            # loop through all the node in the clique and assigning the corresponding bit to that node
            for node in new_clique:
                # assigning the corresponding bit to that node
                binary_dist[node] = bin_d[j]
                j = j+1

            """
            here we can reuse the instantiate evidence function.While doing such we will get all variables
            corresponding to that bit assignment. For e.g. if we have number 1 and two variables X1 and X2
            then the bit generated in the second loop the binary number of 1 will be 01 hence X1=0 and X2=1
            and then get corresponding values of that assignment
            """

            narr,_ = instantiate_evidence(binary_dist,bucket_array)

            # append the indexes which are looped through
            idxes.append(list(binary_dist.values()))
            # multiply all the corresponding factors
            fact.append(np.sum(narr[:,1]))

        # converting to numpy arrays
        idxes = np.array(idxes)
        fact = np.array(fact)

        # converting the arrays in dataframe of columns having nodes and factor
        temp_df = pd.DataFrame(np.hstack((idxes,fact)),columns= new_clique + ['factor'])
        # drop the key/bucket node
        temp_df = temp_df.drop(columns = key)
        # convert factor to float
        temp_df['factor'] = temp_df['factor'].astype(np.float64)

        # remove the bucket node from the array. thus giving us a new factor which is summed out and free of that node
        new_clique.remove(key)
        # check if the length of new clique is > 0
        if len(new_clique)>0:
            # if yes then we need to sum out
            # using groupby function of pandas and grouping by the new_cliques as the keys
            ## this time we use logsumexp for handling the log of summation of factors
            temp_df.groupby(new_clique,as_index=False)
            summed_out_factor = temp_df.groupby(new_clique,as_index=False)['factor'].apply(lambda x: logsumexp(x))['factor']

            # converting the lists to numpy array
            new_clique = np.array(new_clique)
            summed_out_factor = np.array(summed_out_factor)

            # we store new clique factor and the corresponding summed out value
            new_row_of_clique_and_factor = np.array((new_clique,summed_out_factor),dtype=object)

            # adding it to the last row of the factor and tables array
            cp_factors_and_fact_table = np.vstack((cp_factors_and_fact_table,new_row_of_clique_and_factor))

        else:
            # if length is = 0 this means that we have no factors left to sum out hence we have only one value at the end
            # this value will be stored and multiplied with the final factor remaining after the bucket elimination algorithm
            # we sum all the values and store
            reduced_evidence_factors.append(logsumexp(temp_df['factor']))

        # deleting the factors processed in one pass of bucket elimination algorithm
        cp_factors_and_fact_table = np.delete(cp_factors_and_fact_table,(bucket_array[:,-1].tolist()),axis=0)

    # # returning the reduced factors which includes single factors and the factor at the very end of the bucket elimination algorithm
    return reduced_evidence_factors




"""
instantiate_evidence(evidence,arr)
params: evidence variables (a dictionary containing the variable and the value) and the "COMBINED ARRAY"
uses evidence variables and reduces factors accordingly
stores the reduced factors once the evidence is instantiated
returns modified array after instantiating the array and reduced evidence factors
"""
def instantiate_evidence(evidence,arr):
    # copying the combined array contents into a new array to prevent it from modifying original array
    narr = deepcopy(arr)
#     narr = np.array(narr,dtype=np.float64)
    # initializing the empty arrays
    reduced_evidence_factors = []
    index_to_delete = []


    for key,value in evidence.items():
        for i in range(0,narr.shape[0]):
            if key in narr[:,0][i]:
                #find the position of key in the list and instantiate evidence accordingly
                dimension = len(narr[:,][i][0]) #finding the number of variables in the clique
                idx = np.where(narr[:,][i][0]==key)[0][0] # getting the index of evidence variable in the clique
                factor_index_to_keep = get_binary(dimension,idx,value) # getting the index of the factor to keep
                narr[:,1][i] = np.array(narr[:,1][i])[factor_index_to_keep] # getting the value of that index i.e evidence variable
                narr[:,][i][0] = np.delete(narr[:,][i][0],np.where(narr[:,][i][0]==key)) # deleting the variable
                if narr[:,][i][0].size==0: # checking if the reduced clique is 0
                    # if yes then store the single factor
                    reduced_evidence_factors.append(np.array(narr[:,][i][1][0],dtype=np.float64))
                    index_to_delete.append(i)
    # return the modified array and reduced factor
    return narr,reduced_evidence_factors



"""
function to get min degree order of the variables
input: graph and evidence_nodes
output: min degree variables order
"""
def get_min_degree_order(graph,evidence_nodes):
    evidence_nodes = list(evidence_nodes.keys())
    order = []
    # copy the graph and retain the original graph
    tp_g = graph.copy()

    # Get evidence nodes and remove them from the graph before computing the min degree order
    for node in evidence_nodes:
        tp_g.remove_node(node)

    # loop through all the nodes
    for i in range(0,len(tp_g.nodes)):

        # get degree of that node
        temp = dict(tp_g.degree())
        # upgrade it into a dictionary of key value pairs having node as key and degree of that node as value
        # and sort the dictionary by value
        temp = dict(sorted(temp.items(), key=lambda item: item[1]))
        # take the node with least degree
        s = list(temp.keys())[0]
        # get all the edges of that node
        edges = [i[1] for i in list(tp_g.edges(s))]
        # if there are more than 1 edges then we need to connect the edges once we delete the node
        if len(edges)>1:
            # connect all possible edges of all the nodes which are connected to the node to be deleted
            tp_g.add_edges_from(list(set(itertools.combinations(edges,2))))
        # delete the selected node
        tp_g.remove_node(s)
        # append the node to order
        order.append(s)
    return order


"""
function takes input dimension, position of variable, and evidence
suppose there are 2 variables- (X1,X2) then
dimension i.e n = 2
and lets say that the instantiated variable is X2 then var_pos = 2
and assume that the X2 takes the value X2=1 then the evid = 1
"""
def get_binary(n,var_pos,evid):
    # initialize empty array
    l = []
    # loop from 0 to 2^n, used bitwise shift here
    for i in range(1<<n):
        # bin(2)='0b10' and bin(4)='0b100' but we need only the string after b
        s=bin(i)[2:]
        # expanding the binary to fit it to the dimension for e.g. 10 will be 010 in dimension = 3
        s='0'*(n-len(s))+s
        # check if the the corresponding position of evidence matches with the bit value
        # if yes then we only need those
        if s[var_pos]==str(evid):
            l.append(i)
    return np.array(l)



"""
function to get unique variables in a factor
during bucket elimination there might be bucket which contains may factors and some of them get repeated
This function gives unique factors present in the array. For us it gives us unique variables in the bucket
"""
def get_unique_vars_factors(arr):
    output = set()
    for i in range(arr.shape[0]):
        for item in arr[i][0]:
            output.add(item)
    return list(output)



"""
Function runs Bucket Elimination schematically and determines/returns the factors associated with the corresponding bucket

"""
def run_BE_schematically(cp_factors_and_fact_table, order):
    just_factors = cp_factors_and_fact_table[:,0]
    bucket = {k: [] for k in order}
    n_bucket = bucket
    factors = just_factors.copy()
    bucket_elimination_pass = bucket
    ## Running BE schematically and recording the factors
    for key,value in n_bucket.items():

        factor_index = []
        cpt_value = []
        idx = []

        for i in range(factors.shape[0]):
            if key in factors[i]:
                idx.append(i)

        bucket_elimination_pass[key] = factors[idx]

        arr = factors[idx].copy()
        factors = np.delete(factors, idx)
        output = set()

        for i in range(arr.shape[0]):
            for item in arr[i]:
                output.add(item)

        new_clique = list(output)
        new_clique.remove(key)

        new_clique = np.array(new_clique)
        factors = factors.tolist()
        factors.append(new_clique)
        factors = np.array(factors)

    return bucket_elimination_pass


"""
Function uses the w cutset bound and deletes the variables according to the given algorithm
"""
def w_cutset_updated(bucket_elimination_pass,w):
    # given dataframe of executed bucket elimination schematically code
    df = pd.DataFrame(bucket_elimination_pass.items())
    flag = 0
    m = 0
    unique_vars = []
    removed_var = []

    # checking unique variables in a cluster
    for i in range(df.shape[0]):
        unique_vars.append(np.unique([item for sublist in df[1][i] for item in sublist]))
    df['unique'] = unique_vars

    # untill no cluster size is less than w+1 do the following
    while(True):
        max_cluster_size = -1
        # find max cluster size
        for i in range(0,df.shape[0]):
            current_cluster_size = len(df['unique'][i])
            if current_cluster_size > max_cluster_size:
                max_cluster_size = current_cluster_size
        # break if maximum cluster size <= w+1
        if max_cluster_size<= (w+1):
            break

        # if not
        else:
            # make a dataframe of all variables present in all clusters
            ndf = pd.DataFrame([item for sublist in df['unique'] for item in sublist])
            ndf.reset_index(inplace=True,drop=True)
            # count which variable is repeated most time
            count_of_vars_rep = pd.DataFrame(ndf.value_counts(),columns=['count'])
            count_of_vars_rep.reset_index(inplace=True)
            most_rep_vars = count_of_vars_rep[count_of_vars_rep['count'] == count_of_vars_rep['count'].max()]
            # if more than 1 variables we randomly pick a variable
            if most_rep_vars.shape[0]>1:
                most_repeating_element = int(most_rep_vars.sample(1)[0].values)
            # else we pick the most repeating
            else:
                most_repeating_element = most_rep_vars[0][0]

        # updating the dataframe by deleting the most repeated variable
        df = df[df[0]!=most_repeating_element]
        df.reset_index(inplace=True, drop=True)

        for i in range(df['unique'].shape[0]):
            if most_repeating_element in df['unique'][i]:
                df['unique'][i] = [val for val in df['unique'][i] if val!=most_repeating_element]

        removed_var.append(most_repeating_element)
        df.reset_index(inplace=True,drop=True)
    # return the removed variables list
    return removed_var

"""
Generate sample from the evidence variables namely nodes deleted in w_cutset
"""
def generate_simple_sample_from_(discarded_nodes,k):
#     np.random.seed(seed_val)
    evidence = {}
    prod = []
    for variable in discarded_nodes:
        cardinality = k.variable_cardinalities[variable]
        probability_range = 1/cardinality
        prod.append(probability_range)
        no_ = np.random.uniform(0,1)
        for i in range(0,cardinality):
            if no_<(i+1)*probability_range:
                evidence[variable] = i
                break
    return evidence,prod

"""
Generate sample from the evidence variables namely nodes deleted in w_cutset
"""
def generate_sample_from_(discarded_nodes,k):
    evidence = {}
#     np.random.seed(seed_val)
    for variable in discarded_nodes:
        cardinality = k.variable_cardinalities[variable]
        probability_range = 1/cardinality
        no_ = np.random.uniform(0,1)
        for i in range(0,cardinality):
            if no_<(i+1)*probability_range:
                evidence[variable] = i
                break
    return evidence

"""
Getting maximum cardinality of variables deleted in w_cutset
So as to have a dataframe flexible to the variables cardinalities
"""
def get_max_cardinality(discarded_nodes,k):
    mx = 0
    for i in range(len(discarded_nodes)):
        card = k.variable_cardinalities[discarded_nodes[i]]
        if card>mx:
            mx = card
    return mx



"""
Function computes Sampling from Uniform Distribution
"""
def compute_sampleVE(no_of_samples, order, w, k, factors_and_fact_table, true_pr):
    # check time
    start_time = time.time()
    # set seed
    # np.random.seed(seed_val)
    # copy the variables needed
    factors_and_fact_table_cp = deepcopy(factors_and_fact_table)
    bucket_elimination_pass = run_BE_schematically(factors_and_fact_table,order)
    discarded_nodes = w_cutset_updated(bucket_elimination_pass,w)
    Z = 0
    disc = len(discarded_nodes)
    # np.random.seed(seed_val)
    # looping, generating samples and performing BE on the variables
    for i in tqdm(range(0,no_of_samples)):
        evidence = {}
        prod_card = []
        evidence,prod_card = generate_simple_sample_from_(discarded_nodes,k)
        # prod card stores the product of the uniform distribution here in this case it is 0.5^x
        prod_card = np.prod(prod_card)
        factors_and_fact_table_cp = deepcopy(factors_and_fact_table)

        # instantiating the evidence
        new_factors_and_fact_table,reduced_evidence_factors = instantiate_evidence(evidence,factors_and_fact_table_cp)
        # getting the min degree order
        order = get_min_degree_order(k.graph,evidence)
        # performing the VE code
        z = variable_elimination(new_factors_and_fact_table,reduced_evidence_factors, order)
        # converting the log_e space to log_10 space
        z = np.sum(z)/np.log(10)

        # VE would be sum of all factors since we are taking log of partition function
        # the output of VE is already in logspace
        VE = z
        # if no variables are discarded then it is same as exact inference in that case Q=1
        if(len(discarded_nodes)==0):
            Q = 1
            Z = Z + VE/Q
        else:
            Q = (prod_card)
            ## since we are taking log values instead of dividing VE by Q you subtract
            Z = Z + ((VE)-np.log10(Q))

    # computing error
    relative_error = ((true_pr)-(Z/(no_of_samples)))/(true_pr)

    return Z/no_of_samples, relative_error,(time.time() - start_time)


"""
Function computes Adaptive sampling from Proposal Distribution as specified in the algorithm
"""
def compute_adaptiveVE(no_of_samples, order, w, k, factors_and_fact_table, true_pr):
    # log the time
    start_time = time.time()
    # random.seed(seed_val)
    factors_and_fact_table_cp = deepcopy(factors_and_fact_table)
    bucket_elimination_pass = run_BE_schematically(factors_and_fact_table,order)
    discarded_nodes = w_cutset_updated(bucket_elimination_pass,w)
    # getting max cardinality of discarded variables and making the distribution accordingly
    mx = get_max_cardinality(discarded_nodes,k)
    col = []
    # distribution is a dataframe where 'X' denotes the variable 'X0' denotes X=0 and so on...
    col.append('X')
    for i in range(0,mx+1):
        col.append('X' + str(i))
    dist_mat = np.zeros((len(discarded_nodes),mx+2))
    for i in range(len(discarded_nodes)):
        dist_mat[i][0] = discarded_nodes[i]
        for j in range(k.variable_cardinalities[discarded_nodes[i]]):
            dist_mat[i][j+1] = (1/k.variable_cardinalities[discarded_nodes[i]])

    # copy the distribution matrix to the sample distribution and specify the columns accordingly
    sample_distribution = pd.DataFrame(dist_mat,columns=col)
    # initialize Z
    Z = 0
    corrosp_weight = []
    samples_arr = []
    assignment = []
    # loop through all samples
    for ct in tqdm(range(0,no_of_samples)):
        sample = generate_sample_from_(discarded_nodes,k)
        samples_arr.append(sample)
        denominator_Q = []
        for key,value in sample.items():
            # if variable takes value 0 then go to key=variable and column = X0
            ind = 'X' + str(value)
            denominator_Q.append(sample_distribution[sample_distribution['X']==key][ind].values[0])

        # denominator_Q will be the product of the proposal distribtion given the variable assignment
        denominator_Q = np.prod(denominator_Q)
        factors_and_fact_table_cp = deepcopy(factors_and_fact_table)
        evidence = sample.copy()
        new_factors_and_fact_table,reduced_evidence_factors = instantiate_evidence(evidence,factors_and_fact_table_cp)
        order = get_min_degree_order(k.graph,evidence)

        # calling the variable elimination function which returns the partition function
        z = variable_elimination(new_factors_and_fact_table,reduced_evidence_factors, order)

        # converting the log_e space to log_10 space
        z = np.sum(z)/np.log(10)
        log_z = z
        # instead of divide we subtract since we are working in log space
        weight = log_z - np.log10(denominator_Q)
        corrosp_weight.append(weight)

        Z = Z + weight

        assignment_per = []
        for key,value in sample.items():
            assignment_per.append(value)
        assignment.append(assignment_per)
        # once 100 samples are generated we update the propsosal distribution
        # you can change 100 to 10 if it's taking too much time to generate samples > 100
        if (ct+1)%100 == 0:
            # checks the assignment to the variables and computes sum of the weights given that assignment divided by total weight
            proposal_distribution_Q = pd.DataFrame(assignment,columns = np.array(discarded_nodes))
            proposal_distribution_Q['weights'] = corrosp_weight
            total_weight = proposal_distribution_Q['weights'].sum()
            for i in range(0,sample_distribution.shape[0]):
                arr = []
                variable = int(sample_distribution['X'][i])
                arr.append(variable)
                for j in range(0,sample_distribution.shape[1]-1):
                    arr.append(proposal_distribution_Q[proposal_distribution_Q[variable]==j]['weights'].sum()/total_weight)
                idx = sample_distribution[sample_distribution['X']==variable].index
                sample_distribution.iloc[idx] = arr
            sample_distribution['X'] = pd.to_numeric(sample_distribution['X'])

    # compute the error
    relative_error = ((true_pr)-(Z/no_of_samples))/(true_pr)
    return Z/no_of_samples, relative_error,(time.time() - start_time)

"""
function reads the .PR file and gives the true value of Probability/partition function
"""
def get_true_pr(file):
    with open(file) as f:
        lines = f.read().split('\n')
        for line in lines:
            try:
                true_pr = float(line)
            except:
                continue
        return true_pr


"""
when you call the program with its name and arguments this gets called
"""

if __name__ == '__main__':
    if len(sys.argv)<7:
        print("Enter the right number of parameters")
        sys.exit(0)
    algo_type = str(sys.argv[1])
    sample_size = int(sys.argv[2])
    w_cut = int(sys.argv[3])
    network_file = str(sys.argv[4])
    evidence_file = str(sys.argv[5])
    true_pr_file  = str(sys.argv[6])

    try:
        true_pr = get_true_pr(true_pr_file)
    except:
        print('Unable to read the PR file please check the if name or the format is correct')
        sys.exit(0)

    flag = 0
    try:
        k = GraphicalModel()
        k.read_file(network_file,evidence_file,flag)
    except:
        print('Unable to process the Network of evidence file please check the name or the format')
        sys.exit(0)

    factors_and_fact_table = deepcopy(k.factors_and_cpt)
    factors_and_fact_table, reduced_evidence_factors = instantiate_evidence(k.evidence,factors_and_fact_table)
    order = get_min_degree_order(k.graph,k.evidence)
    if algo_type =='adp':
        approx_val, error, time_taken = compute_adaptiveVE(sample_size,order, w_cut, k, factors_and_fact_table, true_pr)
    elif algo_type=='smpl':
        approx_val, error, time_taken = compute_sampleVE(sample_size,order, w_cut, k, factors_and_fact_table, true_pr)
    else:
        print("Enter correct algo parameter for e.g. 'smpl' 'adp'")
        sys.exit(0)
    print('Approximate value of Z is ',approx_val)
    print('Error = ',error)
    print('time_taken = ',time_taken)
