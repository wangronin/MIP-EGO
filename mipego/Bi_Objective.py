import numpy as np
import copy

def s_metric(expected, solutions,n_left,max_iter,ref_time=None,ref_loss=None,par=None):
    sol_and_exp = copy.deepcopy(solutions)
    sol_and_exp.append(expected)#existing solutions and expected solution unified
    if par is None:
        par = pareto(solutions)#pareto front of existing solutions
    par_and_exp = copy.deepcopy(par)
    par_and_exp.append(expected)
    par_and_exp = pareto(par_and_exp)
    
    #calculate epsilon vector
    eps = 0.98 - 0.08*(n_left/max_iter)#vary eps starting at 10% going down to 2%
    eps_par = copy.deepcopy(par)
    for x in eps_par:
        x.time *= eps
        x.loss *= eps
    val=0.0
    skip_hyp_vol = False #in case that the expected point lies outside the rectangle formed by the reference point and (0,0), and that the expected point is also not dominated (this can happen), the hyper volume must not be calculated
    if (not ref_time is None) and (expected.time > ref_time):
        skip_hyp_vol = True
    if (not ref_loss is None) and (expected.loss > ref_loss):
        skip_hyp_vol = True

    if (not dominated(expected, eps_par)) and (not skip_hyp_vol):
        #non epsilon dominated solutions receive benefit of hypervolume
        val += (hyper_vol(par_and_exp,sol_and_exp,ref_time=ref_time,ref_loss=ref_loss)-hyper_vol(par,sol_and_exp,ref_time=ref_time,ref_loss=ref_loss))
    else:
        #epsilon dominated solutions only receive penalty for inferior objectives
        pass
    if dominated(expected,par):
        #dominated expected point receives penalty
        val -= penalty(expected,solutions)
    return val

def penalty(expected,solutions):
    val = 0.0
    for i in range(len(solutions)):
        if dominated(expected, [solutions[i]]):
            pen = (1+expected.loss - solutions[i].loss)*(1+expected.time-solutions[i].time)-1
            val += pen
    return val

def sort_par(par):
    #sort pareto front descending on loss
    for i in range(len(par)):
        for j in range(i+1, len(par)):
            if (par[i].loss < par[j].loss):
                help = par[i]
                par[i] = par[j]
                par[j] = help
    return par

def quicksort_par(par, lo, hi):
    #sort pareto front ascending on time, thus descending on loss
    if lo < hi:
        p = partition_par(par, lo, hi)
        quicksort_par(par, lo, p - 1 )
        quicksort_par(par, p + 1, hi)

def partition_par(par, lo, hi):
    pivot = par[hi].time
    i = lo
    for j in range(lo, hi):
        if par[j].time < pivot:
            help = par[i]
            par[i] = par[j]
            par[j] = help
            i = i + 1
    help = par[i]
    par[i] = par[hi]
    par[hi] = help
    return i

def hyper_vol(par, solutions,ref_time=None,ref_loss=None):
    #set maximum values in pareto front as reference points
    if len(solutions) == 0:
        return 0.0
    if len(par) == 0:
        return 0.0
    if ref_time == None or ref_loss == None:
        ref_time = solutions[0].time
        ref_loss = solutions[0].loss
        for i in range(1,len(solutions)):
            if (solutions[i].time > ref_time):
                ref_time = solutions[i].time
            if (solutions[i].loss > ref_loss):
                ref_loss = solutions[i].loss
        ref_time += 1.0
        ref_loss += 1.0
    
    loc_par = copy.deepcopy(par)

    #remove entries in pareto front bigger than the reference point (can only happen if ref_time and ref_loss are given as parameter to function)
    i = 0
    while i < len(loc_par):
        if loc_par[i].time > ref_time or loc_par[i].loss > ref_loss:
            del loc_par[i]
        else:
            i += 1
    #sort pareto front
    #loc_par = sort_par(loc_par)
    quicksort_par(loc_par, 0, len(loc_par)-1)

    #calculate hypervolume
    if len(loc_par) > 0:
        vol = (ref_loss - loc_par[0].loss) * (ref_time - loc_par[0].time)
        for i in range(1,len(loc_par)):
            vol += (loc_par[i-1].loss - loc_par[i].loss) * (ref_time - loc_par[i].time)
    else:
        vol = 0.0
    return vol

def dominated(point, solutions):
    #returns whether <point> is dominated by the set <solutions>
    for i in range(len(solutions)):
        if point.loss < solutions[i].loss:
            continue
        if point.time < solutions[i].time:
            continue
        if (point.loss == solutions[i].loss) and (point.time == solutions[i].time):
            continue
        return True
    return False

def pareto(solutions):
    #returns the set of non-dominated solutions in <solutions>
    par = []
    for i in range(len(solutions)):
        if not dominated(solutions[i], solutions):
            par.append(copy.deepcopy(solutions[i]))
    return par

def eps_penalty(solution,par):
    #TODO_penalty of epsilon point by dominated paretofron only, reuse penalty function
    val = 0.0
    for i in range(len(par)):
        if solution.time > par[i].time:
            prod1 = 1.0 + solution.time - par[i].time
        else:
            prod1 = 1.0
        if solution.loss > par[i].loss:
            prod2 = 1.0 + solution.loss - par[i].loss
        else:
            prod2 = 1.0
        val += -1.0 + prod1 * prod2
    return val

def test_func_1(x):
    return x+np.random.rand()

def test_func_2(x):
    return x**2 + np.random.rand()
