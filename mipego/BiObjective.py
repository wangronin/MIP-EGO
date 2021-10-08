import numpy as np
import copy

def s_metric(expected, solutions,n_left,max_iter,ref_obj1=None,ref_obj2=None,par=None):
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
        x.obj1 *= eps
        x.obj2 *= eps
    val=0.0
    skip_hyp_vol = False #in case that the expected point lies outside the rectangle formed by the reference point and (0,0), and that the expected point is also not dominated (this can happen), the hyper volume must not be calculated
    if (not ref_obj1 is None) and (expected.obj1 > ref_obj1):
        skip_hyp_vol = True
    if (not ref_obj2 is None) and (expected.obj2 > ref_obj2):
        skip_hyp_vol = True

    if (not dominated(expected, eps_par)) and (not skip_hyp_vol):
        #non epsilon dominated solutions receive benefit of hypervolume
        val += (hyper_vol(par_and_exp,sol_and_exp,ref_obj1=ref_obj1,ref_obj2=ref_obj2)-hyper_vol(par,sol_and_exp,ref_obj1=ref_obj1,ref_obj2=ref_obj2))
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
            pen = (1+expected.obj2 - solutions[i].obj2)*(1+expected.obj1-solutions[i].obj1)-1
            val += pen
    return val

def sort_par(par):
    #sort pareto front descending on obj2
    for i in range(len(par)):
        for j in range(i+1, len(par)):
            if (par[i].obj2 < par[j].obj2):
                help = par[i]
                par[i] = par[j]
                par[j] = help
    return par

def quicksort_par(par, lo, hi):
    #sort pareto front ascending on obj1, thus descending on obj2
    if lo < hi:
        p = partition_par(par, lo, hi)
        quicksort_par(par, lo, p - 1 )
        quicksort_par(par, p + 1, hi)

def partition_par(par, lo, hi):
    pivot = par[hi].obj1
    i = lo
    for j in range(lo, hi):
        if par[j].obj1 < pivot:
            help = par[i]
            par[i] = par[j]
            par[j] = help
            i = i + 1
    help = par[i]
    par[i] = par[hi]
    par[hi] = help
    return i

def hyper_vol(par, solutions,ref_obj1=None,ref_obj2=None):
    #set maximum values in pareto front as reference points
    if len(solutions) == 0:
        return 0.0
    if len(par) == 0:
        return 0.0
    if ref_obj1 == None or ref_obj2 == None:
        ref_obj1 = solutions[0].obj1
        ref_obj2 = solutions[0].obj2
        for i in range(1,len(solutions)):
            if (solutions[i].obj1 > ref_obj1):
                ref_obj1 = solutions[i].obj1
            if (solutions[i].obj2 > ref_obj2):
                ref_obj2 = solutions[i].obj2
        ref_obj1 += 1.0
        ref_obj2 += 1.0
    
    loc_par = copy.deepcopy(par)

    #remove entries in pareto front bigger than the reference point (can only happen if ref_obj1 and ref_obj2 are given as parameter to function)
    i = 0
    while i < len(loc_par):
        if loc_par[i].obj1 > ref_obj1 or loc_par[i].obj2 > ref_obj2:
            del loc_par[i]
        else:
            i += 1
    #sort pareto front
    #loc_par = sort_par(loc_par)
    quicksort_par(loc_par, 0, len(loc_par)-1)

    #calculate hypervolume
    if len(loc_par) > 0:
        vol = (ref_obj2 - loc_par[0].obj2) * (ref_obj1 - loc_par[0].obj1)
        for i in range(1,len(loc_par)):
            vol += (loc_par[i-1].obj2 - loc_par[i].obj2) * (ref_obj1 - loc_par[i].obj1)
    else:
        vol = 0.0
    return vol

def dominated(point, solutions):
    #returns whether <point> is dominated by the set <solutions>
    for i in range(len(solutions)):
        if point.obj2 < solutions[i].obj2:
            continue
        if point.obj1 < solutions[i].obj1:
            continue
        if (point.obj2 == solutions[i].obj2) and (point.obj1 == solutions[i].obj1):
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
        if solution.obj1 > par[i].obj1:
            prod1 = 1.0 + solution.obj1 - par[i].obj1
        else:
            prod1 = 1.0
        if solution.obj2 > par[i].obj2:
            prod2 = 1.0 + solution.obj2 - par[i].obj2
        else:
            prod2 = 1.0
        val += -1.0 + prod1 * prod2
    return val

def test_func_1(x):
    return x+np.random.rand()

def test_func_2(x):
    return x**2 + np.random.rand()
