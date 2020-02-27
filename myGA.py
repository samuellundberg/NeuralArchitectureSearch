import numpy as np

# initiate a random population of size N
def rand_init(N):
    pop = np.zeros(N)
    # do something
    return pop

# evaluate performance of phenotypes in pop
def evaluate(pop):
    N = len(pop)
    rank = range(N)
    # do something
    return rank

# Gives a new popolation given the old polulation and its rank
def update_population(old_pop, rank):
    N = len(old_pop)
    new_pop = []
    for idx, pheno in enumerate(old_pop):
        if rank[idx] < N/2:
            new_pop.append(pheno)
        else:
            new_pop.append(rand_init(1)[0])



