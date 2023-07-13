import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Calcular el valor de fitness de cada solución en la población actual.
    # La función de aptitud calcula la suma de los productos 
    #entre cada entrada y su peso correspondiente.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selección de los mejores individuos de la generación actual
    # como progenitores para producir la descendencia de la generación siguiente.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # Punto en el que se produce el cruce entre dos progenitores. 
    #Por lo general, se encuentra en el centro.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Índice del primer progenitor en aparearse.
        parent1_idx = k%parents.shape[0]
        # Índice del segundo progenitor en aparearse.
        parent2_idx = (k+1)%parents.shape[0]
        # La nueva descendencia tendrá la primera mitad de sus genes 
        #tomados del primer progenitor.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # La nueva descendencia tendrá la segunda mitad de sus genes
        # tomados del segundo progenitor.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # La mutación cambia un número de genes definido por el argumento 
    #num_mutations. Los cambios son aleatorios.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # El valor aleatorio que se añadirá al gen.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover