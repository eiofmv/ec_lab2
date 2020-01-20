import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import networkx as nx


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance


def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route


def initial_population(population_size, city_list):
    population = []

    for i in range(0, population_size):
        population.append(create_route(city_list))
    return population


def rank_routes(population):
    fitness_results = {}
    for i in range(0, len(population)):
        fitness_results[i] = Fitness(population[i]).route_distance()

    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=False)


def selection(population_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(population_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(population_ranked[i][0])

    for i in range(0, len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(population_ranked)):
            if pick <= df.iloc[i, 3]:
                selection_results.append(population_ranked[i][0])
                break
    return selection_results


def mating_pool(population, selection_results):
    mp = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        mp.append(population[index])
    return mp


def crossover(parent1, parent2):
    child = parent1.copy()

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    i = end_gene % len(parent1)
    j = end_gene % len(parent2)
    while i != start_gene:
        if parent2[j] not in child[start_gene:end_gene]:
            child[i] = parent2[j]
            i = (i + 1) % len(parent1)
        j = (j + 1) % len(parent2)
    return child


def crossover_population(mp, elite_size):
    children = []
    length = len(mp) - elite_size
    pool = random.sample(mp, len(mp))

    for i in range(0, elite_size):
        children.append(mp[i])

    for i in range(0, length):
        child = crossover(pool[i], pool[len(mp) - i - 1])
        children.append(child)
    return children


def mutate(individual):
    c = random.random()

    if c < 0.5:
        swap1 = random.randint(0, len(individual) - 1)
        swap2 = random.randint(0, len(individual) - 1)
        while swap2 == swap1:
            swap2 = random.randint(0, len(individual) - 1)

        individual[swap1], individual[swap2] = individual[swap2], individual[swap1]
    elif c < 0.75:
        a = random.randint(0, len(individual) - 1)
        b = random.randint(1, len(individual))
        start = min(a, b)
        end = max(a, b)
        r = individual[start:end]
        r.reverse()
        individual[start:end] = r
    else:
        a = random.randint(0, len(individual) - 1)
        b = random.randint(0, len(individual) - 1)
        while a == b:
            b = random.randint(0, len(individual) - 1)

        start = min(a, b)
        end = max(a, b)

        v = individual[end]
        del individual[end]
        individual.insert(start + 1, v)

    return individual


def mutate_population(population, mutation_rate):
    mutated_pop = []

    for ind in range(0, len(population)):
        if random.random() < mutation_rate:
            mutated_ind = mutate(population[ind])
            mutated_pop.append(mutated_ind)
        else:
            mutated_pop.append(population[ind])

    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate):
    population_ranked = rank_routes(current_gen)
    selection_results = selection(population_ranked, elite_size)
    mp = mating_pool(current_gen, selection_results)
    children = crossover_population(mp, elite_size)
    return mutate_population(children, mutation_rate)


def draw_route(route):
    pos = {i: (v.x, v.y) for i, v in enumerate(route)}
    X = nx.Graph()

    X.add_nodes_from(pos.keys())

    for i in range(len(route)):
        X.add_edge(i, (i + 1) % len(route))

    nx.draw(X, pos=pos, node_size=10, arrowsize=3)
    plt.show()


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    city_list = []
    for line in lines[8:-1]:
        n, x, y = map(int, line.split())
        city_list.append(City(x=x, y=y))

    return city_list


def genetic_algorithm(population, population_size, elite_size, mutation_rate, generations):
    pop = initial_population(population_size, population)
    print("Initial distance: %.3f" % (rank_routes(pop)[0][1]))

    for i in tqdm_notebook(range(0, generations)):
        pop = next_generation(pop, elite_size, mutation_rate)

    print("Final distance: %.3f" % (rank_routes(pop)[0][1]))
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    return best_route
