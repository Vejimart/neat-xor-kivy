"""
This is based on the paper "Evolving Neural Networks through Augmenting Topologies", from
Kenneth O. Stanley and Risto Miikkulainen. Any comments mentioning "The paper" are a
reference to this.

Arturo Velázquez Jiménez, 2020
"""

import random
from datetime import datetime
from copy import deepcopy

import neural_network as nn


# Function that will return True with a probability of (probability * 100)%
def chance(probability):
    return random.random() < probability


def randomize_gene(genotype, key):
    w = (random.random() * 10)
    genotype[key]["weight"] = random.choice([w, -w])


def mutate_genotype(genotype, mutation_probability, perturbation_probability, perturbation_delta):

    for key in genotype:
        if chance(mutation_probability):
            if chance(perturbation_probability):
                # Perturbate weight
                genotype[key]["weight"] += random.choice([perturbation_delta, -perturbation_delta])
            else:
                randomize_gene(
                    genotype=genotype,
                    key=key
                )


def random_insert_node(genotype, split_conn_tracker, conn_innov_counter, node_innov_counter):

    # A node can only be inserted if there are existing connections. Avoid  that with a try-finally block
    try:
        key = random.choice([k for k in genotype if genotype[k]["enable"] is True])
        old_gene = genotype[key]

        if key in split_conn_tracker.keys():

            # This was already mutated in this generation. Keep the previous mutation

            into_node = deepcopy(split_conn_tracker[key]["in_gene"])
            into_node_key = split_conn_tracker[key]["in_gene_key"]
            out_of_node = deepcopy(split_conn_tracker[key]["out_gene"])
            out_of_node_key = split_conn_tracker[key]["out_gene_key"]
            """
            The out_gene stored in split_conn_tracker will have the weight of the original connection
            it replaced. It is necessary to change it to the weight of the connection being replaced.
            """
            out_of_node["weight"] = old_gene["weight"]

        # except ValueError:
        else:
            # couldn't find innov in split_conn_tracker, so make a new mutation

            # Make a new dictionary associated with "key" to store this mutation's data
            split_conn_tracker[key] = dict()

            # New connection leading into the new node
            into_node = dict()
            into_node_key = conn_innov_counter
            conn_innov_counter += 1
            into_node["weight"] = 1
            into_node["conn"] = []
            into_node["conn"].append(old_gene["conn"][0])
            into_node["conn"].append(node_innov_counter)
            into_node["enable"] = True

            # New connection leaving out of the new node
            out_of_node = dict()
            out_of_node_key = conn_innov_counter
            conn_innov_counter += 1
            out_of_node["weight"] = old_gene["weight"]
            out_of_node["conn"] = []
            out_of_node["conn"].append(node_innov_counter)
            out_of_node["conn"].append(old_gene["conn"][1])
            out_of_node["enable"] = True

            # Update node counter
            node_innov_counter += 1

            # Keep track of split connections
            split_conn_tracker[key]["in_gene_key"] = into_node_key
            split_conn_tracker[key]["in_gene"] = deepcopy(into_node)
            split_conn_tracker[key]["out_gene_key"] = out_of_node_key
            split_conn_tracker[key]["out_gene"] = deepcopy(out_of_node)

        # Disable old_gene
        old_gene["enable"] = False

        # append new genes to genome
        genotype[into_node_key] = into_node
        genotype[out_of_node_key] = out_of_node

    finally:
        return conn_innov_counter, node_innov_counter


def random_new_connection(genotype, individual_input_keys, individual_output_keys, new_conn_tracker, conn_innov_counter):

    # Make a list of active node connections, to avoid duplicates
    en_conn_list = [g["conn"] for g in genotype.values() if g["enable"] is True]

    # Make a list of inactive node connections, to re-activate them instead of making a new one
    dis_conn_list = [g["conn"] for g in genotype.values() if g["enable"] is False]

    # Make a set of nodes that can be the input end of a connection (all nodes)
    in_node_keys = set()
    for g in genotype.values():
        in_node_keys.add(g['conn'][0])
        in_node_keys.add(g['conn'][1])
    in_node_keys.update(individual_input_keys)
    in_node_keys.difference_update(individual_output_keys)

    # Make a set of nodes that can be the output end of a connection (all nodes, except input nodes)
    out_node_keys = set()
    for g in genotype.values():
        out_node_keys.add(g['conn'][0])
        out_node_keys.add(g['conn'][1])
    out_node_keys.difference_update(individual_input_keys)

    # Turn in_node_keys and out_node_keys into lists
    in_node_numbers = list(in_node_keys)
    out_node_numbers = list(out_node_keys)

    """
    Every node(M) can connect to any other non_input node(N). Non_input nodes are allowed to
    connect to themselves. Random numbers following the former criteria are selected to be the
    ends of new connections. M*N is used as the maximum number of allowed attempts, but
    keeping in mind that pairs are randomly selected, there's no guarantee that every possible
    pair will be tested.
    """
    # Initialize pair with None
    pair = [None, None]
    max_attempts = len(in_node_numbers) * len(out_node_numbers)

    while True:

        # Randomly chose an in_node and out_node
        pair[0] = random.choice(in_node_numbers)
        pair[1] = random.choice(out_node_numbers)

        # Found an available pair. Exit the loop
        if pair not in en_conn_list:
            break

        max_attempts -= 1
        if max_attempts <= 0:
            # Too many failed attempts. Return with no new connection
            return conn_innov_counter

    if pair in dis_conn_list:
        # Existing but disabled connection, re-enable it
        gene = next((g for g in genotype.values() if g["conn"] == pair))
        gene["enable"] = True

    else:
        conn_tracker_key = tuple(pair)
        if conn_tracker_key in new_conn_tracker.keys():
            # This was already mutated in this generation. Keep the previous mutation

            new_conn = deepcopy(new_conn_tracker[conn_tracker_key]["gene"])
            new_conn_key = new_conn_tracker[conn_tracker_key]["gene_key"]

        else:
            # New connection, append it to genome
            new_conn = dict()

            new_conn_key = conn_innov_counter
            conn_innov_counter += 1
            new_conn["conn"] = pair
            new_conn["weight"] = 0
            new_conn["enable"] = True

            # Make a new dictionary associated with old_gene_key to store this mutation's data
            new_conn_tracker[conn_tracker_key] = dict()

            # Keep track of new connection
            new_conn_tracker[conn_tracker_key]["gene"] = deepcopy(new_conn)
            new_conn_tracker[conn_tracker_key]["gene_key"] = new_conn_key

        # Append new gene to genome
        genotype[new_conn_key] = new_conn

        # Randomize new gene weight
        randomize_gene(
            genotype=genotype,
            key=new_conn_key
        )

    return conn_innov_counter


# This is the compatibility formula from the paper
def calculate_comp_delta(individual1, individual2, ce=1, cw=1, normalize=False):

    # Get genotype's length
    n1 = len(individual1.genotype)
    n2 = len(individual2.genotype)

    if normalize:
        # Set n to be the largest genotype's length
        n = max(n1, n2)
    else:
        n = 1

    # Make a set with each genotype's innov numbers (in this case, innov numbers are the genes dictionary's keys)
    g1 = set()
    g1.update([g for g in individual1.genotype])

    g2 = set()
    g2.update([g for g in individual2.genotype])

    # Get the set of intersecting genes
    intersecting_genes = g1.intersection(g2)

    """
    The paper makes a distinction between excess and disjoint genes. This
    implementation considers both disjoint and excess genes as excess.
    """
    # Get the number of excess genes
    e = len(g1.union(g2)) - len(intersecting_genes)

    # Average weight difference of matching genes
    w = 0
    for i in intersecting_genes:
        w += abs(individual1.genotype[i]["weight"] - individual2.genotype[i]["weight"])
    try:
        w /= len(intersecting_genes)
    except ZeroDivisionError:
        # For practical purposes, 1/0 means infinity
        w = float('inf')

    delta = ((ce * e) / n) + (cw * w)

    return delta


# individuals have genotypes, and phenotypes(networks) generated by genotypes
class Individual:

    def __init__(self, input_keys, output_keys):
        self.id = None
        self.genotype = dict()

        self.birth_date = str(datetime.now())
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.fitness = 0
        self.shared_fitness = None
        self.lectotype_delta = None
        self.network = None
        self.generations_alive = 0
        self.allow_recurrency = False

        self.activation_function = None

    def disable_connections(self, connections):
        for c in connections:
            for g in self.genotype.values():
                if g['conn'] == c:
                    g['enable'] = False
                    break

    def build_network(self):
        self.network = nn.Network()

        # Add the input and output nodes first
        for k in self.input_keys:
            self.network.add_input_node(
                node_id=k,
                activation_function=self.activation_function
            )

        for k in self.output_keys:
            self.network.add_output_node(
                node_id=k,
                activation_function=self.activation_function
            )

        for g in self.genotype.values():
            if g["enable"]:
                # Both the input and output node need to exist in order to link them
                in_key = g["conn"][0]
                out_key = g["conn"][1]

                # Ensure the nodes exists
                if in_key not in self.network.nodes:
                    self.network.add_hidden_node(
                        node_id=in_key,
                        activation_function=self.activation_function
                    )

                if out_key not in self.network.nodes:
                    self.network.add_hidden_node(
                        node_id=out_key,
                        activation_function=self.activation_function
                    )

                self.network.add_connection(
                    weight=g["weight"],
                    in_node_key=in_key,
                    out_node_key=out_key
                )

        # Optional recurrency
        if not self.allow_recurrency:
            for n in self.network.nodes.values():
                broken_connections = nn.break_loops(n)
                self.disable_connections(broken_connections)



# Species have individuals
class Species:
    def __init__(self):
        self.parent1 = None
        self.parent2 = None
        self.individuals = []
        self.maximum_offspring = None
        self.shared_fitness = None
        self.maximum_fitness = 0
        self.no_improvements_counter = 0


def crossover(parent1, parent2):
    # If both parents are the same, just copy the genotype
    if parent1 is parent2:
        new_genotype = deepcopy(parent1.genotype)
    else:
        new_genotype = dict()

        # Ensure parent1 is the fittest parent
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1

        for g in parent1.genotype:
            # This will cycle trough the fittest parent's gene keys
            try:
                # Select a gene from a randomly selected parent
                new_genotype[g] = deepcopy(random.choice([parent1.genotype[g], parent2.genotype[g]]))
            except KeyError:
                # Key not found in least fitted parent. Defaults to the fittest parent gene
                new_genotype[g] = deepcopy(parent1.genotype[g])

    new_individual = Individual(
        input_keys=parent1.input_keys,
        output_keys=parent1.output_keys
    )

    new_individual.genotype = new_genotype

    return new_individual


# Populations have species
class Population:

    def __init__(self, input_keys, output_keys, population_size, activation_function=None):
        # Initialize a random seed for all the things that require randomness
        random.seed()  # No arguments, so it uses system time

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.population_size = population_size

        # Default values, these are meant to be set after Population is instanced
        self.excess_compatibility_coefficient = 1
        self.weight_compatibility_coefficient = 1
        self.compatibility_threshold = 1
        self.gene_mutation_probability = 0
        self.gene_perturbation_probability = 1
        self.gene_perturbation_delta = 0.01
        self.activation_function = activation_function
        self.allow_recurrency = False
        self.allow_elitism = True
        self.max_stagnant_generations = 0
        self.normalize_compatibility_calculation = False

        self.species_dict = dict()
        self.species_count = 0
        self.individual_counter = 0

        self.conn_innov_counter = 0
        self.node_innov_counter = 0

        # The whole population on first generation belongs to the same species
        self.make_new_species(population_size=self.population_size)

    def make_new_species(self, population_size):
        # Initial pattern for genotypes
        genotype_pattern = dict()
        for i in self.input_keys:
            for o in self.output_keys:
                """
                A dictionary is used to represent the genotype. Dictionary keys are
                equivalent to the original paper's innovation numbers. Connections
                start disabled just to register input and output nodes, they
                appear as needed
                """
                # TODO make this probability adjustable
                if chance(1):
                    genotype_pattern[self.conn_innov_counter] = {
                        "conn": [i, o],
                        "weight": 0,
                        "enable": True
                    }
                    self.conn_innov_counter += 1

        self.species_dict[self.species_count] = Species()

        for i in range(0, population_size):
            # Randomize initial gene weights
            for g in genotype_pattern:
                mutate_genotype(
                    genotype=genotype_pattern,
                    mutation_probability=1,
                    perturbation_delta=self.gene_perturbation_delta,
                    perturbation_probability=self.gene_perturbation_probability
                )

            # Make a new individual
            new_individual = Individual(
                input_keys=self.input_keys,
                output_keys=self.output_keys
            )
            # Give it an id
            new_individual.id = self.individual_counter
            self.individual_counter += 1
            # Assign the randomized genotype to the phenotype
            new_individual.genotype = deepcopy(genotype_pattern)
            # Set the individual's activation function
            new_individual.activation_function = self.activation_function
            # Set other individual parameters:
            new_individual.allow_recurrency = self.allow_recurrency
            # Add the individual to the species
            self.species_dict[self.species_count].individuals.append(new_individual)

        self.species_count += 1

    def build_networks(self):
        for s in self.species_dict.values():
            for i in s.individuals:
                i.build_network()

    def breed(self, in_chance, nc_chance):
        # Before doing anything else, shared fitness and offspring amount are calculated
        population_shared_fitness = 0
        for s in self.species_dict.values():
            s.shared_fitness = 0
            individuals_len = len(s.individuals)
            for i in s.individuals:
                """
                There's a formula in the paper used to calculate shared fitness,
                but it boils down to dividing each phenotype's fitness by the
                total number of phenotypes belonging in the same species.
                """
                i.shared_fitness = i.fitness / individuals_len
                s.shared_fitness += i.shared_fitness
                population_shared_fitness += i.shared_fitness

        for s in self.species_dict.values():
            s.maximum_offspring = round((s.shared_fitness * self.population_size) / population_shared_fitness)

        # Selection
        for s in self.species_dict.values():
            # Sort each species' individuals in descending order (Fittest phenotypes first)
            s.individuals.sort(key=lambda p: p.fitness, reverse=True)

            # Keep track of species fitness improvement
            if s.individuals[0].fitness > s.maximum_fitness:
                s.maximum_fitness = s.individuals[0].fitness
                s.no_improvements_counter = 0
            else:
                s.no_improvements_counter += 1

            # Select parents
            if len(s.individuals) > 1:
                s.individuals[0].generations_alive += 1
                s.parent1 = s.individuals[0]
                s.individuals[1].generations_alive += 1
                s.parent2 = s.individuals[1]
            else:
                # If there´s only one individual, it will be selected as parent twice, effectively creating clones
                s.individuals[0].generations_alive += 1
                s.parent1 = s.individuals[0]
                s.parent2 = s.individuals[0]

        offspring = []
        for s in self.species_dict.values():

            # Force extinct stagnant species:
            if self.max_stagnant_generations != 0 and s.no_improvements_counter > self.max_stagnant_generations:
                s.maximum_offspring = 0

            if self.allow_elitism:
                # Keep the fittest phenotypes
                if s.maximum_offspring > 1:
                    offspring.append(s.parent1)
                    offspring.append(s.parent2)
                    s.maximum_offspring -= 2

            if s.maximum_offspring < 0:
                s.maximum_offspring = 0

            # Generate new individuals and mutate them

            # A node appears when a connection (innovation) is split
            split_conn_tracker = dict()
            # A connection appears between nodes (each node will be present on a set of innovations)
            new_conn_tracker = dict()

            for i in range(0, s.maximum_offspring):
                # Crossover
                new_individual = crossover(s.parent1, s.parent2)
                new_individual.id = self.individual_counter
                self.individual_counter += 1
                new_individual.activation_function = self.activation_function
                new_individual.allow_recurrency = self.allow_recurrency

                # Mutation

                """
                Keep track of current generation mutations/innovations(add connection, split connection)
                to avoid repeating them by chance and having different innovation numbers for the
                same mutation. Lists are used to keep track of mutations reset before every new
                generation
                """

                mutate_genotype(
                    genotype=new_individual.genotype,
                    mutation_probability=self.gene_mutation_probability,
                    perturbation_delta=self.gene_perturbation_delta,
                    perturbation_probability=self.gene_perturbation_probability
                )
                if chance(in_chance):
                    self.conn_innov_counter, self.node_innov_counter = random_insert_node(
                        genotype=new_individual.genotype,
                        split_conn_tracker=split_conn_tracker,
                        conn_innov_counter=self.conn_innov_counter,
                        node_innov_counter=self.node_innov_counter
                    )
                if chance(nc_chance):
                    self.conn_innov_counter = random_new_connection(
                        genotype=new_individual.genotype,
                        individual_input_keys=new_individual.input_keys,
                        individual_output_keys=new_individual.output_keys,
                        new_conn_tracker=new_conn_tracker,
                        conn_innov_counter=self.conn_innov_counter
                    )

                # Add new individual to offspring list
                offspring.append(new_individual)

        # Species classification
        for s in self.species_dict.values():
            # Clear individuals
            s.individuals = []

        for child in offspring:
            # Default to create a new species
            new_species = True

            for s in self.species_dict.values():
                # The fittest parent is taken as lectotype
                child.lectotype_delta = calculate_comp_delta(
                    individual1=s.parent1,
                    individual2=child,
                    ce=self.excess_compatibility_coefficient,
                    cw=self.weight_compatibility_coefficient,
                    normalize=self.normalize_compatibility_calculation
                )

                if child.lectotype_delta < self.compatibility_threshold:
                    # Children are assigned to the first species in which they fit
                    new_species = False
                    s.individuals.append(child)
                    break

            if new_species is True:
                # If they don't fit into any species, they are the first of a new one
                self.species_dict[self.species_count] = Species()

                self.species_dict[self.species_count].individuals.append(child)
                # The new phenotype is considered to be both parents of this new species
                self.species_dict[self.species_count].parent1 = child
                self.species_dict[self.species_count].parent2 = child

                self.species_count += 1

        # Remove empty species:
        extinct_keys = []
        for s in self.species_dict:
            if self.species_dict[s].individuals == []:
                extinct_keys.append(s)

        for k in extinct_keys:
            del (self.species_dict[k])

        # Ensure there´s at least one species left
        if len(self.species_dict) == 0:
            self.make_new_species(population_size=self.population_size)