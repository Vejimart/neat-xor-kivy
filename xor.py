import neat


def activation_function(number):
    # return 1 / (1 + exp(-4.9*number))
    return number / (1 + abs(number)) # This one seems to be working better


def update_individual(inputs, individual):
    # Flush network nodes, to avoid memorization
    individual.network.flush()
    # Set network inputs
    individual.network.set_inputs(inputs)
    """# Activate 2 times, to prevent deep networks (this problem shouldn't require them anyway)
    for r in range(0, 2):
        individual.network.activate()"""
    # Activate as many times as there are nodes
    for r in range(0, len(individual.network.nodes)):
        individual.network.activate()


class NeatXor:
    def __init__(self):
        # XOR has two inputs, but the first input will be the bias node.
        self.p = neat.Population(
            input_keys=[
                "BIAS",
                "IN_A",
                "IN_B"
            ],
            output_keys=["OUT"],
            # population_size=150,
            population_size=1500,
            activation_function=activation_function
        )

        # NEAT parameters
        """
        self.p.excess_compatibility_coefficient = 1
        self.p.weight_compatibility_coefficient = 0.4
        self.p.compatibility_threshold = 3

        self.p.gene_mutation_probability = 0.8
        self.p.gene_perturbation_probability = 0.9
        self.p.gene_perturbation_delta = 0.1

        self.p.max_stagnant_generations = 15
        self.p.normalize_compatibility_calculation = False
        """
        self.p.excess_compatibility_coefficient = 1
        self.p.weight_compatibility_coefficient = 0.4
        self.p.compatibility_threshold = 3.8

        self.p.gene_mutation_probability = 0.8
        self.p.gene_perturbation_probability = 0.9
        self.p.gene_perturbation_delta = 0.1

        self.p.max_stagnant_generations = 15
        self.p.normalize_compatibility_calculation = False

        # Network parameters:
        self.p.allow_recurrency = False

        self.eval_inputs = [
            {"BIAS": 1, "IN_A": 0, "IN_B": 0},
            {"BIAS": 1, "IN_A": 0, "IN_B": 1},
            {"BIAS": 1, "IN_A": 1, "IN_B": 0},
            {"BIAS": 1, "IN_A": 1, "IN_B": 1}
        ]
        self.eval_outputs = [0, 1, 1, 0]

        self.gen_counter = 0
        self.solved = False
        self.fittest_individual = None

    def evaluate(self):
        # Networks have to be built before being evaluated
        self.p.build_networks()

        for s in self.p.species_dict.values():
            for i in s.individuals:

                i.fitness = 0

                for ins, out in zip(self.eval_inputs, self.eval_outputs):
                    update_individual(
                        inputs=ins,
                        individual=i
                    )
                    # Get the output
                    value = i.network.get_outputs()["OUT"]
                    # Calculate how far it was from the right answer
                    value = 1 - abs(value - out)
                    i.fitness += value
                # Prevent negative fitness from becoming positive after sqaring it.
                if i.fitness >= 0:
                    i.fitness = i.fitness ** 2
                else:
                    i.fitness = 0

    def run_generation(self):
        self.gen_counter += 1

        monitor_str = ""

        self.evaluate()
        top_fitness = 0
        individuals_counter = 0
        species_counter = 0

        for s in self.p.species_dict.values():
            species_counter += 1
            for i in s.individuals:
                individuals_counter += 1
                top_fitness = max(top_fitness, i.fitness)
                if top_fitness == i.fitness:
                    self.fittest_individual = i

        monitor_str += ("Generation {}".format(self.gen_counter))
        monitor_str += '\n' + ("Top fitness: {}".format(top_fitness))
        monitor_str += '\n' + ("Species: {}".format(species_counter))
        monitor_str += '\n' + ("Species counter: {}".format(self.p.species_count))
        monitor_str += '\n' + ("phenotypes: {}".format(individuals_counter))
        monitor_str += '\n' + ("Fittest phenotype: {}".format(self.fittest_individual.id))
        monitor_str += '\n' + ("Fittest phenotype alive time: {}".format(self.fittest_individual.generations_alive))
        monitor_str += '\n' + ("Fittest phenotype genome length: {}".format(len(self.fittest_individual.genotype)))
        monitor_str += '\n' + ("Fittest phenotype active genes: {}".format(
            len([g for g in self.fittest_individual.genotype.values() if g["enable"] is True])
        ))
        monitor_str += '\n' + ("Fittest phenotype nodes: {}".format(len(self.fittest_individual.network.nodes)))

        right_output_counter = 0
        monitor_str += '\n\n' + ("Outputs:")
        for ins, out in zip(self.eval_inputs, self.eval_outputs):
            update_individual(
                inputs=ins,
                individual=self.fittest_individual
            )
            raw_out = self.fittest_individual.network.get_outputs()["OUT"]
            round_out = round(raw_out)
            if round_out == out:
                right_output_counter += 1
            monitor_str += '\n' + ("{}: {}({})".format([i for i in ins.values()], round_out, raw_out))

        # monitor_str += '\n' + str_node_info(self.fittest_individual.network)

        # Store generation information before breeding
        species_stats = {k: len(self.p.species_dict[k].individuals) for k in self.p.species_dict}

        self.p.breed(
            in_chance=0.03,
            nc_chance=0.05
        )

        if right_output_counter == 4:
            self.solved = True

        return self.solved, monitor_str, species_stats

