from math import exp
from copy import deepcopy


class Node:

    @staticmethod
    def activation_function(number):
        return 1 / (1 + exp(-number))

    def __init__(self):
        self.output = 0
        self.inputs = []
        self.conn_weights = []
        self.input_values = []
        self.id = None

        self.dependant_nodes = []

    # Save the present value of outputs
    def pre_activate(self):
        self.input_values = []
        for i in self.inputs:
            self.input_values.append(i.output)

    # Update node output
    def activate(self):
        """
        Empty list means it's an input node. It´s output will be directly assigned, so no calculation required
        """
        if self.conn_weights != []:
            self.output = 0

            for z in zip(self.input_values, self.conn_weights):
                self.output += (z[0] * z[1])

            self.output = self.activation_function(self.output)

# Returns a set containig the original node and all of the nodes that have influence on its output
def get_influencing_nodes(node, influencing_nodes_set, visited_nodes_set):

    if node.id in visited_nodes_set:
        # Return to avoid a recursive loop
        return influencing_nodes_set

    # Add the present node id to visited nodes set
    visited_nodes_set.add(node.id)
    # Add the present node id to influencing nodes set
    influencing_nodes_set.add(node.id)

    for n in node.inputs:

        # Get influencing nodes for the present input node
        get_influencing_nodes(
            node=n,
            influencing_nodes_set=influencing_nodes_set,
            visited_nodes_set=visited_nodes_set.copy()
        )

    return influencing_nodes_set


def break_loops(node):
    delete_indexes = []
    index = 0

    for n in node.inputs:
        # Get influencing nodes for the present input node, including itself
        influencing_nodes = get_influencing_nodes(
            node=n,
            influencing_nodes_set=set(),
            visited_nodes_set=set()
        )
        """
        If an input node is influenced by the main node being checked, there´s a loop.
        Register it for removal from main node's input list. That will break the loop
        """
        if node.id in influencing_nodes:
            # There's a loop
            delete_indexes.append(index)

        index += 1

    # Delete them in reverse, because every removed item will cause the following to move to the previous index.
    delete_indexes.reverse()
    broken_connections = []
    for i in delete_indexes:
        # Main node will no longer depend on the node being disconnected
        node.inputs[i].dependant_nodes.remove(node)
        # Append broken connection to list
        broken_connections.append([node.inputs[i].id, node.id])
        # Remove connection and its weight
        del(node.inputs[i])
        del(node.conn_weights[i])

    return broken_connections


class Network:

    def __init__(self):
        self.input_nodes_keys = []
        self.output_nodes_keys = []

        # Using a dict allows to "name" the nodes, unlike a list in which they are identified by their index
        self.nodes = dict()

    def add_hidden_node(self, node_id, activation_function=None):
        self.nodes[node_id] = Node()
        self.nodes[node_id].id = node_id
        if activation_function is not None:
            self.nodes[node_id].activation_function = activation_function

    def add_input_node(self, node_id, activation_function=None):
        self.nodes[node_id] = Node()
        self.nodes[node_id].id = node_id
        if activation_function is not None:
            self.nodes[node_id].activation_function = activation_function
        self.input_nodes_keys.append(node_id)

    def add_output_node(self, node_id, activation_function=None):
        self.nodes[node_id] = Node()
        self.nodes[node_id].id = node_id
        if activation_function is not None:
            self.nodes[node_id].activation_function = activation_function
        self.output_nodes_keys.append(node_id)

    def add_connection(self, weight, in_node_key, out_node_key):
        # out_node_key is the node at the destination side of the connection
        # in_node_key is the node at the origin side of the connection

        # Prevent connections between non-existent nodes
        if in_node_key not in self.nodes.keys():
            return
        if out_node_key not in self.nodes.keys():
            return

        # Prevent duplicate connections
        if self.nodes[in_node_key] in self.nodes[out_node_key].inputs:
            return

        self.nodes[out_node_key].inputs.append(self.nodes[in_node_key])
        self.nodes[out_node_key].conn_weights.append(weight)
        self.nodes[in_node_key].dependant_nodes.append(self.nodes[out_node_key])

    # Directly set value of input nodes
    def set_inputs(self, inputs):
        for key in self.input_nodes_keys:
            self.nodes[key].output = inputs[key]

    # Get a list with the last output of output nodes
    def get_outputs(self):
        return {key: self.nodes[key].output for key in self.output_nodes_keys}

    def activate(self):
        # Pre_activation
        for n in self.nodes.values():
            n.pre_activate()

        # Update node outputs
        for n in self.nodes.values():
            n.activate()

    # Sets all node outputs to zero. This can be useful to prevent memorization
    def flush(self):
        for n in self.nodes.values():
            n.output = 0

