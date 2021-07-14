from time import sleep


def print_node_info(network):
    for n in network.nodes.values():
        print("{}: {}".format(
            n.id,
            list(zip([i.id for i in n.inputs], n.conn_weights))
        ))


def str_node_info(network):
    ret_str = ""
    for n in network.nodes.values():
        ret_str += "\n" + ("{}: {}".format(
            n.id,
            list(zip([i.id for i in n.inputs], n.conn_weights))
        ))

    return ret_str


def dict_node_info(network):
    nodes = dict()
    for n in network.nodes.values():
        nodes[n.id] = list(zip([i.id for i in n.inputs], n.conn_weights))

    return nodes

"""
Return a list of dictionaries, each dictionary representing a layer and using
node IDs as keys and nodes as values.

The first dictionary in list is is the input layer, subsequent dictionaries
represent hidden layers (if any) and the final dictionary is the output layer.
"""


def layerize_network(network):
    # Keep track of processed nodes by adding their ID to a set
    layerized_nodes_keys = set()
    """# The first layer is always only the input nodes
    input_layer = {k: network.nodes[k] for k in network.input_nodes_keys}"""
    # The first layer is all the nodes that have no inputs
    input_layer = {k: network.nodes[k] for k in network.nodes if network.nodes[k].inputs == []}
    layerized_nodes_keys.update(input_layer.keys())

    """
    Sometimes, the input layer will have nodes that are not inputs. That´s because, trough mutation,
    these nodes are left with no input of their own. Implement pointless node removal in
    neural_network
    """
    # Last layer is always only the output nodes
    output_layer = {k: network.nodes[k] for k in network.output_nodes_keys}
    layerized_nodes_keys.update(output_layer.keys())

    # Initialize hidden layer list as empty
    hidden_layers = list()
    # Initialize layerized nodes keys set with input and output layer keys
    layerized_keys_set = set(list(input_layer.keys()) + list(output_layer.keys()))
    # Initialize non-layerized node keys set with all keys, minus layerized keys set
    non_layerized_keys_set = (set(list(network.nodes.keys())) - layerized_keys_set)
    # Initialize previous layer keys set with input layer keys
    previous_layer_keys_set = set(list(input_layer.keys()))
    while len(non_layerized_keys_set) > 0:
        # Initialize this layer keys set as empty
        this_layer_keys_set = set()
        for k in non_layerized_keys_set:
            # Make a set with this node input's keys
            node_input_keys_set = set([i.id for i in network.nodes[k].inputs])
            # Check if any of the input keys belong to the previous layer keys set
            if len(node_input_keys_set.intersection(previous_layer_keys_set)) > 0:
                # add its key to this layer keys set
                this_layer_keys_set.add(k)
                # add its key to layerized node keys set
                layerized_keys_set.add(k)
        # if this layer keys set is empty, add all non-layerized node to the layer keys set and layerized keys set
        if len(this_layer_keys_set) == 0:
            this_layer_keys_set.update(non_layerized_keys_set)
            layerized_keys_set.update(non_layerized_keys_set)
        # Remove this layer keys set from non-layerized node keys set
        non_layerized_keys_set.difference_update(this_layer_keys_set)
        # Make a dictionary wit this layer keys and their respective nodes
        this_layer = {k: network.nodes[k] for k in this_layer_keys_set}
        # Append this layer dictionary to hidden layers list
        hidden_layers.append(this_layer)
        # Make previous layer keys be the keys of this layer, for the next iteration
        previous_layer_keys_set = this_layer_keys_set

    # Assemble layers list and return
    return [input_layer] + hidden_layers + [output_layer]


def get_connections(network):
    # Start with an empty set
    conn_set = set()

    for n in network.nodes.values():
        for i in n.inputs:
            # Make a tuple with input ID and node ID
            conn = (i.id, n.id)
            # Add tupple to set
            conn_set.add(conn)

    # Make a list with connections set and return
    return list(conn_set)

