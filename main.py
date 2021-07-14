from kivy.app import App
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, ListProperty, DictProperty, NumericProperty, StringProperty
from kivy.clock import Clock, mainthread
import threading
from sys import setswitchinterval
from kivy.core.text import Label as CoreLabel
import evolutionmonitor


def new_text_texture(text, font_size):
    # https://groups.google.com/forum/#!topic/kivy-users/zRCjfhBcX4c
    label = CoreLabel(
        text=text,
        font_size=font_size,
    )
    label.refresh()
    texture = label.texture
    return texture

from random import choice, uniform

from xor import NeatXor
from neural_network_display import layerize_network, get_connections


class NodeWidget(RelativeLayout):
    pos_hint = DictProperty(defaultvalue={
        'center_x': 0,
        'center_y': 0
    })

    lbl = ObjectProperty(Label(text='hi'))
    color = ListProperty([0, 1, 1, 0.5])
    txt = StringProperty("hi")

    def __init__(self, *args, **kwargs):
        super().__init__()

        if 'node' in kwargs:
            self.txt = str(kwargs['node'])
        if 'color' in kwargs:
            self.color = kwargs['color']


class ConnWidget(RelativeLayout):
    # TODO make recurrent connections to self a round arrow that returns, or something like that

    start_x = NumericProperty()
    inter_x = NumericProperty()
    end_x = NumericProperty()

    start_y = NumericProperty()
    inter_y = NumericProperty()
    end_y = NumericProperty()

    start_node_widget = ObjectProperty(None)
    end_node_widget = ObjectProperty(None)

    def __init__(self):
        super().__init__()
        self.bind(pos=self.update_conn_points)
        self.bind(size=self.update_conn_points)
        self.bind(start_node_widget=self.update_conn_points)
        self.bind(end_node_widget=self.update_conn_points)

    def update_conn_points(self, *args):
        if (self.start_node_widget is not None) and (self.end_node_widget is not None):
            # Set size hints based on start and end hints
            self.size_hint_x = abs(
                self.start_node_widget.pos_hint['center_x'] - self.end_node_widget.pos_hint['center_x']
            )

            self.size_hint_y = abs(
                self.start_node_widget.pos_hint['center_y'] - self.end_node_widget.pos_hint['center_y']
            )

            # Determine widget position depending on start and end
            self.pos_hint = {
                'x': min(
                    self.start_node_widget.pos_hint['center_x'],
                    self.end_node_widget.pos_hint['center_x']
                ),
                'y': min(
                    self.start_node_widget.pos_hint['center_y'],
                    self.end_node_widget.pos_hint['center_y']
                )
            }

            self.start_x = 0
            self.end_x = self.width
            if self.start_node_widget.pos_hint['center_x'] > self.end_node_widget.pos_hint['center_x']:
                self.start_x, self.end_x = self.end_x, self.start_x

            self.start_y = 0
            self.end_y = self.height
            if self.start_node_widget.pos_hint['center_y'] > self.end_node_widget.pos_hint['center_y']:
                self.start_y, self.end_y = self.end_y, self.start_y

            self.inter_x = self.start_x + ((self.end_x - self.start_x) * 0.1)
            self.inter_y = self.start_y + ((self.end_y - self.start_y) * 0.1)


class NetworkMonitor(RelativeLayout):
    network = ObjectProperty(None)

    node_widgets = dict()
    conn_widgets = list()

    def on_network(self, instance, value):
        # Clear widgets to draw a new network:
        self.clear_widgets()
        self.node_widgets = dict()
        self.conn_widgets = list()
        # Organize nodes by layers (including input and output layers)
        layers = layerize_network(self.network)
        # Get a list of node connections
        connections = get_connections(self.network)
        # Determine network depth
        network_depth = len(layers)
        # Find the widest layer, call it network_width
        network_width = 0
        for l in layers:
            network_width = max(len(l), network_width)
        # Network depth will be horizontal, network width will be vertical
        # Calculate node size_hint_x required to fit all layers given the widget's width
        size_hint_x = (1 / network_depth) * 0.25
        # Calculate node size_hint_y required to fit the widest layer given the widget's height
        size_hint_y = (1 / network_width) * 0.25
        # Use the smallest from size_hint_x or size_hint_y as node diameter

        x_hint_inc = 1 / (network_depth + 1)
        x_hint = x_hint_inc
        for l in layers:
            y_hint_inc = 1 / (len(l) + 1)
            y_hint = y_hint_inc
            for n in l:
                # Create node widget
                if n in self.network.input_nodes_keys:
                    color = (0, 1, 0, 0.5)
                    y_offset = 0
                    x_offset = 0
                elif n in self.network.output_nodes_keys:
                    color = (0, 0.5, 1, 0.5)
                    y_offset = 0
                    x_offset = 0
                else:
                    color = (0, 1, 1, 0.5)
                    y_offset = uniform(y_hint_inc * 0.05, y_hint_inc * 0.1)
                    x_offset = uniform(x_hint_inc * 0.05, x_hint_inc * 0.1)

                node = Factory.NodeWidget(node=n, color=color)

                # Set x and y pos hint depending on network depth and width
                node.pos_hint = {
                    'center_x': x_hint + choice([x_offset, -x_offset]),
                    'center_y': y_hint + choice([y_offset, -y_offset])
                }
                node.size_hint = (size_hint_x, size_hint_y)
                # Add to node_widgets dict
                self.node_widgets[n] = node

                y_hint += y_hint_inc

            x_hint += x_hint_inc

        for c in connections:
            conn = Factory.ConnWidget()

            # Set start and end node
            conn.start_node_widget = self.node_widgets[c[0]]
            conn.end_node_widget = self.node_widgets[c[1]]

            # Add to conn_widgets list
            self.conn_widgets.append(conn)

        # Add connection widgets to represent connections and their activation state
        for c in self.conn_widgets:
            self.add_widget(c)
            pass
        # Add node widgets to represent nodes and their current output value
        for n in self.node_widgets.values():
            self.add_widget(n)


class XorAppLayout(BoxLayout):

    stop = threading.Event()
    gen_index_lock = threading.Lock()
    # Set thread switching interval. Small values seem to keep GUI responsive. Big values seem to make computing faster.
    setswitchinterval(1e-4)

    test = NeatXor()
    generations = []
    gen_index = -1

    textMonitor = ObjectProperty(None)
    networkMonitor = ObjectProperty(None)
    evolutionMonitor = ObjectProperty(None)

    neat_thread = threading.Thread()

    update_monitor_event = None

    def start_neat_thread(self, *args):
        self.neat_thread = threading.Thread(target=self.neat_thread_fn)
        self.neat_thread.start()

    def neat_thread_fn(self):
        exit_loop = False
        while not exit_loop:
            # Prevent thread to keep running after window is closed
            if self.stop.is_set():
                break

            # Do the important things
            exit_loop, monitor_str, species_stats = self.test.run_generation()

            # Do not change gen_index if it is not pointing to the last generation available
            if self.gen_index == (len(self.generations)-1):
                with self.gen_index_lock:
                    self.gen_index += 1

            self.generations.append(
                {
                    'str': monitor_str,
                    'nw': self.test.fittest_individual.network
                }
            )

            self.evolutionMonitor.add_species(species_stats)

    def set_gen_index(self, new_index):
        with self.gen_index_lock:
            max_index = len(self.generations) - 1
            self.gen_index = new_index
            if self.gen_index < 0:
                self.gen_index = 0
            elif self.gen_index > max_index:
                self.gen_index = max_index
        Clock.schedule_once(self.update_monitor, 0)

    @mainthread
    def update_monitor(self, *args):
        if len(self.generations) > 0:
            with self.gen_index_lock:
                gen = self.generations[self.gen_index]
                self.textMonitor.text = gen['str']
                self.networkMonitor.network = gen['nw']
                self.evolutionMonitor.draw_species()

    def reset_evolution(self):
        # Stop neat thread, if running
        if self.neat_thread.isAlive():
            self.stop.set()
            while self.neat_thread.is_alive():
                pass

        # Cancel monitor update
        self.update_monitor_event.cancel()

        # Reset as required
        self.test = NeatXor()
        self.generations = []
        self.gen_index = -1
        self.species_history = []
        self.evolutionMonitor.clear()

        # Clear the stop event
        self.stop.clear()
        # Start neat thread
        self.start_neat_thread()

        # Re-schedule monitor update
        self.update_monitor_event = Clock.schedule_interval(self.update_monitor, 0.5)


class XorKivyApp(App):
    def on_stop(self):
        self.root.stop.set()

    def build(self):
        layout = XorAppLayout()
        Clock.schedule_once(layout.start_neat_thread, 0)
        layout.update_monitor_event = Clock.schedule_interval(layout.update_monitor, 0.5)
        return layout


if __name__ == '__main__':
    XorKivyApp().run()