"""
Polygons are stored in a generations list.

Each generation (list element) is a quads dictionary. Each value is a dictionary
containing the quad itself and how many individuals belong to the species it
represents. Keys are the id's of the species being graphically represented.

There's a separate dictionary to store species color, using species' id as keys.

When a new species dictionary is assigned to the dict property, a new quads
dictionary is created using the incoming information and appended to generations list.

For every species in any given generation, a new quad is created. Its points are determined as
follows:

The x component from all four quad points are assigned first. This is to place
all y component calculations inside a function, since they will have to be
re-calculated every time the widget is resized or the maximum registered population changes.

The top and bottom left y components are the same as the top and bottom right
from that species' last generation quad.

The bottom right y component is the same as the top right y component from this
generation's last added quad (previous species)

Species pixel size is calculated by taking this species size, and adjusting it
in proportion to the maximum registered population size and widget height. (The
generation that holds the maximum population size will reach the top of the
widget)

The top right y is the bottom right y, plus the pixel size of the species it represents.
"""


from kivy.app import App
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scatter import  Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.graphics import Color, Rectangle, Quad, Line
from kivy.clock import Clock, mainthread

from random import uniform
from kivy.properties import DictProperty, ObjectProperty
from copy import deepcopy
from collections import namedtuple, deque


def get_px_size(px_height, size, max_size):
    """
    pixel size is calculated by taking size, and adjusting it
    in proportion to the provided maximum size and pixel height.

    :param px_height:
        Maximum allowed size, in pixels
    :param size:
        Current size, integer
    :param max_size:
        Maximum allowed size, integer
    :return:
        Current size, in pixels
    """
    return int(size * px_height / max_size)


class EvolutionGraph(Widget):
    """
    Every new generation is represented by a dictionary:
    key: Species id
    value: How many individuals belong to that species
    """

    def __init__(self, *args, **kwargs):
        super(EvolutionGraph, self).__init__(**kwargs)

        self.x_step = 20

        self.clear()

        self.bind(pos=self.adjust_canvas, height=self.adjust_canvas)

    def draw_species(self):

        request_canvas_adjust = False

        while len(self.pending_generations) > 0:

            new_gen = dict()
            species = self.pending_generations.pop()

            # Grow the widget as new species are added
            present_x = self.last_x + self.x_step
            self.width = present_x

            # Sort keys in descending order
            sorted_keys = sorted(species.keys())

            # Add quads according to incoming species dictionary
            last_s_key = None

            for s_key in sorted_keys:
                if s_key not in self.species_hsv:
                    opposite_hue = self.last_hue + 0.5
                    hue_range = 0.25
                    new_hue = opposite_hue + uniform(-hue_range, hue_range)

                    if new_hue > 1:
                        new_hue -= 1
                    if new_hue < 0:
                        new_hue += 1

                    self.last_hue = new_hue

                    new_hsv = (
                        new_hue,
                        1,
                        0.5
                    )

                    self.species_hsv[s_key] = new_hsv

                """
                Points are defined in the following order:
                Top left, Bottom left, Bottom right, Top right
                Y components will be assigned later
                """
            sorted_keys.reverse()
            for s_key in sorted_keys:
                s_val = species[s_key]
                quad_points = [
                    self.last_x, None,
                    self.last_x, None,
                    present_x, None,
                    present_x, None
                ]

                prev_hor = None
                try:
                    if s_key in self.generations[-1]:
                        prev_hor = self.generations[-1][s_key]
                except IndexError:
                    pass

                prev_vert = None
                try:
                    prev_vert = new_gen[last_s_key]
                except KeyError:
                    None
                except TypeError:
                    None

                self.calculate_vertical_components(
                    points=quad_points,
                    prev_horizontal=prev_hor,
                    prev_vertical=prev_vert,
                    px_size=get_px_size(
                        px_height=self.height,
                        size=s_val,
                        max_size=self.max_size
                    )
                )

                quad_info = {
                    'quad': None,
                    'size': s_val,
                    'prev_hor': prev_hor,
                    'prev_vert': prev_vert
                }

                with self.canvas:
                    Color(*self.species_hsv[s_key], mode='hsv')
                    quad_info['quad'] = Quad()

                quad_info['quad'].points = quad_points

                new_gen[s_key] = quad_info
                last_s_key = s_key

            with self.canvas:
                Color(1, 1, 1, 0.3, mode='rgba')
                self.vert_lines.append(
                    Line(points=[self.last_x, 0, self.last_x, self.height], width=1.0)
                )

            self.generations.append(new_gen)
            self.last_x = present_x

            # Update maximum, if required
            population = sum(list(species.values()))
            if population > self.max_size:
                self.max_size = population
                request_canvas_adjust = True

        if request_canvas_adjust:
            self.adjust_canvas()

    def adjust_canvas(self, *args):
        print(uniform(0, 1))
        if len(self.generations) > 0:
            for g in self.generations:
                for k in g.keys():
                    p = g[k]['quad'].points
                    self.calculate_vertical_components(
                        points=p,
                        prev_horizontal=g[k]['prev_hor'],
                        prev_vertical=g[k]['prev_vert'],
                        px_size=get_px_size(
                            px_height=self.height,
                            size=g[k]['size'],
                            max_size=self.max_size
                        )
                    )
                    # Points is a kivy list property, list assignment is required to trigger the update
                    g[k]['quad'].points = p


            for l in self.vert_lines:
                points = l.points
                points[3] = self.height
                l.points = points
        self.canvas.ask_update()

    def clear(self):
        self.canvas.clear()
        self.generations = list()
        self.vert_lines = list()
        self.species_hsv = dict()
        self.width = 0
        self.max_size = 1
        self.last_x = 0
        self.last_hue = uniform(0,1)

        self.pending_generations = deque()

    def calculate_vertical_components(self, points, prev_horizontal, prev_vertical, px_size):
        """
        Calculates the Y components, based previous sets of points and a
        given px size

        Points are defined in the following order:
        Top left, Bottom left, Bottom right, Top right
        X components were previously assigned
        """

        if prev_horizontal is None:
            # It's a new species, so it starts at the bottom
            points[1] = 0
            points[3] = 0
        else:
            points[1] = prev_horizontal['quad'].points[7]
            points[3] = prev_horizontal['quad'].points[5]

        if prev_vertical is None:
            # It's the first on generation, so it starts at the bottom
            points[5] = 0
            points[7] = px_size
        else:
            points[5] = prev_vertical['quad'].points[7]
            points[7] = prev_vertical['quad'].points[7] + px_size


class EvolutionMonitor(ScrollView):
    def __init__(self, *args, **kwargs):
        super(EvolutionMonitor, self).__init__(**kwargs)
        self.graph = EvolutionGraph()
        self.graph.size_hint_x = None
        self.graph.size_hint_y = 0.95
        self.add_widget(self.graph)

    def add_species(self, species):
        self.graph.pending_generations.appendleft(species)

    def draw_species(self):
        self.graph.draw_species()

    def clear(self):
        self.graph.clear()

    def add_dummy_species(self, *args):
        dummy_hist = [
            {0: 150},
            {0: 100, 1: 30, 2: 20},
            {0: 70, 1: 50, 2: 30},
            {0: 40, 1: 30, 2: 20, 3: 60},
            {0: 30, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 140},
            {0: 30, 2: 40, 3: 110},
            {0: 90, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 110},
            {0: 30, 2: 10, 3: 110},
        ]

        for d in dummy_hist:
            self.add_species(d)

if __name__ == '__main__':

    class MainApp(App):
        def build(self):

            rootwidget = EvolutionMonitor()
            rootwidget.scroll_type = ['bars', 'content']
            rootwidget.bar_width = 20

            Clock.schedule_once(rootwidget.add_dummy_species, 1)

            return rootwidget

    MainApp().run()

