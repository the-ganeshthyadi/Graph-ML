import matplotlib.pyplot as plt
import numpy as np

class MLGraph:
    def __init__(self, title="Graph"):
        self.title = title
        self.background_color = "#ffffff"
        self.fig, self.ax = plt.subplots()

    def set_background(self, color):
        self.background_color = color
        self.ax.set_facecolor(color)

    def set_labels(self, x_label="", y_label=""):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def plot(self, X, Y, colors=None, linewidth=2, markers=None):
        if colors is None:
            colors = ["#1f77b4"]
        if markers is None:
            markers = ['o'] * len(X)

        for i, y in enumerate(Y):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            self.ax.plot(X, y, color=color, linewidth=linewidth, marker=marker)

    def scatter(self, X, Y, colors=None, markers=None):
        if colors is None:
            colors = ["#1f77b4"]
        if markers is None:
            markers = ['o'] * len(X)

        for i, y in enumerate(Y):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            self.ax.scatter(X, y, color=color, marker=marker)

    def save_graph(self, filename):
        self.ax.set_title(self.title)
        self.fig.savefig(filename)

    def show(self):
        plt.show()
