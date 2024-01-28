import matplotlib.pyplot as plt
import numpy as np

from reliability.Probability_plotting import Weibull_probability_plot, plot_points, plotting_positions

# failure data from oil pipe corrosion
bend = [74, 52, 32, 76, 46, 35, 65, 54, 56, 20, 71, 72, 38, 61, 29]
valve = [78, 83, 94, 76, 86, 39, 54, 82, 96, 66, 63, 57, 82, 70, 72, 61, 84, 73, 69, 97]
joint = [74, 52, 32, 76, 46, 35, 65, 54, 56, 25, 71, 72, 37, 61, 29]

# combine the data into a single array
data = np.hstack([bend, valve, joint])
color = np.hstack([["red"] * len(bend), ["green"] * len(valve), ["blue"] * len(joint)])

# create the probability plot and hide the scatter points
Weibull_probability_plot(failures=data, show_scatter_points=False)

# redraw the scatter points. kwargs are passed to plt.scatter so a list of color is accepted
plot_points(failures=data, color=color, marker="^", s=100)

# To show the legend correctly, we need to replot some points in separate scatter plots to create different legend entries
x, y = plotting_positions(failures=data)
plt.scatter(x[0], y[0], color=color[0], marker="^", s=100, label="bend")
plt.scatter(x[len(bend)], y[len(bend)], color=color[len(bend)], marker="^", s=100, label="valve")
plt.scatter(
    x[len(bend) + len(valve)],
    y[len(bend) + len(valve)],
    color=color[len(bend) + len(valve)],
    marker="^",
    s=100,
    label="joint",
)
plt.legend()

plt.show()
