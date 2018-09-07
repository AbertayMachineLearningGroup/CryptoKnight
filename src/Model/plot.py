import matplotlib.pyplot as plt
import numpy as np
import os

class Plot:
	number_of_figures = 0
	figures = []
	figure_names = []
	axis = []
	x = []
	y = []
		
	def create(self, title, x_label, y_label, name):
		self.number_of_figures += 1
		self.figure_names.append(name)
		new_fig = self.figures.append("fig"+str(self.number_of_figures))
		new_ax = self.axis.append("ax"+str(self.number_of_figures))
		new_fig = plt.figure()
		new_fig.suptitle(title, fontsize=14, fontweight='bold')
		new_ax = new_fig.add_subplot(111)
		new_fig.subplots_adjust(top=0.85)
		new_ax.set_title('')
		new_ax.set_xlabel(x_label)
		new_ax.set_ylabel(y_label)
		new_x = []
		new_y = []
		self.x.append(new_x)
		self.y.append(new_y)

	def add_points(self, x, y, name):
		self.x[self.figure_names.index(name)].append(x)
		self.y[self.figure_names.index(name)].append(y)

	def draw(self, label, name):
		plt.figure(self.figure_names.index(name)+1)
		plt.plot(self.x[self.figure_names.index(name)], self.y[self.figure_names.index(name)], label=label)

	def points(self, label, name):
		open(os.path.join("data/", name), 'w').close()
		f = open(os.path.join("data/", name), "w")
		x = self.x[self.figure_names.index(name)]
		y = self.y[self.figure_names.index(name)]
		for i in range(0, len(x)):
			f.write("(" + str(x[i]) + "," + str(y[i]) + ")")
		f.close()

	def show(self):
		plt.show() 
		
