# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from snowland.graphics.core.analytic_geometry_2d import Ellipse, Circle, Hyperbola

p = (3, 4)
r = 2
ellipse = Circle(p=p, r=r)
theta = np.linspace(0, np.pi, 50)
xi = p[0] + r * np.cos(theta)
yi1, yi2 = ellipse.get(xi)

ellipse2 = Circle(ellipse.params)
print(ellipse2)
print("圆心:", ellipse2.centre)
print("半径:", ellipse2.radius)


print('--'*10)


a = 4
b = 2
p = (3, 4)
xi_e3 = p[0] + a * np.cos(theta)

ellipse3 = Ellipse(a=a, b=b, p=p)
yi1_e3, yi2_e3 = ellipse3.get(xi_e3)
print(ellipse3)
print("长轴：", ellipse3.major_axis)
print("短轴：", ellipse3.minor_axis)
print("焦距：", ellipse3.focal_length)
print("离心率：", ellipse3.eccentricity)

print('--'*10)

a = 4
b = 2
p = (-2, -3)
xi_h4 = p[0] + a / np.cos(theta)

hyperbola = Hyperbola(a=a, b=b, p=p)

yi1_h4, yi2_h4 = hyperbola.get(xi_h4)
print(hyperbola)
print("离心率：", hyperbola.eccentricity)


print('--'*10)

plt.plot(xi, yi1, 'b', label=str(ellipse))
plt.plot(xi, yi2, 'b')

plt.plot(xi_e3, yi1_e3, 'r', label=str(ellipse3))
plt.plot(xi_e3, yi2_e3, 'r')


plt.plot(xi_h4, yi1_h4, 'g', label=str(hyperbola))
plt.plot(xi_h4, yi2_h4, 'g')


plt.legend()
plt.axis("equal")
plt.grid("on")
plt.show()
