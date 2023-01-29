# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 深圳星河软通科技有限公司 A.Star
# @contact: astar@snowland.ltd
# @site: www.astar.ltd
# @file: 
# @time: 
# @Software: PyCharm

from abc import ABCMeta
from numbers import Number
from typing import List

import numpy as np
from astartool.error import ParameterTypeError, ParameterValueError
from astartool.number import equals_zero
from snowland.graphics.core.analytic_geometry_base import Line
from snowland.graphics.core.computational_geometry_2d import Point2D

npa = np.array
npl = np.linalg

__all__ = [
    'Line2D',
    'ConicalSection',
    'Ellipse',
    'Circle',
    'Hyperbola',
    'Parabola',
    'Polynomial',

    'LEFT',
    'ON',
    'RIGHT'
]

LEFT = -1
ON = 0
RIGHT = 1


class Line2D(Line):
    def __init__(self, a=None, b=0, c=0, p1: (Point2D, np.ndarray, list) = None, p2: (Point2D, np.ndarray, list) = None,
                 k=None,
                 *, dtype=None, **kwargs):
        if a is not None and b is not None and c is not None:
            # ax+by+c=0方式定义的线
            self.a, self.b, self.c = a, b, c
        elif p1 is not None and p2 is not None:
            # 两点式定义的直线
            p1, p2 = Point2D(p1, dtype=dtype), Point2D(p2, dtype=dtype)
            self.a, self.b, self.c = p2.y - p1.y, p1.x - p2.x, p2.x * p1.y - p1.x * p2.y
        elif k is not None:
            if p1 is not None:
                # 点斜式确定的直线
                # y-y0 = k(x-x0)
                p1 = Point2D(p1, dtype=dtype)
                self.a, self.b, self.c = k, -1, p1.y - k * p1.x
            else:
                # 斜截式定义的直线
                # y = kx+b
                self.a, self.b, self.c = k, -1, self.b
        else:
            # TODO: 其他方式确定的直线
            raise ParameterValueError("错误的输入")

    def __distance_point(self, point):
        """
        计算点到直线距离
        :param point: 点
        :return:
        """
        p = Point2D(point)
        return np.fabs(self.a * p.x + self.b * p.y + self.c) / (self.a ** 2 + self.b ** 2) ** 0.5

    def distance_points(self, points):
        """
        计算多点到直线的距离
        :param points: 可以是np.ndarray, 也可以是list(Point2D)
        :return:
        """
        # TODO: 这里可以用矩阵进行运算的，以后补上
        if isinstance(points, list):
            return npa([self.__distance_point(each) for each in points])
        elif isinstance(points, np.ndarray):
            if len(points.shape) == 2:
                return npa([self.__distance_point(each) for each in points])
            else:
                return self.__distance_point(points)
        else:
            raise ValueError('points 参数错误')

    def is_parallel(self, other, eps=1e-8):
        """
        判断直线平行
        """
        if isinstance(other, Line2D):
            return -eps < (self.a - other.a) < eps and -eps < self.b - other.b < eps
        elif isinstance(other, (list, tuple, np.ndarray)):
            return -eps < (self.a - other[0]) < eps and -eps < self.b - other[1] < eps
        else:
            raise ParameterTypeError("输入类型错误")

    def intersection(self, other, eps=1e-8):
        """
        直线交点
        :param other:
        :param eps:
        :return:
        """
        if isinstance(other, Line2D):
            if -eps < (self.a - other.a) < eps and -eps < self.b - other.b < eps:
                return None  # 不相交的两条直线
            else:
                return Point2D([-(self.b * other.c - self.c * other.b) / (other.a * self.b - self.a * other.b), \
                                (self.a * other.c - self.c * other.a) / (other.a * self.b - self.a * other.b)])
        elif isinstance(other, (list, tuple, np.ndarray)):
            assert len(other) == 3
            if -eps < (self.a - other[0]) < eps and -eps < self.b - other[1] < eps:
                return None  # 不相交的两条直线
            return Point2D([-(self.b * other[2] - self.c * other[1]) / (other[0] * self.b - self.a * other[1]), \
                            (self.a * other[2] - self.c * other[0]) / (other[0] * self.b - self.a * other[1])])
        else:
            raise ParameterTypeError("输入类型错误")

    def get(self, x: (float, np.ndarray) = None, y: (float, np.ndarray) = None, eps=1e-10):
        """
        获得对应的取值
        :param x: x值
        :param y: y值
        :param eps: 在此精度下认为是 0
        :return:
        """
        assert (x is None) ^ (y is None), "x和y不全为None"
        if x is None:
            if isinstance(y, (int, float, np.float, np.int)):
                if equals_zero(self.a, eps):
                    return np.inf
                else:
                    return (-self.c - self.b * y) / self.a
            else:
                y = npa(y)
                return (-self.c - self.b * y) / self.a
        else:
            if isinstance(x, (int, float, np.float, np.int)):
                if equals_zero(self.b, eps):
                    return np.inf
                else:
                    return (-self.c - self.a * x) / self.b
            else:
                x = npa(x)
                return (-self.c - self.a * x) / self.b

    def location(self, point: (Point2D, np.ndarray), eps=1e-8):
        point = Point2D(point)
        res = self.a * point.x + self.b * point.y + self.c
        if -eps < res < eps:
            return ON
        elif res > eps:
            return RIGHT
        else:
            return LEFT

    def locations(self, points: (List[Point2D], np.ndarray), eps=1e-8):
        if isinstance(points, list):
            return [self.location(point, eps) for point in points]
        elif isinstance(points, np.ndarray):
            result = np.ones(len(points)) * ON
            res = self.a * points[:, 0] + self.b * points[:, 1] + self.c
            result[res > eps] = RIGHT
            result[res < -eps] = LEFT
            return result
        else:
            raise ParameterTypeError("不支持的points类型")

    def __str__(self):
        ax = ''
        if np.isclose(self.a, 0):
            ax = ''
        elif np.isclose(self.a, 1):
            ax = 'x'
        elif np.isclose(self.a, -1):
            ax = '-x'
        else:
            ax = '{a}x'.format(a=self.a)

        if np.isclose(self.b, 0):
            by = ''
        elif np.isclose(self.b, 1):
            by = 'y'
        elif np.isclose(self.b, -1):
            by = '-y'
        else:
            by = '{b}y'.format(b=self.b)
        s = "{ax}+{by}+{c}=0".format(ax=ax, by=by, c=self.c)
        return s.replace("+-", "-")


class ConicalSection(Line, metaclass=ABCMeta):
    """
    圆锥曲线
    """
    def __init__(self, params, *args, dtype=None, **kwargs):
        self.params = npa(params, dtype=dtype)


class Ellipse(ConicalSection):
    """
    椭圆
    """
    def __init__(self, params=None, a=None, b=None, p=(0, 0), *args, dtype=None, **kwargs):
        if params is not None:
            # Ax2+By2+Cxy+Dx+Ey+F=0
            super().__init__(params, *args, dtype=dtype, **kwargs)
        elif a is not None and b is not None:
            a2 = a * a
            b2 = b * b
            p = Point2D(p, dtype=dtype)
            px2 = p.x * p.x
            py2 = p.y * p.y
            super().__init__([b2, a2, 0, -2 * p.x * b2, -2 * p.y * a2, b2 * px2 + a2 * py2 - a2 * b2],
                             dtype=dtype, **kwargs)

        else:
            raise ParameterValueError("参数错误")

    @property
    def a(self):
        return self.params[0]

    @property
    def b(self):
        return self.params[1]

    @property
    def c(self):
        return self.params[2]

    @property
    def d(self):
        return self.params[3]

    @property
    def e(self):
        return self.params[4]

    @property
    def f(self):
        return self.params[5]

    @property
    def eccentricity(self):
        return self.focal_length / self.major_axis / 2

    @property
    def major_axis(self):
        """
        半长轴
        :return:
        """
        return max(self.a, self.b) ** 0.5

    @property
    def minor_axis(self):
        """
        半短轴
        :return:
        """
        return min(self.a, self.b) ** 0.5

    @property
    def focal_length(self):
        """
        焦距
        :return:
        """
        return 2 * abs(self.a - self.b) ** 0.5

    def get(self, x):
        # Ax2+By2+Cxy+Dx+Ey+F=0  ==>  by^2+(cx+e)y + ax^2+dx+f = 0
        a = self.b
        b = self.c * x + self.e
        c = self.a * x * x + self.d * x + self.f
        delta = b * b - 4 * a * c
        sqrt_delta = delta ** 0.5
        return (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a)

    def to_string(self):
        return f"{self.a}x^2+{self.b}y^2+{self.c}xy+{self.d}x+{self.e}y+{self.f}=0".replace("+-", "-")

    def __str__(self):
        return self.to_string()


class Circle(Ellipse):
    def __init__(self, param=None, p=None, r=None, *args, **kwargs):
        if param is not None:
            # x^2+y^2+Dx+Ey+F=0
            if len(param) == 6:
                super(Circle, self).__init__(param, *args, **kwargs)
            elif len(param) == 3:
                p = np.ones(6)
                p[2] = 0
                p[3:] = param
                super(Circle, self).__init__(p, *args, **kwargs)
            else:
                raise ParameterValueError("param参数应为3个或者6个")
            self.centre = Point2D([-self.d / 2, -self.e / 2])
            self.radius = (self.centre.x * self.centre.x + self.centre.y * self.centre.y - self.f) ** 0.5
        elif p is not None and r is not None:
            params = [1, 1, 0, -2 * p[0], -2 * p[1], p[0] ** 2 + p[1] ** 2 - r * r]
            super(Circle, self).__init__(params, *args, **kwargs)
            self.centre = Point2D(p)
            self.radius = r
        else:
            raise ParameterValueError("参数错误")


class Hyperbola(ConicalSection):
    """
    双曲线
    """

    def __init__(self, params=None, a=None, b=None, p=(0, 0), eps=1e-8, axis='x', dtype=None, **kwargs):
        # Ax2+2Bxy+Cy2+2Dx+2Ey+F=0
        if params is not None:
            super().__init__(params, dtype=dtype, **kwargs)
            self.axis = axis
            if equals_zero(np.linalg.det([[self.a, self.b, self.d],
                                          [self.b, self.c, self.e],
                                          [self.d, self.e, self.f]]), eps):
                raise ParameterValueError("参数矩阵非满秩")

            if self.b * self.b - self.a * self.c <= eps:
                raise ParameterValueError("b^2-ac应该大于0")
        elif a is not None and b is not None:
            a2 = a * a
            b2 = b * b
            p = Point2D(p, dtype=dtype)
            px2 = p.x * p.x
            py2 = p.y * p.y
            super().__init__([b2, 0, -a2, -p.x * b2, p.y * a2, b2 * px2 - a2 * py2 - a2 * b2],
                             dtype=dtype, **kwargs)
            self.axis = axis

        else:
            raise ParameterValueError("参数错误")

    @property
    def focal_length(self):
        """
        焦距
        :return:
        """
        return 2 * (abs(self.a) + abs(self.c)) ** 0.5

    @property
    def eccentricity(self):
        if self.axis == 'x':
            return self.focal_length / self.a / 2
        else:
            return self.focal_length / self.b / 2

    @property
    def a(self):
        return self.params[0]

    @property
    def b(self):
        return self.params[1]

    @property
    def c(self):
        return self.params[2]

    @property
    def d(self):
        return self.params[3]

    @property
    def e(self):
        return self.params[4]

    @property
    def f(self):
        return self.params[5]

    def get(self, x):
        # Ax2+2Bxy+Cy2+2Dx+2Ey+F=0  ==>  cy^2+(2bx+2e)y + ax^2+2dx+f = 0
        a = self.c
        b = 2 * self.b * x + 2 * self.e
        c = self.a * x * x + 2 * self.d * x + self.f
        delta = b * b - 4 * a * c
        sqrt_delta = np.sqrt(delta)
        return (-b + sqrt_delta) / (2 * a), (-b - sqrt_delta) / (2 * a)

    def to_string(self):
        return f"{self.a}x^2+2*{self.b}xy+{self.c}y2+2*{self.d}x+2*{self.e}y+{self.f}=0".replace("+-", "-")

    def __str__(self):
        return self.to_string()


class Parabola(ConicalSection):
    """
    抛物线
    """

    def __init__(self, params, *args, dtype=None, **kwargs):
        # ax2 +2hxy +by2 +2gx +2fy+c =0
        super(Parabola, self).__init__(params, dtype=dtype, *args, **kwargs)

    @property
    def a(self):
        return self.params[0]

    @property
    def b(self):
        return self.params[2]

    @property
    def h(self):
        return self.params[1]

    @property
    def g(self):
        return self.params[3]

    @property
    def c(self):
        return self.params[5]

    @property
    def f(self):
        return self.params[4]

    def to_string(self):
        return f"{self.a}x^2+2*{self.h}xy+{self.b}y^2+2*{self.g}x+2*{self.f}y+{self.c}=0".replace("+-", "-")

    def __str__(self):
        return self.to_string()


class Polynomial(Line):
    """
    多项式
    """

    def __init__(self, polynomial=None, coefficient=None, exponent=None):
        """
        coefficient: 系数
        exponent: 指数
        """
        if polynomial is not None:
            if isinstance(polynomial, Polynomial):
                self.polynomial_dict = polynomial.polynomial_dict
            elif isinstance(polynomial, dict):
                self.polynomial_dict = polynomial
            else:
                raise ParameterTypeError("参数类型错误")
        elif coefficient is not None:
            if exponent is not None:
                self.polynomial_dict = dict(zip(exponent, coefficient))
            else:
                self.polynomial_dict = dict(zip(range(len(coefficient)), coefficient))
        else:
            self.polynomial_dict = {}

    @property
    def coefficient(self):
        return list(self.polynomial_dict.values())

    @property
    def exponent(self):
        return list(self.polynomial_dict.keys())

    def diff(self, eps=1e-8):
        """
        导函数
        """
        return Polynomial({k - 1: k * v for k, v in self.polynomial_dict.items() if not (-eps < k < eps)})

    def get(self, x):
        """
        获得x对应的值
        """
        y = np.zeros_like(x)
        for k, v in self.polynomial_dict.items():
            y += v * x ** k
        return y

    def tangent_line(self, x, eps=1e-8):
        """
        求x位置上的切线方程
        :param x:
        :param eps:
        :return:
        """
        diff_poly = self.diff(eps)
        return Line2D(k=diff_poly(x), p1=(x, self(x)))

    def simplify(self, eps=1e-8):
        """
        化简多项式，删去系数为0的项
        :param eps: 判断0的精度
        :return:
        """
        self.polynomial_dict = {k: v for k, v in self.polynomial_dict.items() if not (-eps < v < eps)}

    def __call__(self, x, *args, **kwargs):
        return self.get(x)

    def __iadd__(self, other):
        if isinstance(other, Polynomial):
            for k, v in other.polynomial_dict.items():
                if k in self.polynomial_dict:
                    self.polynomial_dict[k] += v
                else:
                    self.polynomial_dict[k] = v
        elif isinstance(other, Number):
            if 0 in self.polynomial_dict:
                self.polynomial_dict += other
            else:
                self.polynomial_dict[0] = other
        else:
            raise ParameterTypeError("错误的数据类型")

    def __isub__(self, other):
        if isinstance(other, Polynomial):
            for k, v in other.polynomial_dict.items():
                if k in self.polynomial_dict:
                    self.polynomial_dict[k] -= v
                else:
                    self.polynomial_dict[k] = -v
        elif isinstance(other, Number):
            if 0 in self.polynomial_dict:
                self.polynomial_dict -= other
            else:
                self.polynomial_dict[0] = -other
        else:
            raise ParameterTypeError("错误的数据类型")

    def __add__(self, other):
        poly = self.polynomial_dict
        if isinstance(other, Polynomial):
            for k, v in other.polynomial_dict.items():
                if k in poly:
                    poly[k] += v
                else:
                    poly[k] = v
        elif isinstance(other, Number):
            if 0 in poly:
                poly += other
            else:
                poly = other
        else:
            raise ParameterTypeError("错误的数据类型")
        return Polynomial(poly)

    def __sub__(self, other):
        poly = self.polynomial_dict
        if isinstance(other, Polynomial):
            for k, v in other.polynomial_dict.items():
                if k in poly:
                    poly[k] -= v
                else:
                    poly[k] = -v
        elif isinstance(other, Number):
            if 0 in poly:
                poly[0] -= other
            else:
                poly[0] = -other
        else:
            raise ParameterTypeError("错误的数据类型")
        return Polynomial(poly)

    def to_string(self, symbol='x'):
        res = [(k, v) for k, v in self.polynomial_dict.items()]
        res.sort()
        s = []
        for p, c in self.polynomial_dict.items():
            if np.isclose(c, 1):
                c_str = ''
            elif np.isclose(c, -1):
                c_str = '-'
            elif np.isclose(c, 0):
                continue
            else:
                c_str = "{c}".format(c=c)

            if np.isclose(p, 0):
                p_str = ''
            elif np.isclose(p, 1):
                p_str = '{symbol}'.format(symbol=symbol)
            elif p > 0:
                p_str = '{symbol}^{p}'.format(symbol=symbol, p=p)
            else:
                p_str = '{symbol}^({p})'.format(symbol=symbol, p=p)
            s.append(c_str + p_str)

        s_str = '+'.join(s)
        return s_str.replace("+-", "-")

    def __str__(self):
        return self.to_string('x')
