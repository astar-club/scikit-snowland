# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 深圳星河软通科技有限公司 A.Star
# @contact: astar@snowland.ltd
# @site: www.astar.ltd
# @file: utils.py
# @time: 2022/01/06 20:48
# @Software: PyCharm
from typing import List

import numpy as np
from astartool.data_structure import LinkedList
from astartool.number import equals_zero_all
from scipy.spatial.distance import cdist, euclidean

npa = np.array
npl = np.linalg


def middle(x, x1, x2, eps=1e-8):
    """
    判断点是否在x1, x2之间
    x, x1, x2 均为ndarray
    """
    dx1 = x - x1
    dx2 = x - x2
    return ((x1 < x) & (x < x2)) | ((x1 > x) & (x > x2)) | ((-eps < dx1) & (dx1 < eps)) | ((-eps < dx2) & (dx2 < eps))


def is_in_between(x, x1, x2, eps=1e-8):
    """
    判断点是否在x1, x2之间
    x, x1, x2 均为ndarray
    """
    dx1 = x - x1
    dx2 = x - x2
    return ((x1 < x) & (x < x2)) | ((x1 > x) & (x > x2)) | ((-eps < dx1) & (dx1 < eps)) | ((-eps < dx2) & (dx2 < eps))


def get_lines(line_points):
    """
    """
    A = line_points[1:, 1] - line_points[:-1, 1]
    B = line_points[:-1, 0] - line_points[1:, 0]
    C = line_points[1:, 0] * line_points[:-1, 1] - line_points[:-1, 0] * line_points[1:, 1]
    return np.vstack((A, B, C)).T


def get_foot(point, line=None, A=None, B=None, C=None):
    """
    point: 单点, ndarray/list len(points) = 2
    line: Ax+By+C=0
    返回 foot_x, foot_y, 垂足坐标

    """
    if line is not None:
        A, B, C = line[:, 0], line[:, 1], line[:, 2]
    elif A is not None and B is not None and C is not None:
        pass

    x, y = point[0], point[1]
    A2, B2, AB, AC, BC = A * A, B * B, A * B, A * C, B * C
    # foot_x, foot_y 即为垂足坐标
    foot_x = (B2 * x - AB * y - AC) / (A2 + B2)
    foot_y = (A2 * y - AB * x - BC) / (A2 + B2)

    return foot_x, foot_y


def get_intersect_point(line):
    """
    line_points 是 n x 3 矩阵，代表组成线标准式的参数 Ax+By+C=0 的系数 ABC
    返回这些线的交点
    """
    line1 = line[:-1, :]
    line2 = line[1:, :]

    a1 = line1[:, 0]
    b1 = line1[:, 1]
    c1 = line1[:, 2]
    a2 = line2[:, 0]
    b2 = line2[:, 1]
    c2 = line2[:, 2]
    b1c2 = b1 * c2
    b2c1 = b2 * c1
    a1c2 = a1 * c2
    a1b2 = a1 * b2
    a2b1 = a2 * b1
    a2c1 = a2 * c1

    x = (b1c2 - b2c1) / (a1b2 - a2b1)
    y = (a2c1 - a1c2) / (a1b2 - a2b1)
    return x, y


def distance_point_line(points, line_points, metric=euclidean, eps=1e-8):
    """
    points: 单点, ndarray/list len(points) = 2
    line_points: 每两点组成的一条线， 代表一条linestring
    返回tuple(isVe, ind, d)
    isVe: 找到的最近点是不是线的顶点
    p: 距离最近的点，ndarray
    d: 最近距离是多少
    """
    # ABC即为线的参数 Ax + By + C = 0
    A = line_points[1:, 1] - line_points[:-1, 1]
    B = line_points[:-1, 0] - line_points[1:, 0]
    C = line_points[1:, 0] * line_points[:-1, 1] - line_points[:-1, 0] * line_points[1:, 1]

    foot_x, foot_y = get_foot(points, A=A, B=B, C=C)
    # 是否为顶点坐标
    is_vertex = True

    index = is_in_between(foot_x, line_points[1:, 0], line_points[:-1, 0], eps=eps) & \
            is_in_between(foot_y, line_points[1:, 1], line_points[:-1, 1], eps=eps)
    if np.any(index):
        is_vertex = False
        points_check = np.vstack((foot_x[index].T, foot_y[index].T)).T
    else:
        points_check = line_points

    dist = cdist(npa([points]), points_check, metric=metric)
    ind = np.argmin(dist)
    return is_vertex, points_check[ind], dist[0, ind]


def distance_point_line_index(point, line_points, metric=euclidean, eps=1e-8):
    """
    params points: 单点(x, y), ndarray/list len(points) = 2
    line_points: 每两点组成的一条线， 代表一条linestring
    返回tuple(isVe, ind, d)
    is_vertex: 找到的最近点是不是线的顶点
    p: 距离最近的点，ndarray
    d: 最近距离是多少
    ind: index 若 is_vertex 为 True, ind为最近节点的编号, 若 is_vertex 为 False, ind为最近节点的编号

    """
    # ABC即为线的参数 Ax + By + C = 0
    A = line_points[1:, 1] - line_points[:-1, 1]
    B = line_points[:-1, 0] - line_points[1:, 0]
    C = line_points[1:, 0] * line_points[:-1, 1] - line_points[:-1, 0] * line_points[1:, 1]

    foot_x, foot_y = get_foot(point, A=A, B=B, C=C)
    # 是否为顶点坐标
    is_vertex = True

    index = middle(foot_x, line_points[1:, 0], line_points[:-1, 0], eps=eps) & \
            middle(foot_y, line_points[1:, 1], line_points[:-1, 1], eps=eps)
    if np.any(index):
        is_vertex = False
        index_ind, = np.where(index)
        points_check = np.vstack((foot_x[index].T, foot_y[index].T)).T

        dist = cdist(npa([point]), points_check, metric=metric)
        ind = np.argmin(dist)
        return is_vertex, points_check[ind], dist[0, ind], index_ind[ind]
    else:
        points_check = line_points
        dist = cdist(npa([point]), points_check, metric=metric)
        ind = np.argmin(dist)
        return is_vertex, points_check[ind], dist[0, ind], ind


def move_distance(a, dist, flag):
    """
    移线
    """
    lines = get_lines(a)
    norm_line = npl.norm(lines[:, :2], axis=1)
    delta_c = dist * norm_line
    lines_new = lines
    if flag:
        lines_new[:, 2] -= delta_c
    else:
        lines_new[:, 2] += delta_c
    x, y = get_intersect_point(lines_new)
    if len(x):
        x_start, y_start = get_foot(a[0, :], lines[:1, :])
        x_end, y_end = get_foot(a[-1, :], lines[-1:, :])
        x = np.hstack((x_start, x, x_end))
        y = np.hstack((y_start, y, y_end))
    return x, y


def simple_line(points=None, x: np.ndarray = None, y: np.ndarray = None, forward_point=8):
    """
    对一条线(x, y)进行化简，删除折返线
    """
    if x is not None and y is not None:
        x_array = npa(x)
        y_array = npa(y)
        points = np.vstack((x_array, y_array)).T

    length_points = len(points)
    ps_index = np.empty(length_points, dtype=int)
    i, cnt = 0, 0
    ps = [points[0]]
    while i + 1 < length_points:
        p1 = points[i]
        p2 = points[i + 1]
        p_temp = p2
        for j in range(min(i + forward_point, length_points - 1), i + 1, -1):
            p3 = points[j]
            p4 = points[j - 1]
            x, y = get_intersect_by_two_point(p1, p2, p3, p4)
            if x is not None and y is not None:
                p_temp = x, y
                global_j = j - 1
                break
        else:
            global_j = i + 1
        ps_index[i] = cnt
        ps_index[i + 1:global_j] = i + 1
        cnt += 1
        i = global_j
        ps.append(p_temp)
    ps_index[-1] = cnt
    ps_array = npa(ps)
    return ps_array[:, 0], ps_array[:, 1], ps_index


def move_distance_with_endpoints(a, dist, flag):
    """
    移线，注意和move_distance移线不同！！！
    此函数带有端点
    """
    lines = get_lines(a)
    norm_line = npl.norm(lines[:, :2], axis=1)
    delta_c = dist * norm_line
    lines_new = lines
    if flag:
        lines_new[:, 2] -= delta_c
    else:
        lines_new[:, 2] += delta_c
    x, y = get_intersect_point(lines_new)
    x_start, y_start = get_foot(a[0, :], lines[:1, :])
    x_end, y_end = get_foot(a[-1, :], lines[-1:, :])
    x = np.hstack((x_start, x, x_end))
    y = np.hstack((y_start, y, y_end))
    return x, y


def interp_points(line1: np.ndarray, line2: np.ndarray, metric=euclidean, eps=1e-8):
    """
    插入点
    :param line1:
    :param line2:
    :param metric:
    :param eps:
    :return:
    """

    line1 = line1[:, :2]
    line2 = line2[:, :2]

    line1, line2 = alignment(line1, line2, metric, eps=eps)

    point_list_2 = LinkedList()
    new_point_list_2 = []
    for p in line1[1:-1]:
        flag, point, dist, index = distance_point_line_index(p, line2, metric=metric, eps=eps)
        if not flag:
            point_list_2.append((index, point))

    point_list_1 = LinkedList()
    new_point_list_1 = []
    for p in line2[1:-1]:
        flag, point, dist, index = distance_point_line_index(p, line1, metric=metric, eps=eps)
        if not flag:
            point_list_1.append((index, point))

    pre_i = 0
    for item in point_list_1:
        i, p = item
        new_point_list_1.extend(line1[pre_i:i + 1])
        new_point_list_1.append(p)
        pre_i = i + 1
    new_point_list_1.extend(line1[pre_i:])

    pre_j = 0
    for item in point_list_2:
        j, p = item
        new_point_list_2.extend(line2[pre_j:j + 1])
        new_point_list_2.append(p)
        pre_j = j + 1
    new_point_list_2.extend(line2[pre_j:])

    # assert len(new_point_list_1) == len(new_point_list_2)
    return np.vstack(new_point_list_1), np.vstack(new_point_list_2)


def alignment(linestring1: np.ndarray, linestring2: np.ndarray, metric=euclidean, eps: float = 1.0):
    """
    化简车道线,使之按垂直方向对齐
    linestring1: 第一根线 m x 2 ndarray
    linestring2: 第二根线 n x 2 ndarray
    metric: 距离计算函数
    eps: 若距离小于 eps, 则不需要截取
    """
    assert linestring1.shape[1] == 2
    assert linestring2.shape[1] == 2
    is_vertex, points_check, dist, ind = distance_point_line_index(linestring2[0], linestring1, metric)
    if (not is_vertex) and metric(points_check, linestring1[0]) > eps:
        linestring1 = linestring1[ind + 1:]
        linestring1 = np.vstack((points_check, linestring1))
    else:
        is_vertex, points_check, dist, ind = distance_point_line_index(linestring1[0], linestring2, metric)
        if (not is_vertex) and metric(points_check, linestring2[0]) > eps:
            linestring2 = linestring2[ind + 1:]
            linestring2 = np.vstack((points_check, linestring2))

    is_vertex, points_check, dist, ind = distance_point_line_index(linestring2[-1], linestring1, metric)
    if (not is_vertex) and metric(points_check, linestring1[-1]) > eps:
        linestring1 = linestring1[:ind + 1]
        linestring1 = np.vstack((linestring1, points_check))
    else:
        is_vertex, points_check, dist, ind = distance_point_line_index(linestring1[-1], linestring2, metric)
        if (not is_vertex) and metric(points_check, linestring2[-1]) > eps:
            linestring2 = linestring2[:ind + 1]
            linestring2 = np.vstack((linestring2, points_check))

    return linestring1, linestring2


def rotate_geometry(poly: (np.ndarray, List), rad):
    """
    二维旋转rad弧度
    """
    if isinstance(poly, List):
        poly = npa(poly)
    if len(poly.shape) == 1:
        flag = True
        poly = np.expand_dims(poly, 0)
    else:
        flag = False
    matrix = npa([[np.cos(rad), np.sin(rad)],
                  [-np.sin(rad), np.cos(rad)]])
    new_poly = np.dot(poly, matrix)
    return new_poly[0, :] if flag else new_poly


def bounding_box(poly: np.ndarray, eps=1e-10):
    """
    boundingBox
    """
    # shape of min_rect: [4, 2]
    x_min, y_min = np.min(poly, axis=0)
    x_max, y_max = np.max(poly, axis=0)
    return npa([x_min, y_min]), (x_max - x_min), (y_max - y_min)


def min_rotate_rect_a(hull: np.ndarray, eps=1e-10):
    """
    最小外接矩形--按面积
    """
    area = np.inf
    lines_com = []
    if not equals_zero_all(hull[0] - hull[-1], eps):
        points_hull = np.vstack((hull, hull[0]))
    else:
        points_hull = hull

    lines = get_lines(points_hull)
    for i, p2 in enumerate(points_hull[:-1]):
        line = lines[i]
        dist_cmp = (points_hull[:, 0] * line[0] + points_hull[:, 1] * line[1] + line[2])
        dist = dist_cmp / npl.norm(line[:2])

        h_ind = np.argmax(np.abs(dist))  # 得到距离最大的点距离，即为高，同时得到该点坐标
        h = dist[h_ind]
        line_px = line.copy()
        line_px[2] -= dist_cmp[h_ind]
        line_cz = npa([line[1], -line[0], 0])

        dist_cmp = (points_hull[:, 0] * line_cz[0] + points_hull[:, 1] * line_cz[1])
        dist = dist_cmp / npl.norm(line_cz[:2])

        v_ind_max = np.argmax(dist)  # 得到距离最大的点距离，即为高
        v_ind_min = np.argmin(dist)

        w = dist[v_ind_max] - dist[v_ind_min]
        a = abs(h * w)
        if area >= h * w:
            # 使面积最小
            area = a
            line_cz_1 = line_cz.copy()
            line_cz_1[2] -= dist_cmp[v_ind_max]
            line_cz_2 = line_cz.copy()
            line_cz_2[2] = dist_cmp[v_ind_min]
            lines_com = [line, line_cz_1, line_px, line_cz_2, line]

    x, y = get_intersect_point(np.vstack(lines_com))
    return np.vstack((x, y)).T, area


def min_rotate_rect_c(hull: np.ndarray, eps=1e-10):
    """
    最小外接矩形--按周长
    """
    cir = np.inf
    lines_com = []
    if not equals_zero_all(hull[0] - hull[-1], eps):
        points_hull = np.vstack((hull, hull[0]))
    else:
        points_hull = hull

    lines = get_lines(points_hull)
    for i, p2 in enumerate(points_hull[:-1]):
        line = lines[i]
        dist_cmp = (points_hull[:, 0] * line[0] + points_hull[:, 1] * line[1] + line[2])
        dist = dist_cmp / npl.norm(line[:2])

        h_ind = np.argmax(np.abs(dist))  # 得到距离最大的点距离，即为高，同时得到该点坐标
        h = dist[h_ind]
        line_px = line.copy()
        line_px[2] -= dist_cmp[h_ind]
        line_cz = npa([line[1], -line[0], 0])

        dist_cmp = (points_hull[:, 0] * line_cz[0] + points_hull[:, 1] * line_cz[1])
        dist = dist_cmp / npl.norm(line_cz[:2])

        v_ind_max = np.argmax(dist)  # 得到距离最大的点距离，即为高
        v_ind_min = np.argmin(dist)

        w = abs(dist[v_ind_max] - dist[v_ind_min])
        c = h + w
        if cir >= c:
            # 使面积最小
            cir = c
            line_cz_1 = line_cz.copy()
            line_cz_1[2] -= dist_cmp[v_ind_max]
            line_cz_2 = line_cz.copy()
            line_cz_2[2] = dist_cmp[v_ind_min]
            lines_com = [line, line_cz_1, line_px, line_cz_2, line]

    x, y = get_intersect_point(np.vstack(lines_com))
    return np.vstack((x, y)).T, cir * 2


def min_rotate_rect(hull: np.ndarray, cmp: str = 'a', eps=1e-10):
    """
    最小外接矩形
    """
    if cmp.lower().startswith('a'):
        return min_rotate_rect_a(hull, eps)
    else:
        return min_rotate_rect_c(hull, eps)


def get_angle_rad(a, b, eps=1e-20):
    """
    a, b 为一维向量， 多点会错！！！
    返回 向量a和b之间的夹角， 值域是 -pi ~ pi
    """
    a = npa(a)
    b = npa(b)
    norm_a = npl.norm(a, axis=1) if len(a.shape) == 2 else npl.norm(a)
    norm_b = npl.norm(b, axis=1) if len(b.shape) == 2 else npl.norm(b)
    # 单位化（可以不用这一步）
    # a = a / norm_a  # 不能写成 a /= norm_a
    # b = b / norm_b  # 不能写成 b /= norm_b
    # 夹角cos值
    if -eps < norm_a < eps or -eps < norm_b < eps:
        return np.nan
    cos_ = np.dot(a, b) / (norm_a * norm_b)
    # 夹角sin值
    sin_ = np.cross(a, b) / (norm_a * norm_b)
    arctan2_ = np.arctan2(sin_, cos_)
    return arctan2_


def get_intersect_by_two_point(p1, p2, p3, p4):
    """
    判断线段是否相交
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    b1 = (y2 - y1) * x1 + (x1 - x2) * y1
    b2 = (y4 - y3) * x3 + (x3 - x4) * y3
    D = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
    D1 = b2 * (x2 - x1) - b1 * (x4 - x3)
    D2 = b2 * (y2 - y1) - b1 * (y4 - y3)
    x, y = D1 / D, D2 / D
    if middle(x, x1, x2) and middle(x, x3, x4) and middle(y, y1, y2) and middle(y, y3, y4):
        return x, y
    else:
        return None, None


def get_angle_degree(a: (list, np.ndarray), b: (list, np.ndarray)):
    """
    返回量向量之间的夹角，值域是 0~180，单位是度
    """
    # 初始化向量
    degree = np.rad2deg(get_angle_rad(a, b))
    return np.abs(degree)


def get_rotate_angle_degree(v1, v2):
    """
    v1旋转到v2经历的角度， 值域0~360, 单位是度
    """
    return np.rad2deg(get_angle_rad(v1, v2)) % 360


def get_rotate_angle_rad(v1, v2):
    """
    v1旋转到v2经历的角度， 值域0~2*pi, 单位是弧度
    """
    return get_angle_rad(v1, v2) % (2 * np.pi)


def get_point_by_rate(line: np.ndarray, meters, metric=euclidean):
    """
    在line组成的折线中,获得距离起点距离为metres的点
    """
    return get_point_by_rate_index(line, meters, metric)[0]


def get_point_by_rate_index(line: np.ndarray, meters, metric=euclidean):
    """
    在line组成的折线中,获得距离起点距离为metres的点和小于这个点的最大节点编号
    """
    meters_all = npa([metric(a, b) for a, b in zip(line[:-1], line[1:])])
    s = np.cumsum(meters_all)
    if s[-1] < meters:
        return line[-1], len(s)
    ind = np.sum(s < meters_all)
    s = np.insert(s, 0, 0)
    delta_meters = meters - s[ind]
    vector_line = line[1:] - line[:-1]
    return line[ind] + delta_meters / meters_all[ind] * vector_line[ind], ind


def get_arc(o, r, theta1, theta2, length):
    """
    """
    theta = np.linspace(theta1, theta2, length)
    x = r * np.cos(theta) + o[0]
    y = r * np.sin(theta) + o[1]
    return x, y


def rect_to_points(rect):
    """
    rect矩形转坐标点
    """
    points = np.vstack([rect[0],
                        [rect[0][0], rect[0][1] + rect[2]],
                        [rect[0][0] + rect[1], rect[0][1] + rect[2]],
                        [rect[0][0] + rect[1], rect[0][1]],
                        rect[0]
                        ])
    return points


def move_distance_by_point(p, vector, dist, metric=euclidean):
    unit_vector = vector / npl.norm(vector)
    unit_point = unit_vector + p
    unitLen = metric(p, unit_point)
    vecCenter = dist * unit_vector / unitLen
    return vecCenter + p


def move_distance_for_polygon(points, dist, flag=1, metric=euclidean):
    sign = np.sign(flag)
    lines = []
    for p1, p2 in zip(points[:-1, :], points[1:, :]):
        v_edge = p2 - p1
        if equals_zero_all(v_edge):
            continue
        v = rotate_geometry(v_edge, -sign * np.pi / 2)
        new_p1 = move_distance_by_point(p1, v, dist, metric)
        new_p2 = move_distance_by_point(p2, v, dist, metric)
        lines.append(get_lines(np.vstack([new_p1, new_p2])))
    lines.append(lines[0])
    lines_ndarray = np.vstack(lines)
    x, y = get_intersect_point(lines_ndarray)
    return np.hstack((x[-1], x)), np.hstack((y[-1], y))


def curvature(ps):
    """
    计算曲率
    :param ps n x 2 ndarray
    :return:
    """
    vs = ps[1:] - ps[:-1]
    vs2 = ps[2:] - ps[:-2]
    dis1 = np.linalg.norm(vs, axis=1)
    dis2 = np.linalg.norm(vs2, axis=1)
    return 2 * (vs[:-1, 0] * vs[1:, 1] - vs[:-1, 1] * vs[1:, 0]) / (dis1[:-1] * dis1[1:] * dis2)
