# -*- coding: utf-8 -*-

from typing import Iterable

import numpy as np
from pyproj import Transformer

npa = np.array


def utm_transformer_with_index(latitude, longitude):
    """
    4326转utm坐标系
    """
    if isinstance(latitude, Iterable) and isinstance(longitude, Iterable):
        utm_x = np.empty_like(latitude)
        utm_y = np.empty_like(longitude)
        latitude = npa(latitude)
        longitude = npa(longitude)
        zone_number = (np.array(longitude) / 6).astype(int) + 31
        zone_set = set(zone_number)
        for zone_num in zone_set:
            utm_source_str = f"EPSG:326{zone_num:02}"
            ind = zone_number == zone_num
            transformer = Transformer.from_crs("EPSG:4326", utm_source_str)
            utm_x[ind], utm_y[ind] = transformer.transform(latitude[ind], longitude[ind])
    else:
        zone_number = int(longitude // 6) + 31
        utm_source_str = f"EPSG:326{zone_number:02}"
        transformer = Transformer.from_crs("EPSG:4326", utm_source_str)
        utm_x, utm_y = transformer.transform(latitude, longitude)
    return utm_x, utm_y, zone_number


def utm_transformer(latitude, longitude):
    """
    4326转utm坐标系
    """
    return utm_transformer_with_index(latitude, longitude)[:2]


def latlon_transformer(utm_x, utm_y, zone_number):
    """
    utm转4326坐标系
    """
    latitude = np.empty_like(utm_x)
    longitude = np.empty_like(utm_y)
    utm_x = npa(utm_x)
    utm_y = npa(utm_y)

    if isinstance(zone_number, Iterable):
        zone_set = set(zone_number)
        for zone_num in zone_set:
            utm_source_str = f"EPSG:326{zone_num:02}"
            ind = zone_number == zone_num
            transformer = Transformer.from_crs(utm_source_str, "EPSG:4326")
            latitude[ind], longitude[ind] = transformer.transform(utm_x[ind], utm_y[ind])
    else:
        utm_source_str = f"EPSG:326{zone_number:02}"
        transformer = Transformer.from_crs(utm_source_str, "EPSG:4326")
        latitude, longitude = transformer.transform(utm_x, utm_y)
    return latitude, longitude
