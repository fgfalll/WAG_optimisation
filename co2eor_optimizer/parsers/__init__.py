"""Data parsers for CO2 EOR Optimizer"""
from .las_parser import parse_las
from .eclipse_parser import parse_eclipse

__all__ = ['parse_las', 'parse_eclipse']