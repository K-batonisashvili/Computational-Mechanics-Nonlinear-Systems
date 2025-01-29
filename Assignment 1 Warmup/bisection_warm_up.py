"""
Project:        Bisection Method Warm-Up
Author:         Kote Batonisashvili
Description:    Assignment 1. Part of warm-up section. This is the main file which runs the bisection 
                algorithm which we will be utilizing with the tester file.
"""

# Standard Imports
import os
import sys

TOL = 0.0001
ITER = 10000

def algorithm(func, first_num, second_num):
    """
    Parameters:
    func: Function that is passed to the bisection algorithm.
    first_num: First number selected by the user, left-side of the interval.
    second_num: Second number selected by the user, right-side of the interval.

    Return:
    Returns the variable c, which is the value at which bisection occurs.
    """
    count = 0
    a = first_num
    b = second_num
    while count < ITER:
        c = (a + b) / 2
        f_at_a = func(a)
        f_at_c = func(c)
        if abs(c - a) < TOL or abs(f_at_c) < TOL:
            return c
        elif f_at_a * f_at_c >= 0:
            a = c
        else: 
            b = c
        count+= 1
    raise RuntimeError("Ran out of iterations. Please Try different numbers or change tolerance accordingly.")
