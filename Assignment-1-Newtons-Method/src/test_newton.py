"""
Project:        Newton Method
Author:         Kote Batonisashvili
Description:    Assignment 1.1 main newtonian math portion. This will be used as an import for pytest and notebook.
"""


import pytest
import numpy as np
from newton_method import newtonian

'''
We implement a test function which we already know the roots.
Function1:  R1 = x^3 + y^2, x^2 - y^2
Function2:  R2 = x^2 - y^2, x + y
Jacobian1:  J1 = 3x^2, 2y, 2x, -2y
Jacobian2:  J2 = 2x, -2x, 1, 1

When defining the functions and jacobians, we use x[0] and x[1] because we have lambda x. We must use same variable, cannot have "y".

'''


r1 = lambda x: np.array([x[0]**3 + x[1]**2, x[0]**2 - x[1]**2])
#r2 = lambda x: np.array([x[0]**2 - x[1]**2, x[0] + x[1]])
j1 = lambda x: np.array([[3*x[0]**2, 2*x[1]], [2*x[0], -2*x[1]]])
#j2 = lambda x: np.array([2*x[0] - 2*x[1], 1 + 1])
TOL = 1e-6
ITER = 100


'''
Test 1: Comparing solution to tolerance.
        This test checks to ensure basic solving abilities of newtonian method 
'''
def test_for_solver():
    upper_bound = 10
    lower_bound = -10
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)

'''
Test 2: Testing when convergence is not enough.
        This test will check if the bounds are absolutely huge to see if it errors out.
'''
def test_for_large_bounds():
    upper_bound = 1000
    lower_bound = -1000
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)


'''
Test 3: Test for just upper bound being too far.
        If the lower bound is kept at -10, but upper bound is 9999
'''
def test_for_high_upper_bound():
    upper_bound = 9999
    lower_bound = -10
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)

'''
Test 4: Test for just lower bound being too far.
        If the upper bound is kept at 10, but lower bound is -9999
'''
def test_for_high_lower_bound():
    lower_bound = -9999
    upper_bound = 10
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)

'''
Test 5: Test if bounds are incorrect.
        If lower bound > upper bound
'''
def test_for_incorrect_bounds():
    upper_bound = 8
    lower_bound = 10
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)

'''
Test 6: Test if solution is outside the bounds.
        Testing to see if root is outside the presented bounds
'''
def test_for_outside_of_bounds():
    upper_bound = 50
    lower_bound = 5
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-8)

'''
Test 7: Test if tolerance is too low.
        Testing to see if tolerance is set to extremely sensitive numbers.
'''
def test_for_sensitive_tolerance():
    upper_bound = 10
    lower_bound = -10
    sol = newtonian(r1, j1, lower_bound, upper_bound, TOL, ITER)
    actual_roots = np.array([-1,1])
    assert np.allclose(sol, actual_roots, atol=1e-16)