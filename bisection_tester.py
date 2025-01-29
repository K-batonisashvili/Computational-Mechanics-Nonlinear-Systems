"""
Project:        Bisection Method Tester
Author:         Kote Batonisashvili
Description:    Assignment 1. Part of warm-up section. This is a tester which will take in 2 
                numbers and see if the bisection method is working.
"""

# Standard Imports
import os
import sys
from bisection_warm_up import algorithm


def main():
    """
    This is the main function which runs the different testers.
    first_num and second_num are user inputs which define the left bound and right bound.
    test_selector selects 5 example functions which run through the biseciton algorithm. 
    This should return the roots/zeros of the bisection method. 
    """
    try:
        first_num = int(input('Please enter the left part of the interval: '))
        second_num = int(input('Please enter the right part of the interval: '))

        #Error handling for left interval being smaller than right.
        if first_num > second_num:
            raise ValueError("Error: Please make sure the left interval is smaller than the right")

    #Error handling to ensure the input was a number.
    except ValueError as e:  
        print("Error: Please make sure you are using integers for your values")

    try:
        test_selector = int(input('Please type 1 - 5 to select a test for your chosen interval: '))
        if test_selector > 5 or test_selector < 1:
            raise RuntimeError("Error: Your number was either too high or too low. Please select only from tests 1 through 5. Please type just the individual number for each test.")
        elif test_selector == 1:
            test1(first_num, second_num)
        elif test_selector == 2:
            test2(first_num, second_num)
        elif test_selector == 3:
            test3(first_num, second_num)
        elif test_selector == 4:
            test4(first_num, second_num)
        elif test_selector == 5:
            test5(first_num, second_num)

    except ValueError as e:  
        print("Error: Please make sure you are using integers for your values")

   

def test1(first_num, second_num):
    """
    Pre-determined test 1 for the bisection algorithm.
    Basic template parabolic function. 2 Zeros.
    """
    print("------------------------------------------------------------------------------------------------------------")
    print("Test 1 function: x^2 + x - 5")
    func = lambda x: x**2 + x - 5
    bisection_result = algorithm(func, first_num, second_num)
    print(f"After running through the bisection algorithm, we see zero's at the following coordinate: {bisection_result}")
    print("------------------------------------------------------------------------------------------------------------")




def test2(first_num, second_num):
    """
    Pre-determined test 2 for the bisection algorithm.
    Basic 3rd order function. Single zero. 
    """
    print("------------------------------------------------------------------------------------------------------------")
    print("Test 2 function: x^3 + x^2 + x - 5")
    func = lambda x: x**3 + x**2 + x - 5 
    bisection_result = algorithm(func, first_num, second_num)
    print(f"After running through the bisection algorithm, we see zero's at the following coordinate: {bisection_result}")
    print("------------------------------------------------------------------------------------------------------------")




def test3(first_num, second_num):
    """
    Pre-determined test 3 for the bisection algorithm.
    Linear function with a single zero.
    """
    print("------------------------------------------------------------------------------------------------------------")
    print("Test 3 function: 7x - 7")
    func = lambda x: 7*x - 7
    bisection_result = algorithm(func, first_num, second_num)
    print(f"After running through the bisection algorithm, we see zero's at the following coordinate: {bisection_result}")
    print("------------------------------------------------------------------------------------------------------------")




def test4(first_num, second_num):
    """
    Pre-determined test 4 for the bisection algorithm.
    Test 4 is a basic spring mechanics problem where the spring has pre-determined constant of 4 with adjustable displacement.
    Single zero or root for solution.
    """
    k = 4
    print("------------------------------------------------------------------------------------------------------------")
    print("Test 4 function is composed of a basic spring mechanics problem where F = -kx")
    print("Test 4 function has the following adjustable parameters: Displacement")
    func = lambda x: -k*x
    bisection_result = algorithm(func, first_num, second_num)
    print(f"After running through the bisection algorithm, we see zero's at the following coordinate: {bisection_result}")
    print("------------------------------------------------------------------------------------------------------------")




def test5(first_num, second_num):
    """
    Pre-determined test 5 for the bisection algorithm.
    Test 5 is an equilibrium equation where 2 weights are hanging off of a pulley.
    There are 2 solutions to this (Realistically only 1 because weight can't be negative).
    """
    print("------------------------------------------------------------------------------------------------------------")
    print("Test 5 function is composed of a weight-pulley mechanics problem 2 weights are suspended via rope on a pulley")
    print("The equilibrium point where the two weights are stable is modeled with the following equation: -2x^2 + 3x - 12")
    func = lambda x: (-2*(x**2)) + 3*x + 12
    bisection_result = algorithm(func, first_num, second_num)
    print(f"After running through the bisection algorithm, we see zero's at the following coordinate: {bisection_result}")
    print("------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    """
    If statement which runs the main function block
    """
    main()
