"""
Project:        ElastoPlastic Method
Author:         Kote Batonisashvili
Description:    Assignment 1.2 test file which has test cases to ensure math/main code has no errors.
"""

import pytest
import numpy as np
from ElastoPlastic import ElastoPlasticModel

def test_elastic():
    """Test that the material follows Hooke’s Law in the elastic region."""
    model = ElastoPlasticModel(E=1000, Y_initial=10, H_iso=0, H_kin=0)
    strain_increment = 0.005  # Small strain in elastic range
    stress = model.update_step(strain_increment)
    
    # Expected stress
    expected_stress = 1000 * 0.005
    assert np.isclose(stress, expected_stress, atol=1e-6), "Hooke's Law incorrect"

def test_yielding():
    """Test that stress does not exceed the yield stress in the plastic range."""
    model = ElastoPlasticModel(E=1000, Y_initial=10, H_iso=0, H_kin=0)
    
    # Apply large strain increment to induce yielding
    strain_increment = 0.05
    stress = model.update_step(strain_increment)
    
    # Stress should not exceed the yield limit
    assert stress <= 10, "Stress exceeded the yield stress in plastic range"

def test_isotropic_hardening():
    """isotropic hardening"""
    model = ElastoPlasticModel(E=1000, Y_initial=10, H_iso=50, H_kin=0)
    
    # strain increments delta epsilon
    model.update_step(0.02)  # Should cause yielding
    model.update_step(0.02)
    
    # Expected yield stress
    expected_yield = 10 + 50 * model.epsilon_p_n
    assert model.sigma <= expected_yield, "Isotropic hardening not applied correctly"

def test_kinematic_hardening():
    """Test kinematic hardening."""
    model = ElastoPlasticModel(E=1000, Y_initial=10, H_iso=0, H_kin=50)
    
    # strain increments delta epsilon
    model.update_step(0.02)  # Should cause yielding
    model.update_step(0.02)
    
    # Backstress should increase with plastic strain
    expected_alpha = 50 * model.epsilon_p_n
    assert np.isclose(model.alpha, expected_alpha, atol=1e-6), "Kinematic hardening incorrect"

def test_return_mapping():
    """Return mapping correctly projects stress back onto the yield surface."""
    model = ElastoPlasticModel(E=1000, Y_initial=10, H_iso=100, H_kin=0)
    
    # Apply large strain increments to induce plasticity
    model.update_step(0.05)
    model.update_step(0.05)

    # The stress should be within the updated yield surface
    updated_yield_stress = 10 + 100 * model.epsilon_p_n
    assert np.isclose(model.sigma, updated_yield_stress, atol=1e-6), "Return mapping failed"
