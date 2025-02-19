"""
Project:        Newton Method
Author:         Kote Batonisashvili
Description:    Assignment 1.2 main elasto-plastic math portion. This will be used as an import for pytest and notebook.
"""

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt

class ElastoPlasticModel:
    def __init__(self, E, Y_initial, H_iso=0.0, H_kin=0.0):
        """
        Initialize the elasto-plastic model with isotropic and kinematic hardening.
        :param E: Elastic modulus
        :param Y_initial: Initial yield stress
        :param H_iso: Isotropic hardening modulus (H_iso)
        :param H_kin: Kinematic hardening modulus (H_kin)
        """
        self.E = E
        self.Y_initial = Y_initial  # σ_y0
        self.H_iso = H_iso  # H_iso
        self.H_kin = H_kin  # H_kin
        
        self.sigma = 0.0  # σ
        self.eps_p = 0.0  # ε_p
        self.alpha = 0.0  # (backstress)
        self.epsilon_p_n = 0.0  # ε_p_n (equivalent plastic strain)
        self.epsilon_total = 0.0  # Total strain tracker
        self.kinematic_flag = H_kin > 0 # Flag for kinematic hardening
    
    def update_step(self, delta_eps):
        """
        Update step for elastroplastic amterial with built in logic for isotropic and kinematic hardening.
        :param delta_eps: Strain increment which needs to be specified by user when calling this method.
        """
        
        self.epsilon_total += delta_eps  # Track total strain

        # Elastic predictor
        sigma_trial = self.sigma + (self.E * delta_eps)

        # Kinematic Hardening
        if self.kinematic_flag:
            Y_n = self.Y_initial + self.H_kin * self.epsilon_p_n  # Update yield stress
            eta_trial = sigma_trial - self.alpha  # Stress relative to backstress
            phi_trial = abs(eta_trial) - Y_n  # Yield function
            
            if phi_trial > 0:  # Plastic deformation occurs
                delta_epsilon_p = phi_trial / (self.E + self.H_kin)  # Plastic strain increment
                self.epsilon_p_n += delta_epsilon_p  # Update plastic strain
                self.alpha += np.sign(eta_trial) * self.H_kin * delta_epsilon_p  # Backstress update
                self.sigma = sigma_trial - np.sign(eta_trial) * self.E * delta_epsilon_p  # Stress correction
            else:
                self.sigma = sigma_trial  # Elastic step

        # Isotropic Hardening
        else:
            Y_n = self.Y_initial + self.H_iso * self.epsilon_p_n  # Update yield stress
            phi_trial = abs(sigma_trial) - Y_n  # Yield function
            
            if phi_trial > 0:  # Plastic deformation occurs
                delta_epsilon_p = phi_trial / (self.E + self.H_iso)  # Plastic strain increment
                self.epsilon_p_n += delta_epsilon_p  # Update plastic strain
                self.sigma = sigma_trial - np.sign(sigma_trial) * self.E * delta_epsilon_p  # Stress correction
            else:
                self.sigma = sigma_trial  # Elastic step
        
        return self.sigma