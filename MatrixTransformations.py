from math import sin
from math import cos
from math import radians
from operator import matmul
import numpy as np
from numpy import linalg as la  # use la.inv() to invert a matrix


# Returns the product of a matrix and vector. The vector length must match the dimension of the matrix.
def get_vector_matrix_product(vector, matrix):
    return np.matmul(matrix, vector)


# Returns the product of two matrices. The order of multiplication is m1 * m2
def get_matrix_product(m1, m2):
    return np.matmul(m1, m2)


def get_matrix_inverse(m1):
    return la.inv(m1)


# elementary rotation, and displacement matrices -------------------------------------------------------------------------------------

# theta (in degrees) is the angle for "roll" and pivots around the z-axis
def get_theta_matrix(theta):
    theta = radians(theta)
    r1 = [cos(theta), -sin(theta), 0, 0]
    r2 = [sin(theta), cos(theta), 0, 0]
    r3 = [0, 0, 1, 0]
    r4 = [0, 0, 0, 1]
    theta_matrix = np.array([r1, r2, r3, r4])
    return theta_matrix


# phi (in degrees) is the angle for pitch and pivots around the y-axis
def get_phi_matrix(phi):
    phi = radians(phi)
    r1 = [cos(phi), 0, sin(phi), 0]
    r2 = [0, 1, 0, 0]
    r3 = [-sin(phi), 0, cos(phi), 0]
    r4 = [0, 0, 0, 1]
    phi_matrix = np.array([r1, r2, r3, r4])
    return phi_matrix


# psi (in degrees) is the angle for yaw and pivots around the x-axis
def get_psi_matrix(psi):
    psi = radians(psi)
    r1 = [1, 0, 0, 0]
    r2 = [0, cos(psi), -sin(psi), 0]
    r3 = [0, sin(psi), cos(psi), 0]
    r4 = [0, 0, 0, 1]
    psi_matrix = np.array([r1, r2, r3, r4])
    return psi_matrix


# returns the displacement transformation matrix
def get_t_matrix(x, y, z):
    r1 = [1, 0, 0, x]
    r2 = [0, 1, 0, y]
    r3 = [0, 0, 1, z]
    r4 = [0, 0, 0, 1]
    t_matrix = np.array([r1, r2, r3, r4])
    return t_matrix


# returns the augmented displacement transform matrix
# a,b,c represet the positions of the 1s for x,y,z respectively, respecting sign
def get_modified_t_matrix(a, b, c, x, y, z):
    r1 = [0, 0, 0, x]
    r1[abs(a) - 1] = a / abs(a)
    r2 = [0, 0, 0, y]
    r2[abs(b) - 1] = b / abs(b)
    r3 = [0, 0, 0, z]
    r3[abs(c) - 1] = c / abs(c)
    r4 = [0, 0, 0, 1]
    t_matrix = np.array([r1, r2, r3, r4])
    return t_matrix


# gets a TRPY transformation matrix
def get_trpy_matrix(x, y, z, theta, phi, psi):
    theta_matrix = get_theta_matrix(theta)
    phi_matrix = get_phi_matrix(phi)
    psi_matrix = get_psi_matrix(psi)
    t_matrix = get_t_matrix(x, y, z)
    m1 = np.matmul(theta_matrix, phi_matrix)
    m2 = np.matmul(m1, psi_matrix)
    m3 = m2 + np.array([[0, 0, 0, x], [0, 0, 0, y], [0, 0, 0, z], [0, 0, 0, 0]])
    return m3


# elementary inverse displacement and rotation matrices -----------------------------------------------------------------------------------

# get inverse displacement matrix
def get_inv_t_matrix(x, y, z):
    r1 = [1, 0, 0, -x]
    r2 = [0, 1, 0, -y]
    r3 = [0, 0, 1, -z]
    r4 = [0, 0, 0, 1]
    inv_t_matrix = np.array([r1, r2, r3, r4])
    return inv_t_matrix


# get inverse rotation matrix for theta (z-axis)
def get_inv_theta_matrix(theta):
    theta = radians(theta)
    r1 = [cos(theta), sin(theta), 0, 0]
    r2 = [-sin(theta), cos(theta), 0, 0]
    r3 = [0, 0, 1, 0]
    r4 = [0, 0, 0, 1]
    inv_theta_matrix = np.array([r1, r2, r3, r4])
    return inv_theta_matrix


# get inverse rotation matrix for phi (y-axis)
def get_inv_phi_matrix(phi):
    phi = radians(phi)
    r1 = [cos(phi), 0, -sin(phi), 0]
    r2 = [0, 1, 0, 0]
    r3 = [sin(phi), 0, cos(phi), 0]
    r4 = [0, 0, 0, 1]
    inv_phi_matrix = np.array([r1, r2, r3, r4])
    return inv_phi_matrix


# get inverse rotation matrix for psi (x-axis)
def get_inv_psi_matrix(psi):
    psi = radians(psi)
    r1 = [1, 0, 0, 0]
    r2 = [0, cos(psi), sin(psi), 0]
    r3 = [0, -sin(psi), cos(psi), 0]
    r4 = [0, 0, 0, 1]
    inv_psi_matrix = np.array([r1, r2, r3, r4])
    return inv_psi_matrix


# gets inverse trpy transformation matrix
def get_inv_trpy_matrix(x, y, z, theta, phi, psi):
    inv_theta_matrix = get_inv_theta_matrix(theta)
    inv_phi_matrix = get_inv_phi_matrix(phi)
    inv_psi_matrix = get_inv_psi_matrix(psi)

    m1 = np.matmul(inv_psi_matrix, inv_phi_matrix)
    m2 = np.matmul(m1, inv_theta_matrix)

    theta = radians(theta)
    phi = radians(phi)
    psi = radians(psi)
    alpha = -x * cos(theta) * cos(phi) - y * sin(theta) * cos(phi) + z * sin(phi)
    beta = x * (sin(theta) * cos(psi) - cos(theta) * sin(phi) * sin(psi)) - y * (
            cos(theta) * cos(psi) + sin(theta) * sin(phi) * sin(psi)) - z * cos(phi) * sin(psi)
    gamma = -x * (sin(theta) * sin(psi) + cos(theta) * sin(phi) * cos(psi)) + y * (
            cos(theta) * sin(psi) - sin(theta) * sin(phi) * cos(psi)) - z * cos(phi) * cos(psi)
    m3 = m2 + np.array([[0, 0, 0, alpha], [0, 0, 0, beta], [0, 0, 0, gamma], [0, 0, 0, 0]])
    return m3


# gets trpy transformation matrix in degrees
def get_trpy_matrixd(x, y, z, theta, phi, psi):
    th = radians(theta)
    ph = radians(phi)
    ps = radians(psi)
    return get_trpy_matrix(x, y, z, th, ph, ps)


# example data from exercise 2.2
"""
q21 = get_trpy_matrix(-80,50,30,10,-20,-30)
print("Q21")
print(q21)
q31_inv = get_inv_trpy_matrix(70,60,-20,25,-15,40)
print("Q31^-1")
print(q31_inv)
Pr2 = np.array([2,-5,3,1])
q23 = get_matrix_product(q31_inv,q21)
print("Q23")
print(q23)
Pr3 = get_vector_matrix_product(Pr2,q23)
print("Pr3")
print(Pr3)
"""
