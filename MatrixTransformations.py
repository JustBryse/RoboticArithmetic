from math import sin
from math import cos
from math import radians
from operator import matmul
import numpy as np
from numpy import linalg as la # use la.inv() to invert a matrix

# Returns the product of a matrix and vector. The vector length must match the dimension of the matrix.
def get_vector_matrix_product(vector,matrix):
    return np.matmul(matrix,vector)

# Returns the product of two matrices. The order of multiplication is m1 * m2
def get_matrix_product(m1,m2):
    return np.matmul(m1,m2)

# elementary rotation, and displacement matrices -------------------------------------------------------------------------------------

# theta (in degrees) is the angle for "roll" and pivots around the z-axis
def get_theta_matrix(theta):
    theta = radians(theta)
    r1 = [cos(theta),-sin(theta),0,0]
    r2 = [sin(theta),cos(theta),0,0]
    r3 = [0,0,1,0]
    r4 = [0,0,0,1]
    theta_matrix = np.array([r1,r2,r3,r4])
    return theta_matrix

# phi (in degrees) is the angle for pitch and pivots around the y-axis
def get_phi_matrix(phi):
    phi = radians(phi)
    r1 = [cos(phi),0,sin(phi),0]
    r2 = [0,1,0,0]
    r3 = [-sin(phi),0,cos(phi),0]
    r4 = [0,0,0,1]
    phi_matrix = np.array([r1,r2,r3,r4])
    return phi_matrix

# psi (in degrees) is the angle for yaw and pivots around the x-axis
def get_psi_matrix(psi):
    psi = radians(psi)
    r1 = [1,0,0,0]
    r2 = [0,cos(psi),-sin(psi),0]
    r3 = [0,sin(psi),cos(psi),0]
    r4 = [0,0,0,1]
    psi_matrix = np.array([r1,r2,r3,r4])
    return psi_matrix

# returns the displacement transformation matrix
def get_t_matrix(x,y,z):
    r1 = [1,0,0,x]
    r2 = [0,1,0,y]
    r3 = [0,0,1,z]
    r4 = [0,0,0,1]
    t_matrix = np.array([r1,r2,r3,r4])
    return t_matrix

# gets a TRPY transformation matrix
def get_trpy_matrix(x,y,z,theta,phi,psi):
    theta_matrix = get_theta_matrix(theta)
    phi_matrix = get_phi_matrix(phi)
    psi_matrix = get_psi_matrix(psi)
    t_matrix = get_t_matrix(x,y,z)
    m1 = np.matmul(theta_matrix,phi_matrix)
    m2 = np.matmul(m1,psi_matrix)
    m3 = m2 + np.array([[0,0,0,x],[0,0,0,y],[0,0,0,z],[0,0,0,0]])
    return m3

# Gets a forward transform elementary matrix. Inverting the returned matrix gives the matrix inverse.
def get_forward_elementary_matrix(x,y,z,theta,phi,psi):

    tm = get_t_matrix(x,y,z)
    thetam = get_theta_matrix(theta)
    phim = get_phi_matrix(phi)
    psim = get_psi_matrix(psi)

    m0 = np.matmul(tm,thetam)
    m1 = np.matmul(m0,phim)
    m2 = np.matmul(m1,psim)
    return m2

# elementary inverse displacement and rotation matrices -----------------------------------------------------------------------------------

# get inverse displacement matrix
def get_inv_t_matrix(x,y,z):
    r1 = [1,0,0,-x]
    r2 = [0,1,0,-y]
    r3 = [0,0,1,-z]
    r4 = [0,0,0,1]
    inv_t_matrix = np.array([r1,r2,r3,r4])
    return inv_t_matrix

# get inverse rotation matrix for theta (z-axis)
def get_inv_theta_matrix(theta):
    theta = radians(theta)
    r1 = [cos(theta),sin(theta),0,0]
    r2 = [-sin(theta),cos(theta),0,0]
    r3 = [0,0,1,0]
    r4 = [0,0,0,1]
    inv_theta_matrix = np.array([r1,r2,r3,r4])
    return inv_theta_matrix

# get inverse rotation matrix for phi (y-axis)
def get_inv_phi_matrix(phi):
    phi = radians(phi)
    r1 = [cos(phi),0,-sin(phi),0]
    r2 = [0,1,0,0]
    r3 = [sin(phi),0,cos(phi),0]
    r4 = [0,0,0,1]
    inv_phi_matrix = np.array([r1,r2,r3,r4])
    return inv_phi_matrix

# get inverse rotation matrix for psi (x-axis)
def get_inv_psi_matrix(psi):
    psi = radians(psi)
    r1 = [1,0,0,0]
    r2 = [0,cos(psi),sin(psi),0]
    r3 = [0,-sin(psi),cos(psi),0]
    r4 = [0,0,0,1]
    inv_psi_matrix = np.array([r1,r2,r3,r4])
    return inv_psi_matrix

# gets inverse trpy transformation matrix
def get_inv_trpy_matrix(x,y,z,theta,phi,psi):
    inv_theta_matrix = get_inv_theta_matrix(theta)
    inv_phi_matrix = get_inv_phi_matrix(phi)
    inv_psi_matrix = get_inv_psi_matrix(psi)

    m1 = np.matmul(inv_psi_matrix,inv_phi_matrix)
    m2 = np.matmul(m1,inv_theta_matrix)

    theta = radians(theta)
    phi = radians(phi)
    psi = radians(psi)
    alpha = -x*cos(theta)*cos(phi) -y*sin(theta)*cos(phi) + z*sin(phi)
    beta = x*(sin(theta)*cos(psi)-cos(theta)*sin(phi)*sin(psi)) - y*(cos(theta)*cos(psi)+sin(theta)*sin(phi)*sin(psi)) -z*cos(phi)*sin(psi)
    gamma = -x*(sin(theta)*sin(psi)+cos(theta)*sin(phi)*cos(psi))+y*(cos(theta)*sin(psi)-sin(theta)*sin(phi)*cos(psi))-z*cos(phi)*cos(psi)
    m3 = m2 + np.array([[0,0,0,alpha],[0,0,0,beta],[0,0,0,gamma],[0,0,0,0]])
    return m3


thetam = get_theta_matrix(30)
phim = get_phi_matrix(10)
psim = get_psi_matrix(260)
tm = get_t_matrix(50,-100,30)
m0 = np.matmul(phim,tm)
m1 = np.matmul(m0,psim)
m2 = np.matmul(m1,thetam)
q_zxty_f = m2
print("q_axty_f")
print(q_zxty_f)
print("inv_q_axty_f")
print(la.inv(q_zxty_f))