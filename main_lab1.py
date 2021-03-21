import grates
import numpy as np
import matplotlib.pyplot as plt
n = np.load("normals_2008-05.rightHandSide.npy")
N = np.load("normals_2008-05.npy")
x_hat = np.linalg.inv(N)@n
x_hat = x_hat[:,0]
coeff_unravel = grates.utilities.unravel_coefficients(x_hat)
coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
coefficients.anm = coeff_unravel
print(coefficients.anm.shape)
degreeeAmplitudes = coefficients.degree_amplitudes(max_order=96)
coefficientAmplitudes = coefficients.coefficient_amplitudes()
spaceGrid = coefficients.to_grid(kernel='potential')

"""
grates.utilities.kaula_array(Note: kaula_power=4, kaula_scale=1e-14)
grates.utilities.ravel_coefficients
grates.utilities.unravel_coefficients
grates.gravityfield.PotentialCoefficients
grates.gravityfield.PotentialCoefficients.degree_amplitudes
grates.gravityfield.PotentialCoefficients.coefficient_amplitudes
grates.gravityfield.PotentialCoefficients.to_grid(Note: use kernel=‘potential’)
"""

plt.figure(1)
plt.plot(degreeeAmplitudes[1])
plt.figure(2)
plt.pcolor(coefficientAmplitudes.T)
plt.figure(3)
plt.pcolor(spaceGrid.value_array)


K = grates.utilities.kaula_array(1,96,kaula_power=4, kaula_factor=1e-14)
K_ravel = grates.utilities.ravel_coefficients(K)
#fill0_A = np.zeros((9405,4))
#fill0_B = np.zeros((4,9409))
#N = np.vstack((fill0_A.T,N))
#print(N.shape)
#N = np.hstack((fill0_B.T,N))
#fill0_n = np.zeros((4,1))
#print(n.shape, fill0_n.shape)
#n = np.vstack((fill0_n,n))
x_hat_kaula = np.linalg.inv(N+10000*np.linalg.inv(np.diag(K_ravel[4:])))@n
x_hat_kaula = x_hat_kaula[:,0]

coeff_unravel = grates.utilities.unravel_coefficients(x_hat_kaula)
coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
coefficients.anm = coeff_unravel
degreeeAmplitudes = coefficients.degree_amplitudes(max_order=96)
coefficientAmplitudes = coefficients.coefficient_amplitudes()
spaceGrid = coefficients.to_grid(kernel='potential')
plt.figure(10)
plt.plot(degreeeAmplitudes[1])
plt.figure(20)
plt.pcolor(coefficientAmplitudes.T)
plt.figure(30)
plt.pcolor(spaceGrid.value_array)