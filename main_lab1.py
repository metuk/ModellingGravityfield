import grates
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


n = np.load("normals_2008-05.rightHandSide.npy")
N = np.load("normals_2008-05.npy")
x_hat = np.linalg.inv(N)@n
x_hat = x_hat[:,0]
coeff_unravel = grates.utilities.unravel_coefficients(x_hat)
coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
coefficients.anm = coeff_unravel
degreeeAmplitudes = coefficients.degree_amplitudes(max_order=96)
coefficientAmplitudes = coefficients.coefficient_amplitudes()
spaceGrid = coefficients.to_grid(kernel='potential')

"""
#grates.utilities.kaula_array(Note: kaula_power=4, kaula_scale=1e-14)
#grates.utilities.ravel_coefficients
#grates.utilities.unravel_coefficients
#grates.gravityfield.PotentialCoefficients
#grates.gravityfield.PotentialCoefficients.degree_amplitudes
#grates.gravityfield.PotentialCoefficients.coefficient_amplitudes
#grates.gravityfield.PotentialCoefficients.to_grid(Note: use kernel=‘potential’)
"""




K = grates.utilities.kaula_array(1,96,kaula_power=4, kaula_factor=1e-14)
K_ravel = grates.utilities.ravel_coefficients(K)

#x_hat_kaula = np.linalg.inv(N+np.linalg.inv(np.diag(K_ravel[4:])))@n
#x_hat_kaula = x_hat_kaula[:,0]
#coeff_unravel = grates.utilities.unravel_coefficients(x_hat_kaula)
#coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
#coefficients.anm = coeff_unravel
#degreeeAmplitudes_kaula = coefficients.degree_amplitudes(max_order=96)
#coefficientAmplitudes_kaula = coefficients.coefficient_amplitudes()
#spaceGrid_kaula = coefficients.to_grid(kernel='potential')

#x_hat_kaula = np.linalg.inv(N+100*np.linalg.inv(np.diag(K_ravel[4:])))@n
#x_hat_kaula = x_hat_kaula[:,0]
#coeff_unravel = grates.utilities.unravel_coefficients(x_hat_kaula)
#coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
#coefficients.anm = coeff_unravel
#degreeeAmplitudes_kaula100 = coefficients.degree_amplitudes(max_order=96)
#coefficientAmplitudes_kaula100 = coefficients.coefficient_amplitudes()
#spaceGrid_kaula100 = coefficients.to_grid(kernel='potential')
#
#x_hat_kaula = np.linalg.inv(N+10000*np.linalg.inv(np.diag(K_ravel[4:])))@n
#x_hat_kaula = x_hat_kaula[:,0]
#coeff_unravel = grates.utilities.unravel_coefficients(x_hat_kaula)
#coefficients = grates.gravityfield.PotentialCoefficients(max_degree=96)
#coefficients.anm = coeff_unravel
#degreeeAmplitudes_kaula10000 = coefficients.degree_amplitudes(max_order=96)
#coefficientAmplitudes_kaula10000 = coefficients.coefficient_amplitudes()
#spaceGrid_kaula10000 = coefficients.to_grid(kernel='potential')
#
#
#
#a = np.flip(np.hstack((np.flip(coefficientAmplitudes,1),coefficientAmplitudes)))
#a[a == 0] = 'nan'
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.pcolor(a)
#plt.xticks()
#ax.set_xticks([0,48,96,144,192])
#ax.set_xticklabels(labels=[96,48,0,48,96])  
#ax.set_yticks([0,24,48,72,96])
#ax.set_yticklabels([96,72,48,24,0])
#ax.set_xlabel('Ordnung m')
#ax.set_ylabel('Grad n')
#ax.set_aspect('equal')
#plt.colorbar(orientation='horizontal', pad=.17)
#plt.savefig("coeff.png", dpi=300)
#
#a = np.flip(np.hstack((np.flip(coefficientAmplitudes_kaula10000,1),coefficientAmplitudes_kaula10000)))
#a[a == 0] = 'nan'
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.pcolor(a)
#plt.xticks()
#ax.set_xticks([0,48,96,144,192])
#ax.set_xticklabels(labels=[96,48,0,48,96])  
#ax.set_yticks([0,24,48,72,96])
#ax.set_yticklabels([96,72,48,24,0])
#ax.set_xlabel('Ordnung m')
#ax.set_ylabel('Grad n')
#ax.set_aspect('equal')
#plt.colorbar(orientation='horizontal', pad=.17)
#plt.savefig("coeff_kaula10000.png", dpi=300)
#
#plt.figure(3)
#ax = plt.axes(projection=ccrs.PlateCarree())
#gridlats = np.linspace(90, -90, num=361)
#gridlons = np.linspace(0, 360, num=721)
#plt.pcolormesh(gridlons, gridlats, spaceGrid.value_array, cmap = "viridis")
#ax.coastlines()
#plt.colorbar(orientation='horizontal', label='Joule/kg')
#plt.savefig("grid.png", dpi=300)
#
#plt.figure(30)
#ax = plt.axes(projection=ccrs.PlateCarree())
#gridlats = np.linspace(90, -90, num=361)
#gridlons = np.linspace(0, 360, num=721)
#plt.pcolormesh(gridlons, gridlats, spaceGrid_kaula10000.value_array, cmap = "viridis")
#ax.coastlines()
#plt.colorbar(orientation='horizontal', label='Joule/kg')
#plt.savefig("grid_kaula10000.png", dpi=300)
#
#plt.figure(20)
#plt.plot(degreeeAmplitudes[1], label='standard')
#plt.plot(degreeeAmplitudes_kaula[1], label=r'$\alpha=1$')
#plt.plot(degreeeAmplitudes_kaula100[1], label=r'$\alpha=100$')
#plt.plot(degreeeAmplitudes_kaula10000[1], label=r'$\alpha=10000$')
#plt.grid()
#plt.legend()
#plt.xlabel('Degree n')
#plt.savefig("degreeAmplitudes.png", dpi=300)
plt.figure(32)
plt.imshow([K_ravel[4:]]*1000)
#plt.savefig("kaula_matrix.png", dpi=300)