# =============================== Library imports ===============================

# Plotting and other visualization
import matplotlib.pyplot as plt

# Arrays, array arithmetic
import numpy as np

# Performance measurement
import time

# Compute acceleration through JIT compilation
import numba as nb

# Used to measure memory consumption
import psutil
process = psutil.Process()

# File I/O, working directory management
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_list = os.listdir(dir_path)
os.chdir(dir_path)

# =============================== Model functions ===============================

# General argument signature for functions responsible for model execution (not all may be used):
# E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor

# The variables are:
# E, B, and J - Electric, magnetic, and current density fields (4d float32 arrays)
# conductorFieldSel - Whether each field component in E and J is associated with a conducting volume (4d bool array)
# conductor - Whether each cell in the Spatial Reference Grid (SRG) is conducting (3d bool array)
# xp - Particle positions (3d float32 array)
# pp - Particle momenta (3d float32 array)
# active - Whether each particle is currently alive (1d bool array)
# q - Charge of each particle species (1d float32 array)
# m - Mass of each particle species (1d float32 array)
# dx - Grid division size (actually just 1 due to nondimensionalization)
# dt - Timestep size (NOT just 1)
# conductionFactor - The e^(...) conduction factor term derived in the thesis

# Evolves the electric and magnetic fields by one timestep
@nb.njit(error_model="numpy",cache=True,parallel=True,fastmath=True)
def UpdateEMF(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor):

    # *** Set any current inside conductors to zero ***
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            for k in range(J.shape[2]):
                for w in range(J.shape[3]):
                    if conductorFieldSel[i,j,k,w]:

                        J[i,j,k,w] = 0


    # *** Evolve B ***
    # x component
    B[1:, 1:, 1:, 0] = B[1:, 1:, 1:, 0] - dt/dx*(E[1:, 1:, 1:, 2] - E[1:, 0:-1, 1:, 2] - E[1:, 1:, 1:, 1] + E[1:, 1:, 0:-1, 1])

    # y component
    B[1:, 1:, 1:, 1] = B[1:, 1:, 1:, 1] - dt/dx*(E[1:, 1:, 1:, 0] - E[1:, 1:, 0:-1, 0] - E[1:, 1:, 1:, 2] + E[0:-1, 1:, 1:, 2])

    # z component
    B[1:, 1:, 1:, 2] = B[1:, 1:, 1:, 2] - dt/dx*(E[1:, 1:, 1:, 1] - E[0:-1, 1:, 1:, 1] - E[1:, 1:, 1:, 0] + E[1:, 0:-1, 1:, 0])


    # *** Evolve E ***
    # x component
    E[0:-1, 0:-1, 0:-1, 0] = E[0:-1, 0:-1, 0:-1, 0] + dt/dx*(B[0:-1, 1:, 0:-1, 2] - B[0:-1, 0:-1, 0:-1, 2] - B[0:-1, 0:-1, 1:, 1] + B[0:-1, 0:-1, 0:-1, 1]) - 4 * np.pi * dt * J[0:-1, 0:-1, 0:-1, 0] 
    
    # y component
    E[0:-1, 0:-1, 0:-1, 1] = E[0:-1, 0:-1, 0:-1, 1] + dt/dx*(B[0:-1, 0:-1, 1:, 0] - B[0:-1, 0:-1, 0:-1, 0] - B[1:, 0:-1, 0:-1, 2] + B[0:-1, 0:-1, 0:-1, 2]) - 4 * np.pi * dt * J[0:-1, 0:-1, 0:-1, 1]
    
    # z component
    E[0:-1, 0:-1, 0:-1, 2] = E[0:-1, 0:-1, 0:-1, 2] + dt/dx*(B[1:, 0:-1, 0:-1, 1] - B[0:-1, 0:-1, 0:-1, 1] - B[0:-1, 1:, 0:-1, 0] + B[0:-1, 0:-1, 0:-1, 0]) - 4 * np.pi * dt * J[0:-1, 0:-1, 0:-1, 2]


    # *** Attenuate E inside regions of conductor in accordance with the conduction factor ***
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            for k in range(J.shape[2]):
                for w in range(J.shape[3]):
                    if conductorFieldSel[i,j,k,w]:

                        E[i,j,k,w] = E[i,j,k,w] * (1 - conductionFactor)

    # *** Set E at the boundaries to zero to enforce closed (conducting) boundary conditions ***
    E[:,:,0] = 0
    E[:,0,:] = 0
    E[0,:,:] = 0
    E[:,:,-1] = 0
    E[:,-1,:] = 0
    E[-1,:,:] = 0

    # Return the newly updated electric and magnetic fields
    return (E, B)

# Updates the momenta of the particles in accordance with the timestep size and the local values of the 
# electric and magnetic fields
@nb.njit(error_model="numpy",cache=True,parallel=True,fastmath=True)
def UpdateMomentum(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor):
    # *** Get the number of particlces of each species ***
    n_p = xp.shape[2]
    n_sp = xp.shape[1]


    # *** Calculate the B field averaged to the centers of the cells ***
    B_avg = np.zeros(B.shape, dtype=np.float32)
    
    # x, y, and z components
    B_avg[:, 0:-1, 0:-1, 0] = 0.25*(B[:, 0:-1, 0:-1, 0] + B[:, 1:, 0:-1, 0] + B[:, 0:-1, 1:, 0] + B[:, 1:, 1:, 0])
    B_avg[0:-1, :, 0:-1, 1] = 0.25*(B[0:-1, :, 0:-1, 1] + B[1:, :, 0:-1, 1] + B[0:-1, :, 1:, 1] + B[1:, :, 1:, 1])
    B_avg[0:-1, 0:-1, :, 2] = 0.25*(B[0:-1, 0:-1, :, 2] + B[1:, 0:-1, :, 2] + B[0:-1, 1:, :, 2] + B[1:, 1:, :, 2])           


    # *** Calculate the B field averaged to the centers of the cells ***
    E_avg = np.zeros((E.shape[0], E.shape[1], E.shape[2], 3), dtype=np.float32)

    # x, y, and z components
    E_avg[0:-1, :, :, 0] = 0.5*(E[0:-1, :, :, 0] + E[1:, :, :, 0])
    E_avg[:, 0:-1, :, 1] = 0.5*(E[:, 0:-1, :, 1] + E[:, 1:, :, 1])
    E_avg[:, :, 0:-1, 2] = 0.5*(E[:, :, 0:-1, 2] + E[:, :, 1:, 2])
    
    # *** Update the momentum for each particle ***
    for sp in range(n_sp):

        # For each particle p in the species sp, update its momentum
        # Calculates new particle momenta in parallel because there's zero
        # dependence between loop iterations
        for p in nb.prange(n_p):
            
            # Only update the momentum if this particle is active (i.e. not fluxed back into
            # the model after colliding with a conductor)
            if active[sp,p]:

                # *** Interpolate magnetic field ***
                BInterp = np.zeros(3, dtype=np.float32)
                EInterp = np.zeros(3, dtype=np.float32)

                # *** Calculate SRG coordinates ***
                i_srg = int(np.floor((xp[0,sp,p] + dx/2)))
                j_srg = int(np.floor((xp[1,sp,p] + dx/2)))
                k_srg = int(np.floor((xp[2,sp,p] + dx/2)))

                # *** Calculate the particle's position with respect to the local origin ***
                xLocal = (xp[0, sp, p] - (i_srg * dx))
                yLocal = (xp[1, sp, p] - (j_srg * dx))
                zLocal = (xp[2, sp, p] - (k_srg * dx))

                # *** Compute the interpolated field values ***
                for offsetX in range(-1, 1):
                    for offsetY in range(-1, 1):
                        for offsetZ in range(-1, 1):

                            # Calculate the fractional contribution of this cell to the value of the field experienced by the particle
                            # (This is done using volume weighting)
                            w = (1/2 + (offsetX * 2 + 1) * xLocal) * (1/2 + (offsetY * 2 + 1) * yLocal) * (1/2 + (offsetZ * 2 + 1) * zLocal)

                            # Add each field contribution onto the interpolated values of the fields
                            for fieldComponent in range(3):
                                BInterp[fieldComponent] = BInterp[fieldComponent] + w * B_avg[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ, fieldComponent]
                                EInterp[fieldComponent] = EInterp[fieldComponent] + w * E_avg[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ, fieldComponent]

                # *** Using the interpolated field values, calculate the Lorentz force on the particle ***
                # *** and update the momentum accordingly ***

                # Gamma factor
                g0 = np.sqrt(1 + np.sum(pp[:,sp,p]**2, axis=0) / (m[sp])**2)

                # Velocity and force values
                v = pp[:, sp, p] / (m[sp] * g0)
                F = q[sp] * (EInterp + np.cross(v.ravel(), BInterp))

                # Updated momentum
                pp[:, sp, p] = pp[:, sp, p] + F * dt
    
    # Return the newly updated particle momenta
    return pp

# Displaces the particles and induces the corresponding currents on the electromagnetic grids
@nb.njit(error_model="numpy",cache=True,fastmath=True)
def UpdatePosition(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor):

    # *** Get the number of particlces of each species ***
    n_p = xp.shape[2]
    n_sp = xp.shape[1]

    # Reset all currents deposited from the last timestep
    J[:, :, :, :] = 0

    # Array to store the positions, times, and changes to the SRG indicies associated with each
    # possible new position of the particle
    possibleNewPos = np.zeros((4, 7))

    # Loop over each species
    for sp in range(n_sp):
        # *** Calculate how far each particle needs to be displaced using forward euler ***

        # Gamma factor
        g0 = np.sqrt(1 + np.sum(pp[:,sp,:]**2, axis=0) / (m[sp])**2)

        # Velocity and new position
        v = pp[:, sp, :] / (m[sp] * g0)
        xf = xp[:, sp, :] + v * dt

        # *** Move each particle p within this species sp, inducing currents on the grid (J) along the way ***
        for p in range(n_p):

            # *** Only displace the particle if it is active ***
            if active[sp,p]:

                # *** Locate particle in the Spatial Reference Grid (SRG) ***
                srg = np.array([np.floor(xp[0, sp, p] + 0.5), np.floor(xp[1, sp, p] + 0.5), np.floor(xp[2, sp, p] + 0.5)], dtype=nb.int32)


                while True:

                    # *** Calculate local cell coordinates ***
                    localPos = xp[:, sp, p] - srg
                    possibleNewPos[:,:] = 0

                    # Compute each new intercept position, time
                    for intercept in range(3):

                        # x, y, and z position of corresponding x, y, and z intercepts
                        possibleNewPos[intercept+1, intercept] = (srg[intercept] - 0.5 + (pp[intercept, sp, p] > 0) )
                        possibleNewPos[intercept+1, intercept + 4] = np.sign(pp[intercept, sp, p])

                        # yz, xz, and xy positions of x, y, and z intercepts
                        for dim in range(3):
                            if dim == intercept:
                                continue

                            possibleNewPos[intercept+1, dim] = xp[dim, sp, p] + (pp[dim, sp, p] / pp[intercept, sp, p]) * (possibleNewPos[intercept+1, intercept] - xp[intercept, sp, p])

                        # Determine the time until the particle reaches this intercept
                        possibleNewPos[intercept+1, 3] = abs((possibleNewPos[intercept+1, intercept] - xp[intercept, sp, p]) / v[intercept, p])

                    # Time and position of the particle's final position
                    possibleNewPos[0, 0:3] = xf[:, p]
                    possibleNewPos[0, 3] = np.sqrt(np.sum((xf[:, p] - xp[:, sp, p]) ** 2)) / np.sqrt(np.sum(v[:, p] ** 2))


                    # *** Choose the closest intercept ***
                    if np.all(np.isnan(possibleNewPos[:,3])):
                        # Select the final position if the particle is completely motionless
                        finalPosIdx = 0
                    
                    else:
                        # Loop over the possible new positions and select which one is closest
                        # (measured by which has the minimum time until the particle reaches it)
                        minVal = np.inf

                        for i in range(4):
                            if np.isnan(possibleNewPos[i, 3]):
                                continue

                            else:
                                # If this position is closer than the previous closest,
                                # select it instead.
                                if possibleNewPos[i,3] < minVal:
                                    minVal = possibleNewPos[i,3]
                                    finalPosIdx = i
                    
                    # Displacement vector of the particle
                    delta = possibleNewPos[finalPosIdx, 0:3] - xp[:, sp, p]


                    # *** Compute currents induced on the grid ***

                    # F factors from thesis (algebraic simplification)
                    Fx = 2*localPos[0] + delta[0]
                    Fy = 2*localPos[1] + delta[1]
                    Fz = 2*localPos[2] + delta[2]

                    # K factor
                    K = q[sp] / (4*dt)

                    # x-direction currents for each cell the particle intersects
                    J[srg[0],    srg[1],      srg[2],      0] = J[srg[0],    srg[1],      srg[2],      0] + K * delta[0] * ((1 + Fy) * (1 + Fz) + delta[1] * delta[2] / 3)
                    J[srg[0],    srg[1] - 1,  srg[2],      0] = J[srg[0],    srg[1] - 1,  srg[2],      0] + K * delta[0] * ((1 - Fy) * (1 + Fz) - delta[1] * delta[2] / 3)
                    J[srg[0],    srg[1],      srg[2] - 1,  0] = J[srg[0],    srg[1],      srg[2] - 1,  0] + K * delta[0] * ((1 + Fy) * (1 - Fz) - delta[1] * delta[2] / 3)
                    J[srg[0],    srg[1] - 1,  srg[2] - 1,  0] = J[srg[0],    srg[1] - 1,  srg[2] - 1,  0] + K * delta[0] * ((1 - Fy) * (1 - Fz) + delta[1] * delta[2] / 3)

                    # y-direction currents for each cell the particle intersects
                    J[srg[0],        srg[1],  srg[2],      1] = J[srg[0],        srg[1],  srg[2],      1] + K * delta[1] * ((1 + Fx) * (1 + Fz) + delta[0] * delta[2] / 3)
                    J[srg[0] - 1,    srg[1],  srg[2],      1] = J[srg[0] - 1,    srg[1],  srg[2],      1] + K * delta[1] * ((1 - Fx) * (1 + Fz) - delta[0] * delta[2] / 3)
                    J[srg[0],        srg[1],  srg[2] - 1,  1] = J[srg[0],        srg[1],  srg[2] - 1,  1] + K * delta[1] * ((1 + Fx) * (1 - Fz) - delta[0] * delta[2] / 3)
                    J[srg[0] - 1,    srg[1],  srg[2] - 1,  1] = J[srg[0] - 1,    srg[1],  srg[2] - 1,  1] + K * delta[1] * ((1 - Fx) * (1 - Fz) + delta[0] * delta[2] / 3)

                    # z-direction currents for each cell the particle intersects
                    J[srg[0],        srg[1],      srg[2],  2] = J[srg[0],        srg[1],      srg[2],  2] + K * delta[2] * ((1 + Fx) * (1 + Fy) + delta[0] * delta[1] / 3)
                    J[srg[0] - 1,    srg[1],      srg[2],  2] = J[srg[0] - 1,    srg[1],      srg[2],  2] + K * delta[2] * ((1 - Fx) * (1 + Fy) - delta[0] * delta[1] / 3)
                    J[srg[0],        srg[1] - 1,  srg[2],  2] = J[srg[0],        srg[1] - 1,  srg[2],  2] + K * delta[2] * ((1 + Fx) * (1 - Fy) - delta[0] * delta[1] / 3)
                    J[srg[0] - 1,    srg[1] - 1,  srg[2],  2] = J[srg[0] - 1,    srg[1] - 1,  srg[2],  2] + K * delta[2] * ((1 - Fx) * (1 - Fy) + delta[0] * delta[1] / 3)

                    # Place the particle at the closest of all the potential new positions
                    xp[:, sp, p] = possibleNewPos[finalPosIdx, 0:3]

                    # Update the particle's indicies in the SRG
                    srg = srg + possibleNewPos[finalPosIdx, 4:].astype(np.int32)

                    # Set the particle to be inactive if it is now inside a conductor
                    if conductor[srg[0], srg[1], srg[2]]:
                        active[sp, p] = False
                        break
                    
                    # If this particle is at its fully displaced position,
                    # move on to the next one
                    if finalPosIdx == 0:
                        break


    # Return the updated particle positions, induced currents, and which particles are active
    return (xp, J, active)

# Adds particles back to the computational domain after they have been removed due to collision with a conductor
@nb.njit(error_model="numpy",cache=True)
def FluxParticles(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor):

    # *** Get the number of particlces of each species ***
    n_p = xp.shape[2]
    n_sp = xp.shape[1]


    # *** Look for pairs of postive and negative particles that are currently inactive, ***
    # *** and place them together back into the computational domain ***
    cdomSize = E.shape
    p_e_last = 0

    # *** For each inactive nuclei, see if there is an inactive electron ***
    for p_nuc in range(n_p):
        if ~active[0, p_nuc]:
            for p_e in range(p_e_last, n_p):
                if ~active[1, p_e]:

                    # Set both particles to be active
                    active[0, p_nuc] = True
                    active[1, p_e] = True

                    # Choose a random new position for the particles that is not inside a conductor
                    while True:
                        newPos = np.array([np.random.uniform(dx, cdomSize[0] - dx), np.random.uniform(dx, cdomSize[1] - dx), np.random.uniform(dx, cdomSize[2] - dx)])
                        srg_newPos = np.floor(newPos + 0.5).astype(nb.int32)

                        if ~conductor[srg_newPos[0], srg_newPos[1], srg_newPos[2]]:
                            break
                    
                    # Set the particle positions to the new position
                    xp[:, 0, p_nuc] = newPos
                    xp[:, 1, p_e] = newPos

                    # Give the particles a small amount of initial momentum to make interesting things happen faster
                    pp[:, 0, p_nuc] = np.random.uniform(-0.001,0.001,3)
                    pp[:, 1, p_e] = pp[:, 0, p_nuc] + np.random.uniform(-0.0001,0.0001,3)
                    p_e_last = p_e
                    break
                    
    return (xp, pp, active)

# Calls the previous four functions in sequence tsteps times to evolve the entire model by one timestep.
# The other arguments are:
# E_cath - Electric field to establish the potential difference between the cathode and vacuum chamber walls.
# currentTime - The current timestep index of the overall model
# tsteps - Number of timesteps to execute
@nb.njit(error_model="numpy",cache=True)
def EvolveModel(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor, E_cath, currentTime, tsteps=1):
    
    # Evolve the model tsteps time steps
    for t in range(tsteps):
        # Compute the charge that crossed the interface between the cathode and vacuum chamber
        q_cath = E_cath - E[int(np.round(X.shape[0] / 2)), int(np.round(Y.shape[0] / 2)), 10, 2]
        q_cath += E_cath - E[int(np.round(X.shape[0] / 2))-1, int(np.round(Y.shape[0] / 2)), 10, 2]
        q_cath += E_cath - E[int(np.round(X.shape[0] / 2)), int(np.round(Y.shape[0] / 2))-1, 10, 2]
        q_cath += E_cath - E[int(np.round(X.shape[0] / 2))-1, int(np.round(Y.shape[0] / 2))-1, 10, 2]
        q_cath = q_cath / (4*np.pi*dt)

        # Establish the potential difference between the vacuum chamber and cathode
        E[int(np.round(X.shape[0] / 2)), int(np.round(Y.shape[0] / 2)), 10, 2] = E_cath
        E[int(np.round(X.shape[0] / 2))-1, int(np.round(Y.shape[0] / 2)), 10, 2] = E_cath
        E[int(np.round(X.shape[0] / 2)), int(np.round(Y.shape[0] / 2))-1, 10, 2] = E_cath
        E[int(np.round(X.shape[0] / 2))-1, int(np.round(Y.shape[0] / 2))-1, 10, 2] = E_cath

        # Update the electromagnetic fields
        E, B = UpdateEMF(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor)

        # Flux any particles that have left the domain and have at least one oppositely charged partner
        xp, pp, active = FluxParticles(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor)

        # Update the momentum of the particles
        pp = UpdateMomentum(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor)

        # Update the position of the particles and allow them to induce currents on the EM grid.
        xp, J, active = UpdatePosition(E, B, J, conductorFieldSel, conductor, xp, pp, active, q, m, dx, dt, conductionFactor)

    # Return the updated state of the model
    return E, B, J, xp, pp, q_cath, active, currentTime + tsteps

# Computes and returns the number density distribution of a particular species throughout space
@nb.njit()
def NumberDensity(xp, active, sp):

    # Array to store the number density distribution of the selected particle species
    nDensity = np.zeros(X.shape, dtype=np.float32)

    # Compute the number density contributed by each active particle to the grid
    for p in range(n_p):
        if active[sp,p]:
            i_srg = int(np.floor((xp[0,sp,p] + dx/2) / dx))
            j_srg = int(np.floor((xp[1,sp,p] + dx/2) / dx))
            k_srg = int(np.floor((xp[2,sp,p] + dx/2) / dx))

            xLocal = (xp[0, sp, p] - (i_srg * dx)) / dx
            yLocal = (xp[1, sp, p] - (j_srg * dx)) / dx
            zLocal = (xp[2, sp, p] - (k_srg * dx)) / dx

            for offsetX in range(-1, 1):
                for offsetY in range(-1, 1):
                    for offsetZ in range(-1, 1):
                        w = (1/2 + (offsetX * 2 + 1) * xLocal) * (1/2 + (offsetY * 2 + 1) * yLocal) * (1/2 + (offsetZ * 2 + 1) * zLocal)
                        nDensity[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ] = nDensity[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ] + w
    
    nDensity = nDensity * W

    return nDensity

# Calculates and outputs the distribution of kinetic energy for a particular species.
# Units: [Total kinetic energy] / [grid cell]
@nb.njit()
def KineticEnergy(xp, pp, active, sp):
    # Array to store the kinetic energy distribution
    KEDensity = np.zeros(X.shape, dtype=np.float32)

    # Gamma factor and kinetic energy of each particle
    g0 = np.sqrt(1 + np.sum(pp[:,sp,:]**2, axis=0) / (m[sp])**2)
    K = (g0-1) * m[sp]

    # Compute the kinetic energy contributed by each particle to the grid
    for p in range(n_p):
        if active[sp,p]:
            i_srg = int(np.floor((xp[0,sp,p] + dx/2) / dx))
            j_srg = int(np.floor((xp[1,sp,p] + dx/2) / dx))
            k_srg = int(np.floor((xp[2,sp,p] + dx/2) / dx))

            xLocal = (xp[0, sp, p] - (i_srg * dx)) / dx
            yLocal = (xp[1, sp, p] - (j_srg * dx)) / dx
            zLocal = (xp[2, sp, p] - (k_srg * dx)) / dx

            for offsetX in range(-1, 1):
                for offsetY in range(-1, 1):
                    for offsetZ in range(-1, 1):
                        w = (1/2 + (offsetX * 2 + 1) * xLocal) * (1/2 + (offsetY * 2 + 1) * yLocal) * (1/2 + (offsetZ * 2 + 1) * zLocal)
                        KEDensity[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ] = KEDensity[i_srg + offsetX, j_srg + offsetY, k_srg + offsetZ] + w * K[p]

    return KEDensity

# ========================= Simulation parameters =========================

# *** Output parameters ***

# How many timesteps to execute before writing the model output to a file
outputInterval = 50

# *** Physical parameters ***

# Electromagnetic grid division size (cm)
dx_dim = 0.4 # 5 mm

# Grid voltage (SI)
V_dim = 100000

# Computational domain size (x, y, z)
cdomSize_dim = np.array((22, 22, 51), dtype=np.float32) # cm

# CFL factor [0,1], smaller is slower but yields better results
CFL = 0.75

# Speed of light
c = 29979245800 # cm/s

# Fundamental unit of charge (Fr)
q_e = 4.8*10**(-10)

# Simulation time (ns)
T_dim = 80

# Number of particle species
n_sp = 2

# Number of particles of each species
n_p = 1000000

# Conduction factor (see thesis)
conductionFactor=0.95

# Radius of the vacuum chamber and cathode grid
Rvac_dim = 10 # cm
Rgrid_dim = 5 # cm

# Particle masses (g)
m_e = 9.1*10**-28 # Electrons
m_p = 3.3*10**-24 # Deuterium nuclei

# Mean electron number density
D_e = 10**8 # cm^-3

# ========================= Some Nondimensionalizing ==================

# Convention:
# <variable>_dim - Dimensional quantity
# <variable>0 - Characteristic dimensional quantity
# <variable> - Nondimensional quantity

# Distance
x0 = dx_dim
dx = dx_dim / x0

# Reactor geometry
cdomSize = cdomSize_dim / x0
Rgrid = Rgrid_dim / x0
Rvac = Rvac_dim / x0

# Time step size
t0 = x0 / c
dt_dim = CFL * dx_dim/(c*np.sqrt(3))
dt = dt_dim/t0

# Simulation timesteps
T = T_dim*(10**(-9)) / dt_dim
T = np.round(T).astype(np.int32)

# ========================= Computational domain geometry =========================

# *** Spatial grid ***

x = np.arange(0, cdomSize[0] + dx, dx)
y = np.arange(0, cdomSize[1] + dx, dx)
z = np.arange(0, cdomSize[2] + dx, dx)

[X, Y, Z] = np.meshgrid(x, y, z, indexing='ij')

# *** Fields ***

# All the fields use the indexing scheme:
# [i, j, k, dim], dim is x (0), y (1), or z (2)

# Electric field
E = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3), dtype=np.float32)

# Magnetic field
B = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3), dtype=np.float32)

# Current density
J = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3), dtype=np.float32)

# Whether a field component in a field is associated with a conductor
conductorFieldSel = np.zeros(E.shape, dtype=np.bool_)

# ========================= Conducting boundaries =========================

# Array to store whether each cell in the SRG is conducting
# Three-dimensional array of bools
conductor = np.zeros(X.shape, dtype=np.bool_)

# Edges need to be conducting
conductor[:,:,0] = True
conductor[:,0,:] = True
conductor[0,:,:] = True
conductor[:,:,-1] = True
conductor[:,-1,:] = True
conductor[-1,:,:] = True

# Vacuum chamber walls
conductor[(X - dx/2 - cdomSize[0]/2)**2 + (Y - dx/2 - cdomSize[1]/2)**2 >= Rvac**2] = True

# *** Inner grid (3 circles) ***

# Define the grid circle in the xy-plane
wireSel = (((X - dx/2 - cdomSize[0]/2)**2 + (Y - dx/2 - cdomSize[1]/2)**2) >= Rgrid**2) & (((X - dx/2 - cdomSize[0]/2)**2 + (Y - dx/2 - cdomSize[1]/2)**2) <= (Rgrid + np.sqrt(2)*dx)**2)
wireSel[:, :, 0:int(np.round(Z.shape[2] / 2))] = False
wireSel[:, :, (int(np.round(Z.shape[2] / 2)) + 1):] = False

conductor = conductor | wireSel

# Grid circle in the xz-plane
wireSel = ((X - dx/2 - cdomSize[0]/2)**2 + (Z - dx/2 - cdomSize[2]/2)**2 >= Rgrid**2) & ((X - dx/2 - cdomSize[0]/2)**2 + (Z - dx/2 - cdomSize[2]/2)**2 <= (Rgrid + np.sqrt(2)*dx)**2)
wireSel[:, 0:int(np.round(Y.shape[1] / 2)), :] = False
wireSel[:, (int(np.round(Y.shape[1] / 2) + 1)):, :] = False

conductor = conductor | wireSel

# Grid wire in the yz plane
wireSel = ((Y - dx/2 - cdomSize[1]/2)**2 + (Z - dx/2 - cdomSize[2]/2)**2 >= Rgrid**2) & ((Y - dx/2 - cdomSize[1]/2)**2 + (Z - dx/2 - cdomSize[2]/2)**2 <= (Rgrid + np.sqrt(2)*dx)**2)
closestZPt = np.min(Z[wireSel])
wireSel[0:int(np.round(X.shape[0] / 2)), :, :] = False
wireSel[(int(np.round(X.shape[0] / 2) + 1)):, :, :] = False

conductor = conductor | wireSel

del wireSel # nasty amount of memory


# *** Wire conecting grid to vacuum chamber ***
conductor[int(np.round(X.shape[0] / 2)), int(np.round(Y.shape[0] / 2)), 0:int(closestZPt/dx + 1)] = True

# *** Building the conductor field selection grid ***

conductorFieldSel[:, :, :, 0] = conductor
conductorFieldSel[:, 0:-1, :, 0] = conductorFieldSel[:, 0:-1, :, 0] | conductor[:, 1:, :] # y
conductorFieldSel[:, :, 0:-1, 0] = conductorFieldSel[:, :, 0:-1, 0] | conductor[:, :, 1:] # z
conductorFieldSel[:, 0:-1, 0:-1, 0] = conductorFieldSel[:, 0:-1, 0:-1, 0] | conductor[:, 1:, 1:] # y, z

conductorFieldSel[:, :, :, 1] = conductor
conductorFieldSel[0:-1, :, :, 1] = conductorFieldSel[0:-1, :, :, 1] | conductor[1:, :, :] # x
conductorFieldSel[:, :, 0:-1, 1] = conductorFieldSel[:, :, 0:-1, 1] | conductor[:, :, 1:] # z
conductorFieldSel[0:-1, :, 0:-1, 1] = conductorFieldSel[0:-1, :, 0:-1, 1] | conductor[1:, :, 1:] # x, z

conductorFieldSel[:, :, :, 2] = conductor
conductorFieldSel[0:-1, :, :, 2] = conductorFieldSel[0:-1, :, :, 2] | conductor[1:, :, :] # x
conductorFieldSel[:, 0:-1, :, 2] = conductorFieldSel[:, 0:-1, :, 2] | conductor[:, 1:, :] # y
conductorFieldSel[0:-1, 0:-1, :, 2] = conductorFieldSel[0:-1, 0:-1, :, 2] | conductor[1:, 1:, :] # x, y


# ========================= Remaining nondimensionalization =========================

# Volume of empty space within the reactor
V_react = np.sum(~conductor) * dx_dim**3

# Number of electrons in the plasma
N_e = D_e * V_react

# Particle weight
W = N_e / n_p
m0 = W * m_e
q0 = c*np.sqrt(x0*m0)
#p0 = m0*c

# Electric field
E0 = q0/x0**2
B0 = E0

# Voltage (Gaussian)
V_g = V_dim / 299.792458

# Electric field to establish potential difference between vacuum chamber and cathode
E_cath = V_g / (x0 * E0)

# Characteristic kinetic energy
K0 = m0*c**2

# *** Particles ***

# Particle charges
q = np.ones(n_sp)
q[0] = W*q_e/q0 # Nuclei charge
q[1] = -q[0] # Electron charge

# Particle masses
m = np.ones(n_sp)
m[0] = W*m_p/m0 #Nuclei mass
m[1] = W*m_e/m0 #Electron mass

# Particle positions
xp = np.zeros([3, n_sp, n_p], dtype=np.float32)   # Position [Position component, Species index, Particle index]

# Particle momenta
pp = np.zeros(xp.shape, dtype=np.float32)   # Momentum [Momentum component, Species index, Particle index]

# Particle active
pActive = np.zeros([n_sp, n_p], dtype=np.bool_) # Active [Species index, Particle index]

# ========================= Metadata (used for plotting) ================================

# x0 t0 v0 p0 m0 q0 K0 E0 B0 J0
modelMeta = np.array([x0, t0, m0, q0, K0, E0, B0, dt, dx, outputInterval], dtype=np.float64)

np.save('META.npy', modelMeta)

# ========================= Model execution =========================

# Current timestep
currentTime = 0

# Incrementor for successive model outputs
fileCounter = 0

# Whether this is the first iteration
first = True

# Array to store the charge traversing between the cathode and vacuum chamber over time
q_cath = np.zeros(round(T/outputInterval), dtype=np.float32)

# *** Execute the model until we have simulated the desired amount of time ***
print('Begin model execution')
print('(the first timestep takes forever due to compilation)')
for i in range(round(T/outputInterval)):

    # To measure performance and provide an ETA for completion, record and print the time needed to execute EvolveModel
    prevBlockTime = time.time()
    E, B, J, xp, pp, q_cath[i], pActive, currentTime = EvolveModel(E, B, J, conductorFieldSel, conductor, xp, pp, pActive, q, m, dx, dt, conductionFactor, E_cath, currentTime, tsteps=outputInterval)
    print('Finished block, took', time.time() - prevBlockTime, 'seconds.')
    print('\n',currentTime, '/', T)

    # Record the time since the model began, *after* compilation has occured
    if first:
        beginTime = time.time()
        first = False
    
    # Report the memory usage and ETA to completion if this is not the first time
    else:
        approxRate = (currentTime - outputInterval) / (time.time() - beginTime)
        print('Estimated time remaining:', (T - currentTime) / approxRate, 's')
        print('Memory usage:', process.memory_info().rss/1024**2)

    # Compute the number densities of the electrons and nuclei, as well as the 
    # kinetic energy distribution of the nuclei
    Ne = NumberDensity(xp, pActive, 1)

    # Write data to files and iterate the file counter
    np.save('Nn_' + str(fileCounter), NumberDensity(xp, pActive, 0))
    np.save('Ne_' + str(fileCounter), Ne)
    np.save('K_' + str(fileCounter), KineticEnergy(xp, pp, pActive, 0))
    np.save('q_cath', q_cath)
    fileCounter = fileCounter + 1

    # Plot the electron number density to show that things are happening
    plt.cla()
    plt.pcolormesh(X[:, :, round(z.size*0.5)], Y[:, :, round(z.size*0.5)], Ne[:, :, round(z.size*0.5)], shading='auto')
    plt.draw()
    plt.pause(0.1)
    
# Inform the user that the program has finished
print('Model execution complete, program end.')