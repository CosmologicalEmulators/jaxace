#!/usr/bin/env python
"""
Example script demonstrating jaxace cosmology calculations and plotting.

This script shows how to:
1. Create a cosmology instance
2. Compute comoving distance, growth factor, and growth rate
3. Visualize the results
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxace

def main():
    # Create a cosmology instance with Planck 2018 parameters
    cosmo = jaxace.W0WaCDMCosmology(
        ln10As=3.044,      # ln(10^10 A_s)
        ns=0.9649,         # Scalar spectral index
        h=0.6736,          # Hubble parameter
        omega_b=0.02237,   # Baryon density
        omega_c=0.1200,    # CDM density
        m_nu=0.06,         # Sum of neutrino masses in eV
        w0=-1.0,           # Dark energy equation of state
        wa=0.0             # Dark energy evolution parameter
    )

    # Create redshift array from 0 to 3
    z = jnp.linspace(0.01, 3.0, 100)

    # Compute cosmological quantities
    print("Computing cosmological distances...")
    r_comoving = jaxace.r_z_from_cosmo(z, cosmo)
    dA = jaxace.dA_z_from_cosmo(z, cosmo)
    dL = jaxace.dL_z_from_cosmo(z, cosmo)

    print("Computing growth functions...")
    D = jaxace.D_z_from_cosmo(z, cosmo)
    f = jaxace.f_z_from_cosmo(z, cosmo)

    print("Computing Hubble parameter...")
    E_z = jaxace.E_z(z, cosmo.omega_b + cosmo.omega_c, cosmo.h,
                     mν=cosmo.m_nu, w0=cosmo.w0, wa=cosmo.wa)
    H_z = 100 * cosmo.h * E_z  # H(z) in km/s/Mpc

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Distance measures
    ax = axes[0, 0]
    ax.plot(z, r_comoving, label='Comoving distance $r(z)$', linewidth=2)
    ax.plot(z, dA, label='Angular diameter distance $d_A(z)$', linewidth=2)
    ax.plot(z, dL, label='Luminosity distance $d_L(z)$', linewidth=2)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Distance [Mpc]')
    ax.set_title('Cosmological Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Hubble parameter
    ax = axes[0, 1]
    ax.plot(z, H_z, color='darkblue', linewidth=2)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$H(z)$ [km/s/Mpc]')
    ax.set_title('Hubble Parameter Evolution')
    ax.grid(True, alpha=0.3)

    # Plot 3: Growth factor
    ax = axes[1, 0]
    ax.plot(z, D, color='darkgreen', linewidth=2)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$D(z)$')
    ax.set_title('Linear Growth Factor')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Convention: D grows as z decreases

    # Plot 4: Growth rate
    ax = axes[1, 1]
    ax.plot(z, f, color='darkred', linewidth=2)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$f(z) = d\\ln D / d\\ln a$')
    ax.set_title('Growth Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.suptitle('ΛCDM Cosmology (Planck 2018)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('cosmology_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print some values at specific redshifts
    print("\nCosmological quantities at specific redshifts:")
    print("-" * 60)
    print(f"{'z':<8} {'r(z) [Mpc]':<15} {'D(z)':<10} {'f(z)':<10}")
    print("-" * 60)

    for zi in [0.1, 0.5, 1.0, 2.0, 3.0]:
        idx = jnp.argmin(jnp.abs(z - zi))
        print(f"{zi:<8.1f} {r_comoving[idx]:<15.1f} {D[idx]:<10.4f} {f[idx]:<10.4f}")

if __name__ == "__main__":
    main()