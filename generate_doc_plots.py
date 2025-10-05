#!/usr/bin/env python
"""
Generate plots for the documentation.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import jaxace

# Set matplotlib style to match notebook
# Try to use LaTeX if available, but don't fail if not
try:
    import subprocess
    subprocess.check_output(['latex', '--version'])
    plt.rcParams['text.usetex'] = True
    print("LaTeX detected, using LaTeX rendering for plots")
except (FileNotFoundError, subprocess.CalledProcessError):
    plt.rcParams['text.usetex'] = False
    print("LaTeX not available, using matplotlib's default math rendering")

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def generate_main_cosmology_plots():
    """Generate the main 2x2 cosmology plot."""
    print("Generating main cosmology plots...")

    # Create a cosmology instance with Planck 2018 parameters
    cosmo = jaxace.w0waCDMCosmology(
        ln10As=3.044,
        ns=0.9649,
        h=0.6736,
        omega_b=0.02237,
        omega_c=0.1200,
        m_nu=0.06,
        w0=-1.0,
        wa=0.0
    )

    # Create redshift array
    z = jnp.linspace(0.01, 3.0, 100)

    # Compute quantities
    r_comoving = cosmo.r_z(z)
    dA = cosmo.dA_z(z)
    dL = cosmo.dL_z(z)
    D = cosmo.D_z(z)
    f = cosmo.f_z(z)

    E_z = jaxace.E_z(z, cosmo.omega_b + cosmo.omega_c, cosmo.h,
                     mν=cosmo.m_nu, w0=cosmo.w0, wa=cosmo.wa)
    H_z = 100 * cosmo.h * E_z

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Distance measures
    ax = axes[0, 0]
    ax.plot(z, r_comoving, label=r'$r(z)$', linewidth=2)
    ax.plot(z, dA, label=r'$d_A(z)$', linewidth=2)
    ax.plot(z, dL, label=r'$d_L(z)$', linewidth=2)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'Distance [Mpc]')
    ax.legend()

    # Plot 2: Hubble parameter
    ax = axes[0, 1]
    ax.plot(z, H_z, linewidth=2)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]')

    # Plot 3: Growth factor
    ax = axes[1, 0]
    ax.plot(z, D, linewidth=2)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$D(z)$')
    ax.invert_xaxis()

    # Plot 4: Growth rate
    ax = axes[1, 1]
    ax.plot(z, f, linewidth=2)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$f(z) = \mathrm{d}\ln D / \mathrm{d}\ln a$')
    ax.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig('docs/images/cosmology_main.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved cosmology_main.png")

def generate_omega_m_plot():
    """Generate Omega_m evolution plot."""
    print("Generating Omega_m evolution plot...")

    cosmo = jaxace.w0waCDMCosmology(
        ln10As=3.044, ns=0.9649, h=0.6736,
        omega_b=0.02237, omega_c=0.1200,
        m_nu=0.06, w0=-1.0, wa=0.0
    )

    z = jnp.linspace(0.01, 3.0, 100)
    a = 1.0 / (1.0 + z)
    Omega_m = jaxace.Ωm_a(a, cosmo.omega_b + cosmo.omega_c, cosmo.h,
                          mν=cosmo.m_nu, w0=cosmo.w0, wa=cosmo.wa)

    plt.figure(figsize=(8, 6))
    plt.plot(z, Omega_m, linewidth=2)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Omega_{\mathrm{m}}(z)$')
    plt.xlim(0, 3)
    plt.ylim(0.2, 1.0)
    plt.tight_layout()
    plt.savefig('docs/images/omega_m_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved omega_m_evolution.png")

def generate_cosmology_comparison():
    """Generate comparison of different cosmologies."""
    print("Generating cosmology comparison plots...")

    z = jnp.linspace(0.01, 3.0, 100)

    cosmologies = {
        r'$\Lambda$CDM': {'w0': -1.0, 'wa': 0.0, 'color': 'blue', 'ls': '-'},
        'wCDM': {'w0': -0.9, 'wa': 0.0, 'color': 'green', 'ls': '--'},
        r'$w_0w_a$CDM': {'w0': -0.95, 'wa': 0.3, 'color': 'red', 'ls': ':'},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, params in cosmologies.items():
        cosmo_test = jaxace.w0waCDMCosmology(
            ln10As=3.044, ns=0.9649, h=0.6736,
            omega_b=0.02237, omega_c=0.1200,
            m_nu=0.06, w0=params['w0'], wa=params['wa']
        )

        D_test = cosmo_test.D_z(z)
        f_test = cosmo_test.f_z(z)

        label = f"{name}: " + r"$w_0$=" + f"{params['w0']}, " + r"$w_a$=" + f"{params['wa']}"
        ax1.plot(z, D_test, label=label, linewidth=2, linestyle=params['ls'])
        ax2.plot(z, f_test, label=label, linewidth=2, linestyle=params['ls'])

    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$D(z)$')
    ax1.legend()
    ax1.invert_xaxis()

    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$f(z)$')
    ax2.legend()
    ax2.set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig('docs/images/cosmology_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved cosmology_comparison.png")

def generate_growth_jacobian():
    """Generate Jacobian plots for growth functions."""
    print("Generating growth Jacobian plots...")

    # Define a function that computes growth factor for given parameters
    def growth_factor_function(params, z):
        """Compute growth factor D(z) for given cosmological parameters."""
        omega_c, omega_b, h, m_nu, w0, wa = params
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=3.044,  # Keep fixed for this example
            ns=0.9649,      # Keep fixed for this example
            h=h,
            omega_b=omega_b,
            omega_c=omega_c,
            m_nu=m_nu,
            w0=w0,
            wa=wa
        )
        return cosmo.D_z(z)

    # Define fiducial parameters (Planck 2018)
    fiducial_params = jnp.array([
        0.1200,   # omega_c
        0.02237,  # omega_b
        0.6736,   # h
        0.06,     # m_nu (eV)
        -1.0,     # w0
        0.0       # wa
    ])

    # Redshift array
    z = jnp.linspace(0.01, 3.0, 50)

    # Compute Jacobian matrix: dD/dθ for all parameters at all redshifts
    jacobian_fn = jax.jacobian(growth_factor_function, argnums=0)
    jacobian = jacobian_fn(fiducial_params, z)

    # Plot the derivatives
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    param_names = [r'$\Omega_\mathrm{c}$', r'$\Omega_\mathrm{b}$', r'$h$', r'$m_\nu$ [eV]', r'$w_0$', r'$w_a$']
    param_labels = ['omega_c', 'omega_b', 'h', 'm_nu', 'w0', 'wa']

    for i, (ax, name, label) in enumerate(zip(axes.flat, param_names, param_labels)):
        ax.plot(z, jacobian[:, i], linewidth=2)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\partial D/\partial$' + name)

    plt.tight_layout()
    plt.savefig('docs/images/growth_jacobian.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved growth_jacobian.png")

def generate_growth_jacobian_comparison():
    """Generate comparison of Jacobians for D and f."""
    print("Generating growth Jacobian comparison plots...")

    # Define a function that returns both D and f
    def growth_functions(params, z):
        """Compute both D(z) and f(z) for given parameters."""
        omega_c, omega_b, h, m_nu, w0, wa = params
        cosmo = jaxace.w0waCDMCosmology(
            ln10As=3.044, ns=0.9649,
            h=h, omega_b=omega_b, omega_c=omega_c,
            m_nu=m_nu, w0=w0, wa=wa
        )
        D, f = cosmo.D_f_z(z)
        return jnp.stack([D, f])  # Stack to get shape (2, n_z)

    # Define fiducial parameters
    fiducial_params = jnp.array([
        0.1200,   # omega_c
        0.02237,  # omega_b
        0.6736,   # h
        0.06,     # m_nu (eV)
        -1.0,     # w0
        0.0       # wa
    ])

    z = jnp.linspace(0.01, 3.0, 50)

    # Compute Jacobian for both quantities
    jacobian_both_fn = jax.jacobian(growth_functions, argnums=0)
    jacobian_both = jacobian_both_fn(fiducial_params, z)

    # jacobian_both shape is (2, n_z, n_params)
    jacobian_D = jacobian_both[0]  # Shape: (n_z, n_params)
    jacobian_f = jacobian_both[1]  # Shape: (n_z, n_params)

    # Plot comparison of sensitivities for D and f
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    param_names = [r'$\Omega_\mathrm{c}$', r'$\Omega_\mathrm{b}$', r'$h$', r'$m_\nu$ [eV]', r'$w_0$', r'$w_a$']

    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        ax.plot(z, jacobian_D[:, i], label=r'$\partial D/\partial$' + name,
                linewidth=2)
        ax.plot(z, jacobian_f[:, i], label=r'$\partial f/\partial$' + name,
                linewidth=2, linestyle='--')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'Derivative')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('docs/images/growth_jacobian_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved growth_jacobian_comparison.png")

def main():
    print("Generating documentation plots...")
    print("-" * 40)

    generate_main_cosmology_plots()
    generate_omega_m_plot()
    generate_cosmology_comparison()
    generate_growth_jacobian()
    generate_growth_jacobian_comparison()

    print("-" * 40)
    print("✓ All plots generated successfully!")
    print("\nPlots saved to docs/images/")

if __name__ == "__main__":
    main()
