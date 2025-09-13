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

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def generate_main_cosmology_plots():
    """Generate the main 2x2 cosmology plot."""
    print("Generating main cosmology plots...")

    # Create a cosmology instance with Planck 2018 parameters
    cosmo = jaxace.W0WaCDMCosmology(
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
    ax.plot(z, r_comoving, label='Comoving distance $r(z)$', linewidth=2.5)
    ax.plot(z, dA, label='Angular diameter distance $d_A(z)$', linewidth=2.5)
    ax.plot(z, dL, label='Luminosity distance $d_L(z)$', linewidth=2.5)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('Distance [Mpc]', fontsize=12)
    ax.set_title('Cosmological Distances', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Plot 2: Hubble parameter
    ax = axes[0, 1]
    ax.plot(z, H_z, color='darkblue', linewidth=2.5)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$H(z)$ [km/s/Mpc]', fontsize=12)
    ax.set_title('Hubble Parameter Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Growth factor
    ax = axes[1, 0]
    ax.plot(z, D, color='darkgreen', linewidth=2.5)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$D(z)$', fontsize=12)
    ax.set_title('Linear Growth Factor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Plot 4: Growth rate
    ax = axes[1, 1]
    ax.plot(z, f, color='darkred', linewidth=2.5)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$f(z) = \\mathrm{d}\\ln D / \\mathrm{d}\\ln a$', fontsize=12)
    ax.set_title('Growth Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])

    plt.suptitle('ΛCDM Cosmology (Planck 2018)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/images/cosmology_main.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved cosmology_main.png")

def generate_omega_m_plot():
    """Generate Omega_m evolution plot."""
    print("Generating Omega_m evolution plot...")

    cosmo = jaxace.W0WaCDMCosmology(
        ln10As=3.044, ns=0.9649, h=0.6736,
        omega_b=0.02237, omega_c=0.1200,
        m_nu=0.06, w0=-1.0, wa=0.0
    )

    z = jnp.linspace(0.01, 3.0, 100)
    a = 1.0 / (1.0 + z)
    Omega_m = jaxace.Ωm_a(a, cosmo.omega_b + cosmo.omega_c, cosmo.h,
                          mν=cosmo.m_nu, w0=cosmo.w0, wa=cosmo.wa)

    plt.figure(figsize=(8, 6))
    plt.plot(z, Omega_m, linewidth=3, color='purple')
    plt.xlabel('Redshift $z$', fontsize=12)
    plt.ylabel('$\\Omega_{\\mathrm{m}}(z)$', fontsize=12)
    plt.title('Matter Density Parameter Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
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
        'ΛCDM': {'w0': -1.0, 'wa': 0.0, 'color': 'blue', 'ls': '-'},
        'wCDM': {'w0': -0.9, 'wa': 0.0, 'color': 'green', 'ls': '--'},
        'w0waCDM': {'w0': -0.95, 'wa': 0.3, 'color': 'red', 'ls': ':'},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, params in cosmologies.items():
        cosmo_test = jaxace.W0WaCDMCosmology(
            ln10As=3.044, ns=0.9649, h=0.6736,
            omega_b=0.02237, omega_c=0.1200,
            m_nu=0.06, w0=params['w0'], wa=params['wa']
        )

        D_test = cosmo_test.D_z(z)
        f_test = cosmo_test.f_z(z)

        label = f"{name}: $w_0$={params['w0']}, $w_a$={params['wa']}"
        ax1.plot(z, D_test, label=label, linewidth=2.5,
                color=params['color'], linestyle=params['ls'])
        ax2.plot(z, f_test, label=label, linewidth=2.5,
                color=params['color'], linestyle=params['ls'])

    ax1.set_xlabel('Redshift $z$', fontsize=12)
    ax1.set_ylabel('$D(z)$', fontsize=12)
    ax1.set_title('Growth Factor Comparison', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2.set_xlabel('Redshift $z$', fontsize=12)
    ax2.set_ylabel('$f(z)$', fontsize=12)
    ax2.set_title('Growth Rate Comparison', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])

    plt.suptitle('Dark Energy Model Comparison', fontsize=16, fontweight='bold', y=1.05)
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
        cosmo = jaxace.W0WaCDMCosmology(
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
    param_names = ['$\\Omega_c$', '$\\Omega_b$', '$h$', '$m_\\nu$ [eV]', '$w_0$', '$w_a$']
    param_labels = ['omega_c', 'omega_b', 'h', 'm_nu', 'w0', 'wa']

    for i, (ax, name, label) in enumerate(zip(axes.flat, param_names, param_labels)):
        ax.plot(z, jacobian[:, i], linewidth=2.5, color=plt.cm.viridis(i/5))
        ax.set_xlabel('Redshift $z$', fontsize=11)
        ax.set_ylabel(f'$\\partial D/\\partial${name}', fontsize=11)
        ax.set_title(f'Growth Factor Sensitivity to {name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Add text with max sensitivity
        max_idx = jnp.argmax(jnp.abs(jacobian[:, i]))
        max_z = z[max_idx]
        max_val = jacobian[max_idx, i]
        ax.text(0.95, 0.95, f'Max at z={max_z:.2f}\\n∂D/∂{label}={max_val:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.suptitle('Jacobian of Growth Factor D(z) w.r.t. Cosmological Parameters',
                 fontsize=16, fontweight='bold')
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
        cosmo = jaxace.W0WaCDMCosmology(
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
    param_names = ['$\\Omega_c$', '$\\Omega_b$', '$h$', '$m_\\nu$ [eV]', '$w_0$', '$w_a$']

    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        ax.plot(z, jacobian_D[:, i], label='$\\partial D/\\partial$' + name,
                linewidth=2.5, color='blue')
        ax.plot(z, jacobian_f[:, i], label='$\\partial f/\\partial$' + name,
                linewidth=2.5, color='red', linestyle='--')
        ax.set_xlabel('Redshift $z$', fontsize=11)
        ax.set_ylabel('Derivative', fontsize=11)
        ax.set_title(f'Sensitivity to {name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    plt.suptitle('Comparison of D(z) and f(z) Sensitivities',
                 fontsize=16, fontweight='bold')
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