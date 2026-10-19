# Equations, verification, and reproducibility

## 1. Published groundwater benchmark

We independently implement the 1D finite-difference benchmark in Alecsa et al.,
[arXiv:1911.10774v2](https://arxiv.org/html/1911.10774v2), Section 3.1, Eq. (9),
and Appendix A, Eqs. (26)–(29). The related journal article is
[Numerical benchmark study for flow in highly heterogeneous aquifers](https://doi.org/10.1016/j.advwatres.2020.103558).
We reproduce the formulation and convergence behaviour for two parameter choices;
we do not reproduce the paper's exact coefficient realization, full parameter study,
or tabulated error values. No code or parameter files were copied from the authors' repository.

The source-driven, saturated steady flow equation is

$$-\frac{d}{dx}\left(K\frac{dh}{dx}\right)=s,\qquad q_D=-K\frac{dh}{dx}.$$

Here $x$ is in m, $h$ in m, $K$ in m/s, $q_D$ in m/s, and $s$ in s⁻¹.
Positive flux is in the positive $x$ direction, so $dq_D/dx=s$. Positive source denotes
injection into the aquifer. The manufactured source may have either sign.

For domain length $L=200$ m and correlation length $\lambda=1$ m,

$$h_*(x)=3\,\mathrm m+(1\,\mathrm m)\sin(x/\lambda).$$

Both boundary heads are set from this function. The dimensionless sine argument
makes explicit the normalization used in the published formulation. Write $A=1$ m;
the independent analytical source is

$$s(x)=\frac{A K(x)}{\lambda^2}\sin(x/\lambda)
       -\frac{A K'(x)}{\lambda}\cos(x/\lambda).$$

Source values are not manufactured by applying the discrete operator to $h_*$:
doing that would hide discretization errors.

### Conductivity and our coefficient realization

$$K(x)=\bar K\exp\left[-\frac{\sigma_Y^2}{2}
 +\sigma_Y\sqrt{\frac{2}{N}}\sum_{j=1}^{N}
 \cos\left(\phi_j+2\pi[k_{j1}x/\lambda+k_{j2}]\right)\right].$$

We use the $y/\lambda=1$ slice of the Gaussian-correlated spectral field, with
$N=100$, $\sigma_Y^2\in\{0.1,1\}$, and nominal ensemble mean
$\bar K=15$ m/day, converted to $15/86400$ m/s. A finite realization need not have
that exact spatial mean. The finite cosine sum approximates the intended Gaussian
log-conductivity statistics; no empirical distribution fit is claimed.

Each dimensionless wavenumber component is independently sampled from
$N(0,1/(2\pi^2))$, and phases from $U(0,2\pi)$. This yields the intended
normalized correlation $C(r)/\sigma_Y^2=\exp[-(r/\lambda)^2]$ in the ensemble.
NumPy's `default_rng(42)` generates the wavenumber array first, then the phase array.
All 100 rows are committed as CSV with 18-digit scientific precision. Reproduction
loads that file rather than resampling; the metrics record its SHA-256 digest.

### Discretization and errors

A uniform grid has $n$ intervals and $n+1$ nodes. Conductivity is evaluated analytically
at faces. For each interior node,

$$-K_{i-1/2}h_{i-1}+(K_{i-1/2}+K_{i+1/2})h_i-K_{i+1/2}h_{i+1}
 =s_i\Delta x^2.$$

The tridiagonal system is solved using SciPy's banded solver, with linear memory
and work. Field evaluation is chunked to bound intermediate-array memory.

We report discrete $L^2=\sqrt{\Delta x\sum_i(h_i-h_{*,i})^2}$, its ratio to
$\sqrt{\Delta x\sum_i h_{*,i}^2}$, maximum head error, and
$\mathrm{EOC}=\log_2(E_{\Delta x}/E_{\Delta x/2})$. Error sums use interior nodes.
The dimensional $L^2$ norm has units m·√m; the relative norm and EOC are dimensionless.

Refinements use 1,000, 2,000, 4,000, 8,000, and 16,000 intervals. Acceptance requires
relative head error below $10^{-3}$ on the finest grid and EOC in [1.8, 2.2] on the last
two refinement pairs. The figure shows a 20 m window for readability; errors use the
full 200 m domain, and complete profiles are exported.

Mass balance is checked over the interior nodal control volumes:
$q_{n-1/2}-q_{1/2}=\Delta x\sum_i s_i$. This is a discrete balance using midpoint
sources, not a claim of exact continuous source integration. We normalize its mismatch
by the larger of maximum face-flux magnitude and integrated absolute discrete source.
The scaled linear residual uses $\|Ah-r\|_\infty/(\|A\|_\infty\|h\|_\infty+\|r\|_\infty)$
after eliminating boundary values. Both diagnostics are exported for every grid.

## 2. Cubic-law verification

For horizontal flow between ideal smooth plates separated by **full hydraulic aperture**
$b$, with incompressible Newtonian fluid and negligible inertia,

$$q=-\frac{b^3}{12\mu}\frac{dp}{dx},\qquad
 q_{\rm exact}=\frac{b^3(p_{\rm in}-p_{\rm out})}{12\mu L}.$$

Here $q$ is discharge **per unit fracture width**, in m²/s, not total discharge in m³/s.
For width $W$, $Q=Wq$. Pressure is in Pa and dynamic viscosity $\mu$ in Pa·s.
The corresponding intrinsic permeability is $b^2/12$ (m²), hydraulic conductivity
$\rho g b^2/(12\mu)$ (m/s), and hydraulic transmissivity $\rho g b^3/(12\mu)$ (m²/s).
These are distinct from the pressure mobility $b^3/(12\mu)$ used in the solver.

We use full aperture consistently: some references, including Witherspoon et al., use
a half-aperture symbol and write the full separation as $2b$. Substituting that symbol
without conversion would introduce a factor-of-eight error.

The one-command experiment solves a constant-aperture case at 25 μm, comparing
pressure with the exact linear profile and every face flux with the analytical discharge.
A separate 5–100 μm sweep at fixed $\Delta p=1000$ Pa and $L=100$ m checks the exponent
3 by a log–log fit. That sweep is a constitutive identity check; numerical solver
verification comes from the constant-aperture solve and nonlinear reference comparison.

## 3. Effective-stress-dependent aperture

Compression is positive and effective normal stress is

$$\sigma'_n=\sigma_n-\alpha p,\qquad
 b=b_r+\frac{b_0-b_r}{1+\sigma'_n/\sigma_c}.$$

Total normal stress and pore pressure must use the same pressure datum.
With $b_0=100$ μm, $b_r=5$ μm, $\sigma_c=5$ MPa and $\alpha=1$, the aperture is
$b_0$ at zero effective compression and tends to $b_r$ at high compression.
Increasing pore pressure at fixed normal stress increases aperture.
Tensile effective stress is rejected rather than extrapolated or silently clipped.

This is an **illustrative hydraulic-aperture proxy** inspired by hyperbolic normal
joint closure. In terms of the mechanical closure form, maximum closure is $b_0-b_r$
and $\sigma_c=k_{n0}(b_0-b_r)$. Equating that closure to a hydraulic-aperture change
is an additional assumption, not a universal mechanical-to-hydraulic aperture relation
or a full Barton–Bandis model. No roughness, contact-area, loading-history, shear-dilation,
or experimental calibration is included.

The first experiment sweeps effective compression from 0–40 MPa at fixed pressure
gradient. The second uses $L=100$ m, $\sigma_n=20$ MPa, $p_{\rm in}=12$ MPa,
$p_{\rm out}=2$ MPa, $\mu=0.001$ Pa·s, and $\rho=1000$ kg/m³.

### Nonlinear solver and independent reference

We solve $dq/dx=0$ with face aperture evaluated at $(p_i+p_{i+1})/2$.
Picard iterations freeze that mobility, solve the conservative tridiagonal system,
and relax the pressure update with factor 0.5. At every iteration, face fluxes are
recomputed at the **updated** pressure; conserving a frozen linear iterate alone
would not establish nonlinear convergence.

Convergence requires a maximum pressure update below $10^{-11}$ of the larger boundary
pressure/drop scale and relative face-flux spread below $10^{-8}$. The solver raises
an error after 300 iterations if these conditions are unmet. Equal boundary pressures
return the exact zero-flow equilibrium. The default grid has 400 intervals.

A reference follows independently by integrating the constitutive relation in pressure:

$$q_{\rm ref}=\frac{1}{12\mu L}
 \int_{p_{\rm out}}^{p_{\rm in}} b(\sigma_n-\alpha p)^3\,dp.$$

SciPy adaptive quadrature evaluates this integral without the spatial finite-difference
solver. Acceptance requires relative discharge error below $10^{-4}$. Reversing pressure
boundaries must reverse discharge and mirror the pressure profile. Nonlinear refinement
tests additionally check approximately second-order discharge error.

For comparison, the frozen-aperture case uses $b$ evaluated at mean boundary pressure.
Its pressure profile is linear. The approximately 13.3% difference in discharge is
specific to these illustrative parameters, not a universal enhancement from coupling.

We report $\mathrm{Re}=2\rho|q|/\mu$, using hydraulic diameter $2b$ and mean velocity
$q/b$. The chosen $\mathrm{Re}<1$ criterion is a conservative inertia screen, not a
universal transition threshold or validation of natural-fracture roughness assumptions.
Tests include an out-of-screen case. The aperture and stress sweeps use the smaller
1 kPa pressure drop; their maximum Reynolds number is approximately 0.00167.

## 4. Reproducibility and interpretation

`reproduce` writes CSVs, JSON metrics, and PNG/SVG figures. A failed acceptance gate
returns exit code 1; invalid input, failed iteration, or filesystem error returns 2.
Core solver calls are also available as Python functions in the groundwater and fracture
modules. Runtime versions and the coefficient hash are recorded in the metrics.
`requirements-repro.txt` pins the runtime environment used for committed results;
minor floating-point and rendering differences across platforms are expected.

This project verifies an implementation against analytical/manufactured references.
It does not validate constitutive parameters or predictions against laboratory or field
measurements. The groundwater and fracture experiments are separate problems; the
groundwater solution is not used as a coupled fracture-matrix boundary condition.

The fracture coupling is local pressure → effective stress → aperture → mobility at
prescribed total stress. There is no mechanical equilibrium PDE, deformation-induced
storage, transient injection history, transport, geochemistry, temperature evolution,
fracture network, propagation, or shear. Parameters are synthetic and not representative
of a named site. These limits should accompany any application or interview discussion.

References: [BibTeX bibliography](references.bib).
