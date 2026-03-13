.. _theory-numerics:

Numerical solution
==================

In this section, we first sum up the problem in matrix notation. Then we describe the solution strategy, the equations of the Chasing method and how it is implemented in PyFuga. In particular, it is described how boundary conditions are formulated at another boundary. In the last section, we describe why and how the Chasing method is modified.

Problem statement
-----------------

We now have two linearized sets of differential equations for the first-order perturbation representing the effect of a wind turbine on the logarithmic wind profile. One set is valid for the vertical coordinate :math:`s\geq s_{tr}` and the other set is valid for :math:`t<s_{tr}`.

Stating the problem as a matrix equation and using :math:`z` as a general vertical coordinate, representing either :math:`s` or :math:`t`, the equations become

.. math::
   :label: eq:Problem

   \frac{\partial X_i(z)}{\partial z} = A_{ij}X_j(z)+F_i

for each vertical level :math:`z_n`, where :math:`\mathbf{X}^T(z) = [u(z),u'(z),v(z),v'(z),w(z),p(z)]`, and :math:`F_i` is non-zero for :math:`i=2,4\,\,\text{or}\,\,6` - representing longitudinal, transverse and vertical forcing, respectively. The three orthogonal forcing directions represent three separate solutions, which can in the end be scaled and superimposed to give the response of the full three-dimensional forcing. Depending on which vertical variable is used, :math:`F_i =\phi_n(z)\hat{\mathbf{e}}_i/kK^*` in the case of :math:`s` and :math:`F_i = kK^*\phi_n(z)\hat{\mathbf{e}}_i/\kappa^2` in the case of :math:`t`.

Note that the equation is dimensionless as defined in equations :eq:`eq:Eom Fourier transformed nondim s` and :eq:`eq:Eom Fourier transformed nondim t` of the previous section, but we are omitting the tilde's on the variables for brevity. See appendix C.4 for more detail.

In addition, we have the six boundary conditions distributed over the vertical boundaries, :math:`u(z_0)=v(z_0)=w(z_0)=u(z_{ih})=v(z_{ih})=w(z_{ih}) = 0`.

What we have is a set of linear differential equations with boundary conditions both at the roughness height (:math:`z=z_0`) and at the boundary layer (inversion) height (:math:`z=z_{ih}`). It may sound as an easy problem, but it is not. The precision required for small values of :math:`kz_0` (relevant for offshore wind farms with long wakes) discards shooting methods usually used for Boundary Value Problems (BVP). To solve this, Fuga uses a *modified* version of the Chasing method, first proposed by :cite:`Berezin1965`.

The Chasing method transforms boundary conditions at one boundary to equivalent boundary conditions at the other boundary. With all boundary conditions at the same boundary, we have an Initial Value Problem (IVP) which is integrated using the second-order Runge-Kutta method.

The final goal is to determine :math:`\mathbf{X}(z)` for any height and any forcing.

The Chasing method
------------------

In the following, the Chasing method is described, which is the ideal method that is later modified.

We start by rewriting the boundary conditions as inner products using the vector :math:`\mathbf{X}(z)`,

.. math::
   :label: eq:innerprod

   \hat{\mathbf{e}}_i^\dagger\mathbf{X}(z_0)=\hat{\mathbf{e}}_i^\dagger\mathbf{X}(z_{ih})=0,

where :math:`\hat{\mathbf{e}}_i\in[\hat{\mathbf{e}}_1,\hat{\mathbf{e}}_3,\hat{\mathbf{e}}_5]` is the unit vector in the :math:`i^{\mathrm{th}}` dimension and the ":math:`^\dagger`" represents the Hermitian conjugate, i.e. the transpose with complex conjugation of every entry.

In order to be able to translate the lower boundary conditions into the upper boundary, we state the boundary conditions as a function of height :math:`z`, i.e.

.. math::
   :label: eq:yxb

   \mathbf{Y}^\dagger(z)\mathbf{X}(z) = \mathbf{b}(z).

The columns of :math:`\mathbf{Y}(z)` are now basis vectors spanning a six-dimensional space, representing independent equations tracking how the ground perturbations evolve upward.

With this notation, we have at the lower boundary

.. math::
   :label: eq:YXb-z_0

   \mathbf{Y}^\dagger(z_0)\mathbf{X}(z_0) =
       \begin{bmatrix}
           1 & 0 & 0 & 0 & 0 & 0 \\
           0 & 0 & 1 & 0 & 0 & 0 \\
           0 & 0 & 0 & 0 & 1 & 0 \\
           0 & 1 & 0 & 0 & 0 & 0 \\
           0 & 0 & 0 & 1 & 0 & 0 \\
           0 & 0 & 0 & 0 & 0 & 1
       \end{bmatrix}
       \begin{bmatrix}
           u(z_0) \\ u'(z_0) \\ v(z_0) \\ v'(z_0) \\ w(z_0) \\ p(z_0)
       \end{bmatrix}
       =
       \begin{bmatrix}
           0 \\ 0 \\ 0 \\ b_{4}(z_0) \\ b_{5}(z_0) \\ b_{6}(z_0)
       \end{bmatrix},

equivalently to eq. :eq:`eq:innerprod` in functional form with the unknown conditions included. Note that there is freedom to define the equation differently, as long as the three boundary conditions at :math:`z=z_0`, corresponding to eq. :eq:`eq:innerprod`, are met. Other relevant considerations is e.g. to construct :math:`\mathbf{Y}(z_0)` so that it is unitary.

If we assume that :math:`\mathbf{Y}(z)` satisfies the differential equation

.. math::
   :label: eq:ytricky

   \frac{d (Y^\dagger)_{ij}}{dz} = -(A^\dagger)_{ik}Y_{kj},

and evaluate the derivative of :eq:`eq:yxb`, we get (omitting specifying the :math:`z`-dependence)

.. math::
   :label: eq:dbdz

   \frac{db_i}{dz} = \frac{d}{dz}\Big( (Y^\dagger)_{ij}X_j\Big) = (Y^\dagger)_{ij} F_j.

Equation :eq:`eq:ytricky` leads to the main equations of the Chasing method, namely the equation to evaluate :math:`\mathbf{b}(z)`,

.. math::
   :label: eq:byf

   \mathbf{b}(z_2)-\mathbf{b}(z_1) = \int_{z_1}^{z_2}  \mathbf{Y}^\dagger(z) \mathbf{F}(z)\,\mathrm{d}z.

along with the differential equation for :math:`\mathbf{Y}(z)`

.. math::
   :label: eq:ytrick

   \frac{d\mathbf{Y}(z)}{dz} = -\mathbf{A}^\dagger(z)\mathbf{Y}(z) = \mathbf{M}(z)\mathbf{Y}(z),

where :math:`\mathbf{M}=-\mathbf{A}^\dagger` and is the matrix calculated in ``preluts_generator``.

Equation :eq:`eq:ytrick` (equivalent to eq. :eq:`eq:ytricky`), together with the conditions at one boundary, is referred to as the *adjoint problem*, since solving it is an IVP whose solution is used to solve the original problem for :math:`\mathbf{X}(z)`. Thus, in order to get there, we start by solving for :math:`\mathbf{Y}(z)`, which we need to find :math:`\mathbf{b}(z)` and ultimately :math:`\mathbf{X}(z)` for each forcing direction. This approach to the problem is the core of the Chasing method.

Computational implementation in PyFuga
--------------------------------------

When solving the problem, the problem is kept as general as possible as long as possible so that the computation is as fast as possible once we get the input on e.g. the forcing for the specific wind turbine.

In this context, the use of look-up tables (LUT's) is central. The LUT's are tables of saved pre-calculations which are extracted later, when the full amount of information is available. This explains the separation between ``preluts_generator`` (pre-LUT) and ``flut`` (Fourier LUT); in ``preluts_generator``, the problem is kept very general with stability conditions and range of wavenumbers as the only user input, so that the computation is faster once the geometry for a specific case is known.

Part 1: preluts_generator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The first part is concerned with finding :math:`\mathbf{Y}(z)` from :eq:`eq:ytrick`.

In order to solve it, we need the information about the stability parameter :math:`\xi`, wavevector :math:`\mathbf{k}` and wavevector angle :math:`\beta` which is used to define the matrix :math:`\mathbf{M}`. Note that the equation is completely independent of :math:`\mathbf{F}(z)`.

From eq. :eq:`eq:YXb-z_0` we see that our initial task is to determine the value of the three unknown values of :math:`\mathbf{b}(z_0)` or :math:`\mathbf{b}(z_{ih})` in order to have an IVP which we can solve by integration. There is a clear distinction between the first and the last three rows of the equation for the lower boundary; the first three rows are fixed by the boundary conditions at the bottom, while the choice of values for the last three rows is arbitrary. This is because :math:`\mathbf{Y}(z)` is six-dimensional while there are only three boundary conditions at the bottom, so we do not yet have sufficient information to formulate the full equation. Hence, there is freedom to define the equation differently, but the one defined in equation :eq:`eq:YXb-z_0` is a good choice, since the matrix is then unitary and orthonormal.

Note that in order to be able to propagate the lower boundary conditions through space, we need to ensure the separation between the constrained and free subspaces so that the solutions for $b_1(z_{ih}), b_2(z_{ih})$ and $b_3(z_{ih})$ do not depend on the arbitrary initial choice. The solution to this is explained in :ref:`modified-chasing-method`.

On top of this, we perform a rotation of :math:`\mathbf{Y}(z_0)` by an angle :math:`\beta` in order to align it with the wavevector (see appendix), resulting in

.. math::
   :label: eq:Y_0_init

   \mathbf{Y}_\beta(z_0) =
   \mathbf{R}_z^\dagger(\beta) \mathbf{Y}(z_0) =
   \begin{bmatrix}
   \cos\beta& \sin\beta& 0& 0& 0& 0\\
    0 & 0 & 0 & \cos\beta& \sin\beta& 0\\
    -\sin\beta& \cos\beta &0 &0 &0 &0\\
    0& 0& 0& -\sin\beta &\cos\beta &0 \\
    0& 0& 1& 0& 0& 0\\
    0& 0& 0& 0& 0& 1\\
   \end{bmatrix},

the initial state of the system in ``preluts_generator`` (preluts standing for pre-look-up tables), where :math:`\mathbf{R}_z(\beta)` is a rotation matrix. Note that this definition of :math:`\mathbf{Y}(z_0)` is as valid as the one defined in eq. :eq:`eq:YXb-z_0`, since linear combinations of :math:`u(z_0)=v(z_0)=0` is in agreement with the boundary conditions.

This initial state is set in the ``PrelutNodeFirst`` class:

.. code-block:: python

   class PrelutNodeFirst(PrelutNode):
       """Specialised first node for preLUT generation."""
       
       def __init__(self, beta, ds):
           """Initialise the first node with the prescribed boundary conditions."""
           PrelutNode.__init__(self)
           self.set_s(sleft=0, sright=ds)
           sinbeta, cosbeta = np.sin(beta), np.cos(beta)
           
           # Initial Y matrix rotated by beta (eq. Y_0_init)
           self.Yleft = np.array([
               [-sinbeta, 0, cosbeta, 0, 0, 0],
               [cosbeta, 0, sinbeta, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, -sinbeta, 0, cosbeta, 0, 0],
               [0, cosbeta, 0, sinbeta, 0, 0],
               [0, 0, 0, 0, 0, 1],
           ], dtype=np.complex128).T
           
           self.Rleft = np.eye(6, dtype=np.complex128)

This initial state is integrated upwards from :math:`z_{min}` to :math:`z_{max}` using :eq:`eq:ytrick` and the midpoint method based on 2nd-order Runge Kutta integration, until we have the solution everywhere, :math:`\mathbf{Y}(z)`.

We use integration steps of adaptive size; the midpoint method gives an error estimate after each completed step, representing the degree to which the function is varying, and the size of the next step is adjusted accordingly:

.. code-block:: python

   def integrate_between_stations(self, p, h, yerr, acc, j):
       """
       Adjust the integration step size until the accuracy requirement is met.
       
       Args:
           p: The current node in the preLUT generation process.
           h: The current integration step size.
           yerr: The accumulated error in the state.
           acc: The accuracy goal for the integration.
           j: Coordinate system indicator (COORD_T for t, COORD_S for s).
       """
       while True:
           sright = p.sright
           Yright, h, s2, lastkz = integrate_between_stations(
               p.Yleft, p.sleft, sright,
               p.dbx_const, p.dby_const, p.dbz_const,
               p.dbx_lin, p.dby_lin, p.dbz_lin,
               h, yerr, acc, j,
               self.kz0, self.lastkz, self.zeta0,
               self.cdivkL, self.psi0, self.cosbeta, self.sinbeta
           )
           self.lastkz = lastkz
           p.Yright = Yright
           
           if s2 >= p.sright:
               break
           
           # Create substation if needed
           p.sright = s2
           self.nodes.append(p)
           p = p.get_next(s2, sright)
       
       return p, h

At each integration step, once we have :math:`\mathbf{Y}(z)`, we pre-evaluate :math:`\Delta \mathbf{b}` through :eq:`eq:byf` while keeping the forcing general within the triangular function approximation.

The :math:`\Delta\mathbf{b}` integral can be solved even though the starting values :math:`b_i(z_0)` are unknown for :math:`i\in[4,6]`, because those are constants that can be added after the integration, since

.. math::

   b_i(z_j) = b_i(z_0) + \int_{z_0}^{z_j}\mathbf{Y}^\dagger(z)\mathbf{F}(z) \,\mathrm{d}z.

In order to accommodate any forcing, we solve the integral separately for the constant and the linear part of :math:`\phi_n(z)` for each direction of the forcing. This yields six integrals for each integration step, which are stored in a LUT:

.. code-block:: python

   def modified_midpoint_integration_step(
       y, x, h, j, kz0, psi0, lastkz, zeta0, cdivkL, cosbeta, sinbeta,
       dbx_const, dby_const, dbz_const, dbx_lin, dby_lin, dbz_lin
   ):
       """
       Perform integration step and update forcing accumulators.
       
       Returns:
           Tuple of (Yright, yerr) where Yright is the updated state matrix
           and yerr is the error estimate.
       """
       # ... RK2 integration steps (omitted for brevity) ...
       
       # Update the differential forcing accumulators using Simpson's rule
       # These are the Δb integrals stored in the LUT
       if j == COORD_T:
           kz1, kzm, kz2 = [get_kz(x + s * h, zeta0, kz0, lastkz, psi0, cdivkL) 
                            for s in [0.0, 0.5, 1.0]]
           
           if zeta0 < 0:  # Unstable
               a1 = kz1 * phi_inverse(kz1, cdivkL)
               am = kzm * phi_inverse(kzm, cdivkL)
               a2 = kz2 * phi_inverse(kz2, cdivkL)
           else:  # Stable and neutral
               a1 = 1 / (1 / kz1 + cdivkL)
               am = 1 / (1 / kzm + cdivkL)
               a2 = 1 / (1 / kz2 + cdivkL)
           
           # Store constant part of forcing integral
           dbx_const += h * np.conj(a1*C1*y[1,:] + am*C3*y3[1,:] + 
                                     a2*(C4*y4[1,:] + C2*y2[1,:]))
           # Store linear part of forcing integral
           dbx_lin += h * np.conj(kz1*a1*C1*y[1,:] + kzm*am*C3*y3[1,:] + 
                                   kz2*a2*(C4*y4[1,:] + C2*y2[1,:]))
           # ... similar for dby and dbz ...
       
       return yright, yerr

The information stored in the LUT considerably reduces the computation time for a specific forcing input. The remaining unknowns are the forcing amplitudes :math:`f_{i,n}` which will come back into the equations during the inverse Fourier transform (``trafalgar``).

Part 2: flut
^^^^^^^^^^^^

The second part of the code, ``flut`` requires specification of :math:`z_0 \geq z_{min}` and :math:`z_{ih}\leq z_{max}` together with the rotor diameter :math:`D`. For computational efficiency, only the ``prelut`` data within the relevant vertical range :math:`[z_{min}, z_{ih}]` is retained, discarding pre-calculated values outside this domain:

.. code-block:: python

   def solve_layers(args):
       prelut, beta, kz0, z0, zhub, radius, forcing, lowerjf, upperjf, \
           minlevel, maxlevel, low_level_out, high_level_out = args
       
       # Discard prelut data outside the relevant range
       max_table_level = prelut.level.max().item()
       if max_table_level < minlevel:
           return np.zeros((high_level_out - low_level_out + 1, 6), 
                          dtype=np.complex128)
       
       imin, imax = np.searchsorted(prelut.level, 
                                     [minlevel, min(maxlevel, max_table_level)])
       prelut = prelut.sel(i=slice(imin, imax))  # Keep only relevant levels

The solving process is as follows:

- Forcing at each layer within the rotor solved for a specific :math:`kz_0,\beta` and forcing direction.
- Repeat for each :math:`(kz_0,\beta)` pair
- Repeat for each forcing direction (longitudinal, transverse)

In the end, solutions from all layers are weighted and superimposed to produce the final wake field.

Now that the problem geometry is known, we loop over *layers* of the actuator disk. Each layer is defined by the Triangular function centred in the middle of the layer and extending over the two neighbouring *levels* (vertical grid points). Each iteration (integration between top and bottom level) is divided into three distinct parts: :math:`z_0 \rightarrow z_{n-1}`, :math:`z_{n-1}\rightarrow z_{n+1}`, the layer of the current forcing point, and :math:`z_{n+1}\rightarrow z_{ih}`.

Recall that the goal is to formulate complete initial conditions at a single boundary, and that we want to solve for :math:`\mathbf{X}(z)` using eq. :eq:`eq:yxb`. The first three rows of :math:`\mathbf{Y}^\dagger(z_{ih})\mathbf{X}(z_{ih})=\mathbf{b}(z_{ih})` are equivalent to the lower boundary conditions, since there is no mixing with the last three rows during upwards propagation.

Using the input about the turbine (hub height :math:`z_h`, rotor diameter :math:`D`,...) together with the stored information about :math:`\mathbf{Y}(z)` and :math:`\mathbf{F}(z)`, the first step is to determine the first three components of :math:`\mathbf{b}` (:math:`b_1, b_2` and :math:`b_3`) by propagating the problem from :math:`z_0` to :math:`z_{ih}`.

**Upward propagation to build** :math:`b_1, b_2, b_3`:

.. code-block:: python

   def solve_layer(
       R_upper, R_lower, Q, levels,
       db_const, db_lin,
       phi_const_low, phi_lin_low,
       phi_const_high, phi_lin_high,
       icl_m1, icl, icl_p1,
   ):
       icl_m1, icl, icl_p1 = icl_m1.item(), icl.item(), icl_p1.item()
       
       # Initialize b₁,b₂,b₃ = 0 from minlevel to cl-1 (lower BC)
       b_lower_3 = [np.zeros(3, dtype=np.complex128)] * (icl_m1 + 1)
       
       # Forcing increments for the layer (Chapeau function weights)
       forcing_increments_up = np.concatenate((
           (-db_const[icl_m1:icl, :3] * phi_const_low + db_lin[icl_m1:icl, :3] * phi_lin_low),   # cl-1 to cl
           (-db_const[icl:icl_p1, :3] * phi_const_high + db_lin[icl:icl_p1, :3] * phi_lin_high), # cl to cl+1
       ))
       
       # Propagate through rotor region: add forcing contributions
       for F_increment, RR in zip(forcing_increments_up, R_upper[icl_m1:icl_p1, :3, :3]):
           b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1] + F_increment))
       
       # Propagate from cl+1 to maxlevel: no forcing
       for RR in R_upper[icl_p1:-1, :3, :3]:
           b_lower_3.append(np.dot(np.ascontiguousarray(RR.T), b_lower_3[-1]))

The central issue is now that the equation does not satisfy the boundary conditions at :math:`z_{ih}`. To proceed, we use the following method at :math:`z_{ih}`:

- We enforce the lower boundary conditions in :math:`\tilde{\mathbf{Y}}^\dagger(z_{ih})\mathbf{X}(z_{ih})=\tilde{\mathbf{b}}(z_{ih})`. This is eq. :eq:`eq:yxb` at :math:`z=z_{ih}`, but with the last three rows *replaced* by the upper boundary conditions :math:`u(z_{ih})=v(z_{ih})=w(z_{ih})=0`. We define :math:`\tilde{\mathbf{b}}(z_{ih}) =[b_1(z_{ih}),b_2(z_{ih}),b_3(z_{ih}),0,0,0]^T`, where the first three entries are known from :math:`\Delta \mathbf{b}` together with the lower initial conditions.

- We use :math:`\tilde{\mathbf{Y}}^\dagger(z_{ih})` and :math:`\tilde{\mathbf{b}}(z_{ih})` to determine :math:`\mathbf{X}(z_{ih})`. Since this replacement amounts to a change of basis inside the same 6‑dimensional solution space, :math:`\tilde{\mathbf{Y}}^\dagger(z_{ih})` remains invertible, and :math:`\mathbf{X}(z_{ih})=(\tilde{\mathbf{Y}}^\dagger(z_{ih}))^{-1} \tilde{\mathbf{b}}(z_{ih})` yields the correct :math:`\mathbf{X}(z_{ih})` that satisfies both the dynamics and the boundary conditions.

- We already know the first three components of :math:`\mathbf{b}(z_{ih})` and use the :math:`\mathbf{X}(z_{ih})` that we just found to determine the last three components from the original eq. :eq:`eq:yxb`: :math:`\mathbf{b}(z_{ih}) = \mathbf{Y}^\dagger(z_{ih})\mathbf{X}(z_{ih})`. These last three components (:math:`b_4, b_5, b_6`) carry information from the upper boundary conditions downward through the differential equation dynamics.

**Enforcing boundary conditions at** :math:`z_{ih}`:

.. code-block:: python

       # At inversion height: enforce upper boundary conditions
       # Construct Y_tilde = [Q*(z_ih) first 3 rows; constraints for u,v,w=0]
       Y_tilde = np.concatenate((
           np.conj(Q[-1, :3]),
           np.array([[1, 0, 0, 0, 0, 0],    # u(z_ih) = 0
                     [0, 0, 1, 0, 0, 0],    # v(z_ih) = 0
                     [0, 0, 0, 0, 1, 0]],   # w(z_ih) = 0
                    dtype=np.complex128),
       ))
       
       # Construct b_tilde = [b₁(z_ih), b₂(z_ih), b₃(z_ih), 0, 0, 0]ᵀ
       b_tilde = np.concatenate((b_lower_3[-1][:3], np.asarray([0+0j, 0, 0])))
       
       # Solve for X(z_ih)
       x_ih = linalg.solve(Y_tilde, b_tilde)
       
       # Compute b₄,b₅,b₆ from original Y†X = b
       Y_ih_last_3 = np.conj(Q[-1, 3:])  # Last 3 rows of Y†(z_ih)
       b_full_6 = [np.concatenate((b_lower_3.pop(), Y_ih_last_3 @ x_ih))]

In this way, we have effectively collected the boundary conditions at :math:`z_{ih}`, and the goal of the calculations can be reached: Using the stored information about :math:`\mathbf{Y}(z)` and the pre-calculated forcing integrals, we propagate :math:`\mathbf{X}(z)` *downwards* from :math:`z_{ih}` to :math:`z_0`. At each level, we also find :math:`\mathbf{b}` through :eq:`eq:yxb`. Reaching the bottom, we now have :math:`\mathbf{X}(z)` and :math:`\mathbf{b}(z)` for the entire vertical range.

**Downward propagation:**

.. code-block:: python

       # Propagate downward from maxlevel to cl+1: no forcing
       for RL in R_lower[-1:icl_p1:-1, :, 3:]:
           b_full_6.append(np.concatenate((
               b_lower_3.pop(),
               np.dot(np.ascontiguousarray(RL.T), b_full_6[-1])
           )))
       
       # Adjust indices for downward direction
       icl_m1, icl, icl_p1 = icl_m1 - 1, icl - 1, icl_p1 - 1
       
       # Forcing increments (note sign change for downward)
       forcing_increments_down = np.concatenate((
           (+db_const[icl_p1:icl:-1, 3:] * phi_const_high - db_lin[icl_p1:icl:-1, 3:] * phi_lin_high),  # cl+1 to cl
           (+db_const[icl:icl_m1:-1, 3:] * phi_const_low  - db_lin[icl:icl_m1:-1, 3:] * phi_lin_low),   # cl to cl-1
       ))
       
       # Propagate through rotor region: add forcing to last 3 components
       for F_increment, RL in zip(forcing_increments_down,
                                   R_lower[icl_p1+1:icl_m1+1:-1, :, 3:]):
           b_full_6.append(np.concatenate((
               b_lower_3.pop(),
               np.dot(np.ascontiguousarray(RL.T), b_full_6[-1]) + F_increment
           )))
       
       # From cl-1 to minlevel: last 3 components propagate, first 3 are zero
       for RL in R_lower[icl_m1+1:0:-1, :, 3:]:
           b_full_6.append(np.concatenate((
               np.zeros(3, dtype=np.complex128),
               np.dot(np.ascontiguousarray(RL.T), b_full_6[-1])
           )))
       
       # Extract solutions at level changes only (excluding substations)
       new_level = np.concatenate((np.array([True]), levels[1:-1] > levels[:-2]))
       
       return [
           (np.ascontiguousarray(Q_level.T) @ b_full_6)
           for Q_level, b_full_6 in list(zip(
               [v for v, nl in zip(Q[1:], new_level) if nl],
               [v for v, nl in zip(b_full_6[::-1][1:], new_level) if nl],
           ))
       ]

Part 3: trafalgar
^^^^^^^^^^^^^^^^^

In the last part of the code, the solution of the problem found in ``flut`` is inverse Fourier transformed in order to get the solution to the original problem in spatial coordinates.

Summary of the three-part solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``preluts_generator``: Solves :math:`\frac{d\mathbf{Y}}{dz} = \mathbf{M}\mathbf{Y}` upward from :math:`z_{min}` to :math:`z_{max}` for all :math:`(\beta, kz_0)` pairs. Pre-calculates and stores :math:`\Delta\mathbf{b}` integrals for generic forcing.

2. ``flut``: Given turbine geometry :math:`(z_0, z_{ih}, D, z_h)`, propagates :math:`b_{1,2,3}` upward through rotor layers, enforces boundary conditions at :math:`z_{ih}` to find :math:`\mathbf{X}(z_{ih})` and :math:`b_{4,5,6}`, then propagates :math:`\mathbf{X}(z)` downward to obtain the solution at all levels.

3. ``trafalgar``: Inverse Fourier transforms the spectral solution to obtain the wake field in physical space.

.. _modified-chasing-method:

The modified Chasing method
----------------------------

The Chasing method as described above faces two fundamental challenges:

**Challenge 1: Subspace mixing.**

At the ground level :math:`z_0`, only three of the six components of :math:`\mathbf{b}` are known from the boundary conditions (:math:`b_1=b_2=b_3=0`), while the last three (:math:`b_4, b_5, b_6`) are arbitrary initial values. When we integrate :math:`\mathbf{Y}(z)` upward using equation :eq:`eq:ytrick`, these two subspaces—the constrained subspace (first three columns of :math:`\mathbf{Y}`) and the free subspace (last three columns)—begin to mix through the dynamics encoded in :math:`\mathbf{M}`. Without a mechanism to maintain their separation, the values of :math:`b_1(z_{ih}), b_2(z_{ih}), b_3(z_{ih})` at the inversion height become dependent on our arbitrary initial choice of :math:`b_4(z_0), b_5(z_0), b_6(z_0)`. This makes it impossible to correctly apply the upper boundary conditions.

**Challenge 2: Numerical ill-conditioning.**

Even if we could conceptually separate these subspaces, the matrix :math:`\mathbf{Y}(z)` becomes numerically ill-conditioned during upward integration. This means it develops a mixture of very small and very large singular values, causing severe loss of numerical precision when the matrix must be inverted or used in solving systems of equations. This occurs due to the limited precision of computer arithmetic, regardless of how carefully we initialize :math:`\mathbf{Y}(z_0)` to be unitary.

To address both challenges simultaneously, we introduce the *modified Chasing method*, which uses QR decomposition at regular intervals to maintain both the structural separation of subspaces and numerical stability.

For this use, we define a set of integration *stations*, the exponentially separated :math:`z`-values

.. math::

   z_j = z_{min}\mathrm{e}^{j\Delta s},

where it has been found that the value :math:`\Delta s = 0.05` is suitable. The spacing between the stations is much larger than the spacing between the :math:`z_n`-levels, which are also exponentially distributed. On the smallest scale, we have the integration steps of infinitesimal, adaptive size. The solver accounts for levels on multiple scales by adjusting the integration step if it would cross a station otherwise.

**Creating stations:**

.. code-block:: python

   def make_prelut(self):
       """Generate the preLUT dataset by integrating between stations."""
       self.nodes = []
       first = PrelutNodeFirst(self.beta, self.ds)
       h = np.sqrt(self.acc * 6 / 3.125)
       
       sm = self.sm()  # Transition point between coordinate systems
       
       # Create station locations
       s_lst = np.sort(np.r_[
           0, 
           np.cumsum(np.full(int(self.smaxx // self.ds) + 1, self.ds)), 
           sm
       ])
       
       # Integrate between stations
       segment, h = self.integrate_between_stations(
           first, h, yerr, self.acc, COORD_T
       )
       
       for s1, s2 in tqdm(list(zip(s_lst[1:], s_lst[2:])), disable=True):
           self.nodes.append(segment)
           segment = segment.get_next(s1, s2)
           
           # Choose coordinate system based on whether below/above sm
           coordsys = COORD_T if s1 < sm else COORD_S
           
           segment, h = self.integrate_between_stations(
               segment, h, yerr, self.acc, coordsys
           )

The problem is "reset" at each integration station :math:`z_j`, meaning that we perform a decomposition of :math:`\mathbf{Y}(z)` to obtain a modified matrix which is unitary at that level. Again, the choice comes from the fact that a unitary matrix :math:`\mathbf{U}` is very well-conditioned, i.e. it is as protected as possible from turning ill-conditioned from mixing. Besides of this, it is extremely easy to invert due to the property :math:`\mathbf{U}^{-1}=\mathbf{U}^\dagger`, which is very convenient when solving for :math:`\mathbf{X}(z)` in ``flut``. Then we simply have :math:`\mathbf{X}(z)=\mathbf{Y}(z)\mathbf{b}(z)`, which makes the computation more stable.

The decomposition is possible because when defining a matrix :math:`\mathbf{B}` so that :math:`\mathbf{Y}^{'}=\mathbf{Y}\mathbf{B}^\dagger` and :math:`\mathbf{b}^{'}=\mathbf{B}\mathbf{b}`, we recover the same equations :eq:`eq:byf`, :eq:`eq:ytrick` for :math:`\mathbf{Y}^{'}` and :math:`\mathbf{b}^{'}`, assuming that :math:`\mathbf{B}` is *independent* of :math:`z` between two vertical levels. The matrix :math:`\mathbf{B}` generally differs from level to level.

The type of decomposition we use here is the QR decomposition where :math:`\mathbf{Y}^{'}=\mathbf{Q}` and :math:`\mathbf{B} = (\mathbf{R}^\dagger)^{-1}` so that

.. math::

   \mathbf{Y}(z_j) = \mathbf{QR},

where :math:`\mathbf{R}` is upper triangular and represents the change of basis between stations while :math:`\mathbf{Q}` is unitary, expressing :math:`\mathbf{Y}` in an orthonormal basis at that specific station. It is essentially a Gram--Schmidt orthonormalisation of the vectors formed by the columns of :math:`\mathbf{Y}`.

**QR decomposition at stations:**

.. code-block:: python

   def gmres(Yright):
       """
       Perform QR decomposition to reset numerical conditioning.
       
       Returns:
           Tuple of (Rleft, Yleft, Rright) where:
           - Yleft (Q): Unitary orthonormal basis
           - Rleft: Lower triangular transformation matrix
           - Rright: Inverse for propagation
       """
       Yleft, Rleft = np.linalg.qr(Yright)
       Rleft = np.conj(Rleft.T)  # Make lower triangular
       Rright = np.linalg.inv(Rleft)
       return Rleft, Yleft, Rright
   
   class PrelutNode:
       def get_next(self, sleft, sright):
           """Generate next node via QR decomposition."""
           next_node = self.GMRES()
           next_node.set_s(sleft, sright)
           return next_node
       
       def GMRES(self):
           """Perform QR decomposition and store results."""
           node = PrelutNode()
           Rleft, Yleft, Rright = gmres(self.Yright)
           node.Rleft = Rleft
           node.Yleft = Yleft
           self.Rright = Rright  # Store for downward propagation in flut
           return node

The QR decomposition addresses both challenges:

- **Subspace separation:** The upper triangular structure of :math:`\mathbf{R}` (and thus its inverse :math:`\mathbf{B}`) ensures that the first three columns of :math:`\mathbf{Y}` (constrained by lower BC) remain independent of the last three columns (free subspace). This is because matrix multiplication with an upper triangular matrix cannot mix the first three columns with the last three.

- **Numerical stability:** By resetting to an orthonormal basis :math:`\mathbf{Q}` at each station, we prevent the accumulation of numerical errors that would otherwise make :math:`\mathbf{Y}` ill-conditioned.

Thus, the modification of the Chasing method amounts to performing the QR decomposition at each station :math:`z_j`. We save the value of :math:`\mathbf{R}` and proceed with the integration using :math:`\mathbf{Q}`. Meanwhile, the norm of :math:`\mathbf{Y}(z)` is monitored continuously, so that if :math:`\mathbf{Y}(z)` becomes ill-conditioned between stations, substations are created to perform additional QR decompositions:

.. code-block:: python

   # Inside integrate_between_stations function
   while True:
       # ... perform integration step ...
       
       Ynorm = np.linalg.norm(Yright[:, 0]) / Ynorm1
       
       # If Y becomes ill-conditioned, create a substation
       if Ynorm > Ythreshold:
           if j == COORD_T:
               kz1 = get_kz(t, zeta0, kz0, lastkz, psi0, cdivkL)
               lastkz = kz1
               s1 = np.log(kz1 / kz0)
           else:
               s1 = np.log(t / kz0)
           
           return Yright, h, s1, lastkz  # Return early to create substation

In ``preluts_generator``, we store the :math:`\mathbf{Q}` and :math:`\mathbf{R}` matrices at each station for later use. In ``flut``, when propagating downward from :math:`z_{ih}` to :math:`z_0`, we apply these transformations in sequence. To find :math:`\mathbf{X}(z_j)` from :math:`\mathbf{X}(z_{j+1})`, we use the relation :math:`\mathbf{X}(z_j) = \mathbf{Q}(z_j) \mathbf{b}(z_j)` where :math:`\mathbf{b}(z_j) = \mathbf{R}_j^{-1} \mathbf{b}(z_{j+1})`. This allows us to propagate the solution downward while maintaining both numerical stability and the correct boundary condition structure throughout the domain.

Appendix: Numerical solution
----------------------------

See :download:`appendices/numerical_solution.pdf` for the detailed derivation of the numerical solution.