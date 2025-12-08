.. _theory-mixed-spectral:

Mixed-spectral formulation
==========================

The first-order equations are most conveniently solved in a mixed-spectral setting. This means that 
the equations are Fourier transformed in the horizontal :math:`x` and :math:`y` coordinates so that 
solutions depend on :math:`z` and a 2D wave vector 
:math:`\mathbf{k} = (k_x, k_y) = (k \cos{\beta}, k \sin{\beta})`. The Fourier transformed 
variables, :math:`\mathcal{F}\{g(x, y, z)\} = \hat{g}(k_x, k_y, z)` read 

.. math::

    \begin{aligned}
        \hat{u}_i^1(k_x,k_y,z) &=\frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\mathrm{d}x \
        \int_{-\infty}^{\infty}\mathrm{d}y\,  u_i^1(x,y,z) \mathrm{e}^{-i(x k_x + y k_y)} \\
        \hat{p}^1(k_x,k_y,z) &=\frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\mathrm{d}x \
        \int_{-\infty}^{\infty}\mathrm{d}y\,  p^1(x,y,z) \mathrm{e}^{-i(x k_x + y k_y)} \\
        \hat{f}_i(k_x,k_y,z) &=\frac{1}{(2\pi)^2}\int_{-\infty}^{\infty}\mathrm{d}x \
        \int_{-\infty}^{\infty}\mathrm{d}y\,  f_i(x,y,z) \mathrm{e}^{-i(x k_x + y k_y)} \\
        &= \frac{1}{(2\pi)^2} \mathrm{e}^{-ix_hk_x} \int_{-\infty}^{\infty}\mathrm{d}y\, \
         f_i(x_h,y,z) \mathrm{e}^{-iy k_y},
    \end{aligned}

where we have used the actuator disc definition from :ref:`sec-theory-actuator-disc`. In the case 
of Gaussian smearing, :math:`\hat{f}_i` is multiplied by :math:`\mathrm{e}^{-(k_x \sigma_x)^2/2}`, 
where :math:`\sigma_x` specifies the width of the actuator disc. The inverse transforms are, then: 

.. math::

    \begin{aligned}
        {u}_i^1(x,y,z) &=\int_{-\infty}^{\infty}\mathrm{d}k_x\int_{-\infty}^{\infty}\mathrm{d}k_y\, 
        \hat{u}_i^1(k_x,k_y,z) \mathrm{e}^{i(x k_x + y k_y)} \\
        p^1(x,y,z) &=\int_{-\infty}^{\infty}\mathrm{d}k_x\int_{-\infty}^{\infty}\mathrm{d}k_y\, \
        \hat{p}^1(k_x,k_y,z) \mathrm{e}^{i(x k_x + y k_y)} \\
        f_i(x_h,y,z) &=\int_{-\infty}^{\infty}\mathrm{d}k_x\int_{-\infty}^{\infty}\mathrm{d}k_y\,
        \hat{f}_i(k_x,k_y,z) \mathrm{e}^{i(x k_x + y k_y)} \\
        &= \mathrm{e}^{ix_hk_x} \int_{-\infty}^{\infty}\mathrm{d}k_y\, \hat{f}_i(x_h,y,z) \mathrm{e}^{i y k_y}.
    \end{aligned}

Introducing the transformed variables into the set of first-order equations in 
:ref:`sec-linearisation-first-order`, the resulting set of mixed spectral equations is an ordinary 
differential equation in the :math:`z` variable. In order to make the notation tidy we will write
:math:`u`, :math:`v`, :math:`w`, :math:`p`  and :math:`f` rather than :math:`\hat{u}_1^1`, 
:math:`\hat{u}_2^1`, :math:`\hat{u}_3^1`, :math:`\hat{p}^1` and :math:`\hat{f}` in the following. 
Finally, we use the first-order continuity equations to eliminate derivatives of :math:`w` in the 
first-order momentum equation and introduce two new variables :math:`u'` and :math:`v'`, which 
enable us to write the equations as 6 first-order equations, viz. 

.. math::

    \begin{aligned}
        \frac{\partial u}{\partial z} &= u' \\
        \frac{\partial u'}{\partial z} &= \left( k^2 + \frac{ U^0 i k \cos{\beta}}{K} \right) u
        - \frac{1}{K} \frac{\partial K}{\partial z} u'
        + \left( \frac{1}{K} \frac{\partial U^0}{\partial z} - \frac{\partial K}{\partial z} 
        \frac{i k \cos{\beta}}{K} \right) w \\
        &\qquad + \frac{i k \cos{\beta}}{K} p - \frac{f_1}{K} \\
        \frac{\partial v}{\partial z} &= v' \\
        \frac{\partial v'}{\partial z} &= \left( k^2 + \frac{U^0 i k \cos{\beta}}{K} \right) v - 
        \frac{1}{K} \frac{\partial K}{\partial z} v' - \frac{\partial K}{\partial z} \frac{i k \sin{\beta}}{K} w + 
        \frac{i k \sin{\beta}}{K} p - \frac{f_2}{K} \\
        \frac{\partial w}{\partial z} &= - i k \cos{\beta} u - i k\sin{\beta} v \\
        \frac{\partial p}{\partial z} &= - 2 \frac{\partial K}{\partial z} i k \cos{\beta} u - K \, i k \cos{\beta}u' 
        - 2 \frac{\partial K}{\partial z} i k \sin{\beta} v - K \, i k \sin{\beta} v' \\
        & \qquad - \left( U^0 i k \cos{\beta} + k^2 K \right) w + f_3,
    \end{aligned}

with associated boundary conditions

.. math::

    u(z_0) = 0, \quad v(z_0) = 0, \quad w(z_0) = 0, \quad u(z_i) = 0, \quad v(z_i) = 0, \quad 
    \text{ and } \quad w(z_i) = 0.

Now the three independent variables :math:`(x, y, z)` have been replaced by :math:`(k_x, k_y, z)`, 
where :math:`k_x` and :math:`k_y` act as parameters rather than variables. Note that there is one 
independent set of equations for each :math:`k` -- thus, no coupling between equations for 
different values of :math:`k`. This *decoupling* is the big advantage of the mixed-spectral 
formulation. Instead of having coupled equations for millions of variables -- six for each grid 
point -- the mixed-spectral setting only has six equations and six variables. This is why solving 
the equations in the mixed spectral domain is so much faster than solving the system equations in 
the conventional physical domain. The resulting set of decomposed wavenumber-specific sub-problems 
is conveniently solved on a PC or by a cluster working in parallel.

Forcing decomposition
---------------------

All three drag force components are included in Equation (3), but we shall obtain the final 
solution as the sum of three individual solutions, each one obtained by retaining only one 
component, :math:`f_i`, and setting the other two equal to zero. This is possible due to the 
linearity of Equation (3) and the homogeneous boundary conditions in Equation (4). The solution to
Equation (3) (with only one non-zero force component) is particularly simple if 
:math:`f_i(\mathbf{k}, z)` can be written as a product of two functions, one depending only on 
:math:`\mathbf{k}` and one depending only on :math:`z`. This is not generally the case, but we may 
approximate :math:`f_i(\mathbf{k}, z)` by a sum of such functions. One way of doing this is to set 
:math:`f_i(\mathbf{k}, z) = \sum_n f_{i,n}(\mathbf{k})\phi_n(z)`, where :math:`\phi_n(z)` is a set 
of triangular functions (also known as Chapeau functions): 

.. math::

    \phi_n(z) = \Delta_n(z) = 
    \begin{cases}
        \frac{z-z_{n-1}}{z_n-z_{n-1}},& \text{for } z_{n-1}\leq z\leq z_n\\
        \frac{z-z_{n+1}}{z_n-z_{n+1}},& \text{for } z_{n}\leq z\leq z_{n+1}\\
        0 &\text{elsewhere}.
    \end{cases}

In order to save memory, the :math:`z_n`'s are chosen so as to increase exponentially with height 
(about 5% increase per level). Note that a series of adjacent triangular functions add up to a 
constant function of height 1, see the figure below: 

.. figure:: /../_static/triangular_functions.png
   :name: fig:triangular_function
   :align: center
   :width: 70%

   Diagram illustrating the application of triangular basis functions :math:`\phi(z)`, at discrete 
   points :math:`z_{n-1}`, :math:`z_n`, and :math:`z_{n+1}`. Each function is defined over a 
   specific interval, reaching a peak value of 1 at its corresponding point, and is zero outside 
   its defined range. Note that the spacing between each level :math:`z_{n-1}`, :math:`z_n`, 
   :math:`z_{n+1}`, :math:`\ldots` increases exponentially with :math:`z`. The dashed line 
   represents the sum :math:`\sum_{i=n-1}^{n+1} \phi_i(z)`, which maintains continuity across 
   intervals, demonstrating how these basis functions contribute to a piecewise linear 
   approximation.

It is also possible to model drag forces that vary across the rotor or even the drag force from a 
forest canopy.

The factor :math:`f_{i,n}(\mathbf{k})`, on the other hand, contains all rotor information, such as 
thrust coefficient and wind turbine geometry, including the 'thickness' :math:`\sigma_x`, and 
determines the amplitude of the triangular function at each vertical level :math:`z_n`.

If we solve the problem for each orthogonal component of the forcing :math:`f_i(\mathbf{k}, z)` 
individually, and for a single vertical level at a time, the forcing amplitude disappears from 
Equation (3). That is, we set :math:`f_i = f_{i,n}\phi_n` and the two other forcing components to 
zero, yielding a set of equations valid in the case of a single triangular function 
:math:`\phi_n(z)`. This allows the sub-solutions to be used to construct solutions with any form of 
forcing for any vertical level :math:`z_n`. It is much faster to make a linear combination of 
pre-calculated sub-solutions than to solve Equation (3) with the full forcing. Since the equations 
become decoupled in :math:`f_i`, we can combine the sub-solutions much later to account for the 
complete forcing.

Non-dimensional formulation
---------------------------

In the following, we only take into account :math:`f_1` as forcing, so we set :math:`f_2 = f_3 = 0` 
in Equation (3). That means we show one out of three possible (very similar) examples. Furthermore, 
we restrict the equation to the triangular function of a single vertical level :math:`z_n`, so that 
we can replace :math:`f_1` by :math:`f_{1,n}\phi_n`.

We introduce the dimensionless zeroth order velocity in the :math:`x`-direction 
:math:`U^{0*}=U^0/u_*` together with :math:`K^* = K/u_*`, which we note has dimension length and 
appears in :math:`kK^*` factors in the dimensionless equations. We use the relation for wind shear 
in the boundary layer :math:`\partial U^{0*} / \partial z = 1/K^*(z)`. 

Using the new independent variable :math:`s = kz` and the new dependent variables 
:math:`\tilde{u} = u u_* k / f_{1,n}`, :math:`\tilde{v} = v u_* k / f_{1,n}`, 
:math:`\tilde{w} = w u_* k / f_{1,n}` and :math:`\tilde{p} = p k/f_{1,n}`, we find Equation (3) 
with :math:`f_2=f_3=0` to be

.. math::

    \begin{aligned}
        \frac{\partial \tilde{u}}{\partial s} &= \tilde{u}' \\
        \frac{\partial \tilde{u}'}{\partial s} &= \left( 1 + \frac{i \cos{\beta} U^{0*}}{k K^*} \right) \tilde{u} - \frac{1}{K^*} \frac{\partial K^*}{\partial s} \tilde{u}' + \left(\frac{1}{(kK^*)^2} - \frac{1}{K^*} \frac{\partial K^*}{\partial s} i \cos{\beta} \right) \tilde{w} \\
        & \qquad + \frac{ i \cos{\beta}}{k K^*} \tilde{p} - \frac{\phi_n}{k K^*} \\
        \frac{\partial \tilde{v}}{\partial s} &= \tilde{v}' \\
        \frac{\partial \tilde{v}'}{\partial s} &= \left(1+\frac{i\cos{\beta}U^{0*}}{kK^*}\right) \tilde{v} -\frac{1}{K^*} \frac{\partial K^*}{\partial s} \tilde{v}' - \frac{1}{K^*} \frac{\partial K^*}{\partial s}i \sin{\beta} \tilde{w} + \frac{ i \sin{\beta}}{k K^*} \tilde{p} \\
        \frac{\partial \tilde{w}}{\partial s} &= - i \cos{\beta} \tilde{u} - i \sin{\beta}\tilde{v} \\
        \frac{\partial \tilde{p}}{\partial s} &= - 2 i k \frac{\partial K^*}{\partial s}(\cos{\beta}\tilde{u} + \sin{\beta} \tilde{v}) - i k K^* (\cos{\beta} \tilde{u}' + \sin{\beta} \tilde{v}') \\
        & \qquad - \left(k K^* + U^{0*} i \cos{\beta} \right) \tilde{w},
    \end{aligned}

valid at the vertical level :math:`z_n`. The great advantage of this formulation is that it is 
independent of the information about the specific rotor in question -- we can solve the problem by 
finding sub-solutions for :math:`\phi_n(z)`. The solution with 
:math:`f_i(\mathbf{k}, z) = \sum_n f_{i,n}(\mathbf{k})\phi_n(z)` can then be obtained as the 
weighted sum of the sub-solutions, using :math:`\{f_{i,n}(\mathbf{k})\}` as weights.

Equation (6) is well suited for the upper part of the boundary layer, but very unstable near the 
surface. For :math:`s < s_{\mathrm{tr}}`, where :math:`s_{\mathrm{tr}}` is the value of :math:`s` 
for which :math:`\frac{\partial}{\partial s} = \frac{\partial}{\partial t}` we use an alternative 
formulation with :math:`t = U^{0}\kappa/u_*` as the independent variable,

.. math::

    \begin{aligned}
        \frac{\partial \tilde{u}}{\partial t} &= \tilde{u}' \\
        \frac{\partial \tilde{u}'}{\partial t} &= \left( \frac{U^{0*} i \cos{\beta} k K^*}{\kappa^2} + \frac{(k K^*)^2}{\kappa^2} \right) \tilde{u} + \left( \frac{1}{\kappa^2} - \frac{k}{\kappa} \frac{\partial K^*}{\partial t} i \cos{\beta} \right) \tilde{w} \\
        & \qquad + \frac{i \cos{\beta} k K^*}{\kappa^2} \tilde{p} - \frac{k K^* \phi_n}{\kappa^2} \\
        \frac{\partial \tilde{v}}{\partial t} &= \tilde{v}' \\
        \frac{\partial \tilde{v}'}{\partial t} &= \left( \frac{U^{0*} i \cos{\beta} k K^*}{\kappa^2}+\frac{(k K^*)^2}{\kappa^2} \right) \tilde{v} - \frac{k}{\kappa} \frac{\partial K^*}{\partial t}i \sin{\beta} \tilde{w} + \frac{i \sin{\beta} k K^*}{\kappa^2} \tilde{p} \\
        \frac{\partial \tilde{w}}{\partial t} &= - \frac{ikK^*}{\kappa} \left( \cos{\beta} \tilde{u} + \sin{\beta} \tilde{v} \right) \\
        \frac{\partial\tilde{p}}{\partial t} &= - 2 i k \frac{\partial K^*}{\partial t} (\cos{\beta}\tilde{u} + \sin{\beta}\tilde{v}) - i k K^* (\cos{\beta} \tilde{u}' + \sin{\beta} \tilde{v}') \\
        & \qquad - \left( \frac{(kK^*)^2}{\kappa} + \frac{U^{0*} i \cos{\beta} kK^*}{\kappa} \right) \tilde{w}.
    \end{aligned}

The non-dimensional solutions then become functions of :math:`k z` and three parameters: 
:math:`k z_0`, the angle :math:`\beta` between the wave vector and the wind direction, and a 
stability parameter expressed in terms of the surface roughness height and the Monin-Obukhov length 
:math:`\zeta_0 \equiv z_0/L`, say. Note that there is no additional, explicit dependence on 
:math:`k` or :math:`u_*` or :math:`z_0`, so we solve the equations without actually knowing 
:math:`k`, :math:`u_*` and :math:`z_0`. Details of the formulation, both near the surface and in 
the upper part of the boundary layer, and their implementation can be found in:

- :download:`Appendix B <appendices/s_to_t.pdf>`: *Transition between independent variables*
- :download:`Appendix C <appendices/mixed_spectral.pdf>`: *Governing equations in mixed-spectral form*
