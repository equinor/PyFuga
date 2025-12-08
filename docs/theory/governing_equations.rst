.. _theory-governing-equations:

Governing equations
===================

The starting point for Fuga is a full, non-linear computational fluid dynamics (CFD) model. This 
can be any CFD model based on the *Boussinesq approximation* and an *eddy viscosity closure*. In 
such a setting, the Reynolds-averaged Navier-Stokes equation can be expressed like this:

.. math::

    U_j  \frac{\partial U_i}{\partial x_j } = \frac{\partial}{\partial x_j} K 
    \left( \frac{\partial U_i}{\partial x_j} + \frac{\partial U_j}{\partial x_i} \right)
    - \frac{\partial P}{\partial x_i } + f_i,

where :math:`U_i` is the mean velocity, :math:`P` is the pressure divided by the (constant) 
density, :math:`K` is the eddy viscosity (molecular viscosity is neglected), and :math:`f_i` is the 
external forcing from one or more actuator discs representing the drag forces exerted by turbines. 
For the sake of simplicity, the buoyancy term :math:`-g \theta \delta_{3i}/T` and the corresponding 
transport equation for potential temperature :math:`\theta` have been ignored. This means that 
standing gravity waves are not supported, but turbulent mixing can still be affected by atmospheric 
stability through the eddy viscosity. The Coriolis force has also been ignored. In the absence of 
turbines, we assume that conditions are horizontally homogeneous. This could imply a finite mean 
horizontal pressure gradient, constant with height, which could drive the flow. However, this only 
works if the atmosphere has a finite depth, since otherwise the pressure force on a column of air 
would be infinite, balanced by a Reynolds stress steadily increasing with height. In reality, the 
pressure gradient is balanced by the Coriolis force, while the Reynolds stress dies out with 
height. We therefore follow the normal practice of Monin-Obukhov theory and assume that the mean 
pressure gradient vanishes. This means that the vertical momentum flux is constant with height, 
which can be modelled by assuming that the flow is lid-driven, i.e. that the atmosphere is capped 
by a solid plate/lid moving with a prescribed, constant velocity. Lastly, the incompressibility 
condition yields the continuity equation,

.. math::
    
    \frac{\partial U_i}{\partial x_i} = 0.

Boundary conditions
-------------------

Boundary conditions are just as important as governing equations. Assuming a lid-driven flow with 
an imposed velocity $U_i^{\rm lid}$ at the top of the boundary layer and rough lower boundary, the 
boundary conditions are

.. math::

    \begin{aligned}
        U_i (z_0) &= 0 \\
        U_i (z_i) &= U_i^{\textrm{lid}},
    \end{aligned}

where :math:`z_0` is the roughness length and :math:`z_i` is the boundary layer height (the 
:math:`i` in :math:`z_i` stands for capping inversion, not component index). Specifying the 
momentum flux at the lid is an alternative, but in practice, the choice is not particularly 
important. It would, of course, be more accurate to include the Coriolis force and thus assume 
geostrophic balance at the top of the boundary layer; however, the lid-driven approach is preferred 
because it is simpler. 

Closure
-------

Finally, the eddy viscosity :math:`K` is to be determined by means of a closure scheme. In 
:cite:`Ott2011` three different closures were studied for neutral conditions: the standard 
:math:`k`-:math:`\varepsilon` model, the mixing length closure and the 'simple' closure, where 
:math:`K = \kappa u_* z`, with  :math:`\kappa` being the von Kármán constant and :math:`u_\star` 
the friction velocity. Somewhat surprisingly, the simple closure performed better than the other 
two and has therefore been used subsequently. The simple closure totally neglects any feedback 
mechanism that modifies turbulent mixing (of momentum) in the presence of wakes. The two other 
models are supposed to do that, but apparently without much success. At the time of this 
investigation, it was realised that the (non-linear) standard :math:`k`-:math:`\varepsilon` model 
yields poor results for wakes, and a modified :math:`k`-:math:`\varepsilon`-:math:`f_p` model 
(proposed by :cite:`Vanderlaan2015`) is necessary in order to beat the performance of the simple 
model. It is, therefore, possible that a better choice than the simple closure exists. 
Investigating this is, however, outside the scope of the present work. 

In Fuga, the simple closure is extended to non-neutral conditions by means of Monin-Obukhov theory, 
setting 

.. math::

    K = \frac{\kappa u_*  z}{\phi_m (z/L)},

where :math:`L` is the Monin-Obukhov length and :math:`\phi_m` is the Högström profile function 
\citep{Hogstrom1988}, defined as

.. math::

    \phi_m (z/L) = \begin{cases}
    1+5\frac{z}{L}  & \text{for } \frac{z}{L} \geq 0 \\
    (1-19.3\frac{z}{L})^{^-1/4} & \text{for } \frac{z}{L} < 0.
    \end{cases}

.. _sec-theory-actuator-disc:

Actuator disc model
-------------------

The rotor is modelled as an actuator disc model that exerts a force :math:`f_1` in the 
:math:`x`-direction and is assumed to be perpendicular to the wind, i.e. zero yaw offset. The drag 
force is determined by the free wind, i.e. the wind speed at hub height is determined before 
placing the turbine. For a given free wind, :math:`U_{\text{free}}`, along with the :math:`x`-axis 
and the hub placed at :math:`(x_h, y_h, z_h)` the drag force is modelled as,

.. math::

    f_1 =-\frac{1}{2} C_t U^2_{\text{free}} \delta(x-x_h) \Theta \left(R^2-(y-y_h)^2-(z-z_h)^2 \right),

where :math:`\Theta` is the Heaviside step function and :math:`\delta` is a Dirac 
:math:`\delta`-function. In calculations, the force is always smeared out for numerical reasons. 
This is done by replacing the :math:`\delta`-function by a narrow Gaussian. In classical Fuga, we 
do not include yaw and tilt, corresponding to :math:`f_2 = f_3 = 0`. See :doc:`yawed_extension` for 
an extension including yaw.


