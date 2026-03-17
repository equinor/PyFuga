.. _theory-linearisation:

Linearisation
=============

The linearisation employed by Fuga is achieved through the application of perturbation theory, 
where an approximate solution to the non-linear RANS equations for the disturbed flow field 
(with turbines) is found starting from the solution to the undisturbed flow field (without 
turbines). There are multiple ways to linearise the RANS equations using perturbation theory; 
therefore, we will explain how this has been done in Fuga.

In the present version of Fuga, the drag force :math:`f_i` is scaled by a 'small' perturbation 
parameter :math:`\epsilon`, such that :math:`f_i` becomes :math:`\epsilon f_i` in the governing 
Navier-Stokes equations. The approximate solution is then a function of :math:`\epsilon`, and the 
velocity :math:`U_i` and pressure :math:`P` fields can be expanded as Taylor series in powers of 
:math:`\epsilon`, such that

.. math::
   :label: eq:perturbation_expansion

   \begin{aligned}
      U_i &= U_i^0 + \epsilon u_i^1 + \epsilon^2 u_i^2 + \epsilon^3 u_i^3 + \ldots \\
      P &= P^0 + \epsilon p^1 + \epsilon^2 p^2 + \epsilon^3 p^3 + \dots.
   \end{aligned}


Note that whilst the superscripts for :math:`\epsilon` here are powers, they are not for the 
velocity or pressure, where they instead represent the order of the perturbation. Moreover, both 
lower-case :math:`u` and the exponents indicate perturbation, where lower-case is used to be 
consistent, though not identical, to the manner of expressing the fluctuating component in Reynolds 
decomposition. 

Zeroth-order equations
----------------------

By substituting these expansions into the momentum, continuity and boundary 
condition equations, the :math:`n^{\mathrm{th}}`-order equations are obtained by applying 
:math:`\partial^n / \partial \epsilon^n` to both sides and then setting :math:`\epsilon = 0`. In other 
words, the :math:`n^{\text{th}}`-order equation is obtained by balancing all terms proportional to 
:math:`\epsilon^n`. As a result, the zeroth-order equations can be written as:

.. math::
   :label: eq:zeroth_order_equations

   \begin{aligned}
      U_j^0  \frac{\partial U_i^0}{\partial x_j } &= \frac{\partial}{\partial x_j} K \
      \left( \frac{\partial U_i^0}{\partial x_j} + \frac{\partial U_j^0}{\partial x_i} \right) - \
      \frac{\partial P^0}{\partial x_i} \\
      \frac{\partial U_i^0}{\partial x_i} &= 0 \\
      U_i^0(z_0) &= 0 \\
      U_i^0(z_i) &= U_i^{\mathrm{lid}}.
   \end{aligned}

This is the same as the governing equations without any external forcing. As discussed above, we 
assume that the mean pressure gradient is zero, allowing us to set :math:`P^0=0`. In the absence of 
external forcing, the momentum balance becomes primarily driven by turbulent shear stress and 
Coriolis forces. As a result, the mean pressure gradient can be assumed to vanish, allowing us to 
set :math:`P^0 = 0`. The solution then becomes the familiar Monin-Obukhov velocity profile -- the 
logarithmic profile for neutral conditions.

.. _sec-linearisation-first-order:

First-order equations
---------------------

The zeroth-order equations provide a base flow solution that describes the undisturbed flow field. 
This base solution serves as the foundation for the perturbative analysis. We now turn to the 
first-order equations, which account for the perturbations introduced by the turbine drag force. 
These equations describe how the presence of the turbines modifies the velocity and pressure fields 
relative to the base solution and can be written as

.. math::
   :label: eq:first_order_equations

   \begin{align}
         &U^0  \frac{\partial u_i^1}{\partial x} + w^1 \frac{\partial U^0}{\partial z} \delta_{i1} \
         = \frac{\partial}{\partial x_j} K \left( \frac{\partial u_i^1}{\partial x_j} + \
         \frac{\partial u_j^1}{\partial x_i} \right) - \frac{\partial p^1}{\partial x_i}+f_i \\
         &\frac{\partial u_i^1}{\partial x_i} \hspace{0,26cm}= 0 \\
         &u_i^1 (z_0) = 0 \\
         &u_i^1 (z_i) \hspace{0,06cm}= 0.
   \end{align}

We assume here that the approaching flow is along the :math:`x`-axis and only a function of the 
vertical, :math:`z`, coordinate, such that :math:`U_i^0 = \delta_{i1}U^0(z)`. Additionally, it 
should be noted that the eddy viscosity, :math:`K`, is assumed to remain unaffected by the 
perturbations caused by the turbines, meaning :math:`K=K^0`, as in the undisturbed flow. 

For the full derivation, please refer to:

- :download:`Appendix A <appendices/linearisation.pdf>`: *Linearisation using perturbation theory*
