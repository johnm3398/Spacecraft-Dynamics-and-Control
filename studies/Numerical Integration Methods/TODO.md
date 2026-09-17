# TODO - Numerical Integration Methods Study

This study is intentionally deferred from the momentum-exchange-device course notebook.

The current notebook reuses a classical fixed-step RK4 integrator because it is transparent, easy to inspect, and sufficient for demonstrating how a generic numerical integrator can propagate a coupled spacecraft attitude and momentum-device state.

A separate study should compare numerical integration methods more systematically rather than treating RK4 as a default solution for every simulation problem.

## Topics to Study

- [ ] Forward Euler as a first-order baseline
- [ ] RK2 / Heun methods
- [ ] Classical fixed-step RK4
- [ ] Adaptive Runge-Kutta methods such as RKF45 / Dormand-Prince
- [ ] Higher-order adaptive methods such as DOP853
- [ ] Step-size selection, local truncation error, and global error
- [ ] Accuracy versus computational cost
- [ ] Behaviour for long-duration attitude propagation
- [ ] Conservation of angular momentum and energy in torque-free rigid-body problems
- [ ] Sensitivity of MRP propagation and shadow-set switching to timestep choice
- [ ] Stiffness and when implicit integrators may become useful
- [ ] Geometric / structure-preserving integration methods for rotational dynamics

## Suggested Comparison Problems

1. Scalar ODE with a known analytical solution.
2. Torque-free rigid-body rotation.
3. MRP attitude kinematics with prescribed body rate.
4. Coupled spacecraft + reaction-wheel dynamics.
5. Coupled spacecraft + multi-VSCMG dynamics.

For each case, compare numerical error, invariant drift, runtime, and sensitivity to timestep.

The objective is not merely to identify a single "best" integrator, but to understand why different numerical methods are appropriate for different spacecraft-dynamics problems.
