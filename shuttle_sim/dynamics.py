"""3D Gaussian evolution: exact piecewise affine vs adaptive ODE.

No RF micromotion, stochastic heating or non-Gaussian states.
Internal scaling: L=1 um, T=1 us; public inputs/outputs are SI.
"""
import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp
from .field import Rejected, finite

L=1e-6
T=1e-6


def initial_state(local, mass, nbar=0.):
    if not np.isfinite(mass) or mass<=0 or not np.isfinite(nbar) or nbar<0:
        raise ValueError('positive mass and nonnegative finite nbar required')
    w=finite(local.omega_rad_s,'omega_rad_s');R=finite(local.modes,'modes')
    if w.shape!=(3,) or np.any(w<=0) or R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-10):
        raise ValueError('three positive frequencies and orthonormal modes required')
    C=np.zeros((6,6))
    C[:3,:3]=R@np.diag((2*nbar+1)*hbar/(2*mass*w))@R.T/L**2
    C[3:,3:]=R@np.diag((2*nbar+1)*hbar*w/(2*mass))@R.T*(T/L)**2
    return np.r_[local.position_m/L,np.zeros(3)],C


def summarize(z,C,local,mass,target_m):
    C=(C+C.T)/2
    if not np.isfinite(z).all() or not np.isfinite(C).all():
        raise Rejected('numerical_failure','nonfinite Gaussian state')
    if np.linalg.eigvalsh(C).min() < -1e-12:
        raise Rejected('numerical_failure','negative covariance')
    dx=z[:3]*L-local.position_m;v=z[3:]*L/T
    R=local.modes;w=local.omega_rad_s
    X=R.T@(C[:3,:3]*L**2+np.outer(dx,dx))@R
    V=R.T@(C[3:,3:]*(L/T)**2+np.outer(v,v))@R
    raw=mass*(np.diag(V)+w*w*np.diag(X))/(2*hbar*w)-.5
    if np.min(raw)<-1e-5:
        raise Rejected('numerical_failure','unphysical negative excitation')
    n=np.maximum(raw,0)
    J=np.block([[np.zeros((3,3)),np.eye(3)],[-np.eye(3),np.zeros((3,3))]])
    quantum_half=hbar*T/(2*mass*L**2)
    if np.linalg.eigvalsh(C+1j*quantum_half*J).min() < -1e-7*quantum_half:
        raise Rejected('numerical_failure','covariance violates uncertainty relation')
    sym=np.linalg.eigvals(1j*J@C)
    Xcov=R.T@(C[:3,:3]*L**2)@R
    Vcov=R.T@(C[3:,3:]*(L/T)**2)@R
    nc=mass*(np.diag(Vcov)+w*w*np.diag(Xcov))/(2*hbar*w)-.5
    coherent=mass*((R.T@v)**2+w*w*(R.T@dx)**2)/(2*hbar*w)
    # B14: canonical metrics – residual energy and max transport error
    residual_energy_J = float(np.sum(hbar * w * n))
    # P2: max_transport_error_m should be max over trajectory vs nominal path, not just final error.
    # summarize sets fallback (final error) with flag; propagate overwrites with true max when history available.
    return {'n_modes':n.tolist(),'n_total':float(n.sum()),
            'coherent_n_modes':coherent.tolist(),'coherent_n_total':float(coherent.sum()),
            'covariance_n_modes':nc.tolist(),'covariance_n_total':float(nc.sum()),
            'mean_position_m':(z[:3]*L).tolist(),'mean_velocity_m_s':v.tolist(),
            'position_error_m':float(np.linalg.norm(z[:3]*L-target_m)),
            'well_target_error_m':float(np.linalg.norm(local.position_m-target_m)),
            'max_transport_error_m':float(np.linalg.norm(z[:3]*L-target_m)),  # fallback, flagged below
            'is_max_transport_error_fallback': True,
            'residual_motional_energy_J':residual_energy_J,
            'phase_space_displacement_m':dx.tolist(),
            'covariance_dimensionless':C.tolist(),'covariance_min_eigenvalue':float(np.linalg.eigvalsh(C).min()),
            'min_symplectic_eigenvalue_over_hbar_half':float(np.min(np.abs(sym))/(hbar*T/(2*mass*L**2)))}


def propagate(times_s, centers_m, hessians_J_m2, initial, final, mass, target_m,
             method='exact', hold='linear', return_trajectory=False,
             trajectory_decimate=400, path_samples_per_interval=8):
    """Распространяет гауссово состояние иона в движущейся гармонической яме.

    Механический лагранжиан имеет вид L = m |v|^2 / 2 - U(r, t).
    Уравнение Эйлера—Лагранжа даёт m r̈ = -∇U. После разложения энергии до
    второго порядка здесь решается уравнение осциллятора, меняющееся
    со временем. Ион не закрепляется в мгновенном минимуме; среднее положение
    и ковариация распространяются одним и тем же квадратичным гамильтонианом.

    Аналитическое распространение точно для принятой на каждом интервале
    постоянной кривизны и линейного движения центра; это не точное решение
    для произвольно меняющейся кривизны. Максимальное отставание оценивается
    по плотной сетке точек внутри временных интервалов.
    """
    ts = finite(times_s, 'time')
    centers = finite(centers_m, 'centers')
    hessians = finite(hessians_J_m2, 'H')
    if hold not in ('linear', 'zero_order'):
        raise ValueError('unknown hold rule')
    if (ts.ndim != 1 or len(ts) < 2 or centers.shape != (len(ts), 3)
            or hessians.shape != (len(ts), 3, 3) or np.any(np.diff(ts) <= 0)):
        raise ValueError('inconsistent dynamics grid')
    if not isinstance(path_samples_per_interval, (int, np.integer)) or path_samples_per_interval < 1:
        raise ValueError('path_samples_per_interval must be a positive integer')
    if not isinstance(trajectory_decimate, (int, np.integer)) or trajectory_decimate < 2:
        raise ValueError('trajectory_decimate must be at least 2')
    if not np.allclose(hessians, hessians.transpose(0, 2, 1), rtol=1e-12, atol=0):
        raise ValueError('Hessians must be symmetric')

    z, C = initial_state(initial, mass)
    K = hessians / mass * T * T
    c = centers / L
    tt = ts / T
    if np.min(np.linalg.eigvalsh(K)) <= 0:
        raise Rejected('weak_confinement', 'ROM Hessian not positive')

    # Store a dense path estimate for the transient lag cost and optional plots.
    path_times_s = [float(ts[0])]
    path_means_m = [(z[:3] * L).tolist()]
    path_centers_m = [centers[0].tolist()]

    def transition_matrix(curvature, eigvecs, frequencies, dt):
        cos_block = eigvecs @ np.diag(np.cos(frequencies * dt)) @ eigvecs.T
        sin_block = eigvecs @ np.diag(np.sin(frequencies * dt) / frequencies) @ eigvecs.T
        return np.block([[cos_block, sin_block],
                         [-curvature @ sin_block, cos_block]])

    if method == 'exact':
        for k, dt in enumerate(np.diff(tt)):
            km = K[k] if hold == 'zero_order' else (K[k] + K[k + 1]) / 2
            eig, modes = np.linalg.eigh(km)
            if eig.min() <= 0:
                raise Rejected('weak_confinement', 'nonpositive interval curvature')
            frequencies = np.sqrt(eig)
            velocity = (np.zeros(3) if hold == 'zero_order'
                        else (c[k + 1] - c[k]) / dt)
            eq0 = np.r_[c[k], velocity]
            z0, C0 = z.copy(), C.copy()

            # A zero-order well changes position instantaneously at a sample
            # boundary. Record the state on the new side of that jump as well.
            if k > 0 and hold == 'zero_order':
                path_times_s.append(float(ts[k]))
                path_means_m.append((z0[:3] * L).tolist())
                path_centers_m.append(centers[k].tolist())

            for j in range(1, path_samples_per_interval + 1):
                fraction = j / path_samples_per_interval
                sub_dt = fraction * dt
                Ssub = transition_matrix(km, modes, frequencies, sub_dt)
                if hold == 'linear':
                    center_sub = c[k] + fraction * (c[k + 1] - c[k])
                    equilibrium_sub = np.r_[center_sub, velocity]
                else:
                    center_sub = c[k]
                    equilibrium_sub = np.r_[center_sub, np.zeros(3)]
                zsub = equilibrium_sub + Ssub @ (z0 - eq0)
                path_times_s.append(float(ts[k] + fraction * (ts[k + 1] - ts[k])))
                path_means_m.append((zsub[:3] * L).tolist())
                path_centers_m.append((center_sub * L).tolist())

            S = transition_matrix(km, modes, frequencies, dt)
            if hold == 'linear':
                eq1 = np.r_[c[k + 1], velocity]
            else:
                eq1 = np.r_[c[k], np.zeros(3)]
            z = eq1 + S @ (z0 - eq0)
            C = S @ C0 @ S.T

        if hold == 'zero_order':
            # The final potential is the last sample, after the final ZOH jump.
            path_times_s.append(float(ts[-1]))
            path_means_m.append((z[:3] * L).tolist())
            path_centers_m.append(centers[-1].tolist())

    elif method == 'dop853' and hold == 'zero_order':
        for k, dt in enumerate(np.diff(tt)):
            if k > 0:
                path_times_s.append(float(ts[k]))
                path_means_m.append((z[:3] * L).tolist())
                path_centers_m.append(centers[k].tolist())
            A = np.block([[np.zeros((3, 3)), np.eye(3)],
                          [-K[k], np.zeros((3, 3))]])
            b = np.r_[np.zeros(3), K[k] @ c[k]]

            def rhs(tau, y):
                covariance = y[6:].reshape(6, 6)
                return np.r_[A @ y[:6] + b,
                             (A @ covariance + covariance @ A.T).ravel()]

            sol = solve_ivp(rhs, (0, dt), np.r_[z, C.ravel()], method='DOP853',
                            rtol=2e-10, atol=2e-13, dense_output=True)
            if not sol.success:
                raise Rejected('numerical_failure', sol.message)
            sample_rel = np.linspace(0, dt, path_samples_per_interval + 1)[1:]
            sample_states = sol.sol(sample_rel)
            for sub_tau, state in zip(sample_rel, sample_states.T):
                path_times_s.append(float((tt[k] + sub_tau) * T))
                path_means_m.append((state[:3] * L).tolist())
                path_centers_m.append(centers[k].tolist())
            z = sol.y[:6, -1]
            C = sol.y[6:, -1].reshape(6, 6)
        path_times_s.append(float(ts[-1]))
        path_means_m.append((z[:3] * L).tolist())
        path_centers_m.append(centers[-1].tolist())

    elif method == 'dop853':
        def rhs(tau, y):
            k = min(max(np.searchsorted(tt, tau) - 1, 0), len(tt) - 2)
            fraction = np.clip((tau - tt[k]) / (tt[k + 1] - tt[k]), 0, 1)
            km = (1 - fraction) * K[k] + fraction * K[k + 1]
            rm = (1 - fraction) * c[k] + fraction * c[k + 1]
            A = np.block([[np.zeros((3, 3)), np.eye(3)],
                          [-km, np.zeros((3, 3))]])
            state = y[:6]
            covariance = y[6:].reshape(6, 6)
            return np.r_[A @ state + np.r_[np.zeros(3), km @ rm],
                         (A @ covariance + covariance @ A.T).ravel()]

        sol = solve_ivp(rhs, (tt[0], tt[-1]), np.r_[z, C.ravel()],
                        method='DOP853', rtol=2e-10, atol=2e-13,
                        max_step=float(np.min(np.diff(tt))), dense_output=True)
        if not sol.success:
            raise Rejected('numerical_failure', sol.message)
        z = sol.y[:6, -1]
        C = sol.y[6:, -1].reshape(6, 6)
        chunks = [np.linspace(a, b, path_samples_per_interval + 1)[:-1]
                  for a, b in zip(tt[:-1], tt[1:])]
        sample_tau = np.concatenate(chunks + [tt[-1:]])
        sample_states = sol.sol(sample_tau)
        sample_centers = np.column_stack([
            np.interp(sample_tau, tt, c[:, axis]) for axis in range(3)
        ])
        path_times_s = (sample_tau * T).tolist()
        path_means_m = (sample_states[:3].T * L).tolist()
        path_centers_m = (sample_centers * L).tolist()
    else:
        raise ValueError('unknown propagator')

    summary = summarize(z, C, final, mass, target_m)
    path_times = np.asarray(path_times_s, dtype=float)
    path_means = np.asarray(path_means_m, dtype=float)
    path_centers = np.asarray(path_centers_m, dtype=float)
    if path_means.shape != path_centers.shape or path_means.shape[1:] != (3,):
        raise Rejected('numerical_failure', 'inconsistent within-step path samples')
    lag = np.linalg.norm(path_means - path_centers, axis=1)
    if not np.isfinite(lag).all():
        raise Rejected('numerical_failure', 'nonfinite transient ion-well lag')
    summary.update({
        'max_transport_error_m': float(np.max(lag)),
        'max_ion_well_lag_m': float(np.max(lag)),
        'mean_ion_well_lag_m': float(np.sum(0.5 * (lag[1:] + lag[:-1]) * np.diff(path_times))
                                      / (ts[-1] - ts[0])),
        'max_transport_error_vs_well_m': float(np.max(lag)),
        'is_max_transport_error_fallback': False,
        'path_samples_per_interval': int(path_samples_per_interval),
    })

    if return_trajectory:
        if len(path_times_s) > trajectory_decimate:
            idx = np.linspace(0, len(path_times_s) - 1, trajectory_decimate, dtype=int)
            path_times_s = [path_times_s[i] for i in idx]
            path_means_m = [path_means_m[i] for i in idx]
            path_centers_m = [path_centers_m[i] for i in idx]
        summary['trajectory'] = {
            'times_s': path_times_s,
            'mean_positions_m': path_means_m,
            'well_centers_m': path_centers_m,
            'well_times_s': path_times_s,
        }
    return summary

def validate_full_field_centroid(trap,times_s,voltages,initial,target_m,stray=None,frequency_scale=1., return_trajectory=False, trajectory_decimate=400):
    """Nonlinear pseudopotential force; reports coherent excitation only, not total n."""
    if not np.isfinite(frequency_scale) or frequency_scale<=0:
        raise ValueError('positive frequency scale required')
    t=finite(times_s,'time')/T;V=finite(voltages,'voltages');z0=np.r_[initial.position_m/L,np.zeros(3)]
    def rhs(tau,z):
        v=np.array([np.interp(tau,t,V[:,j]) for j in range(V.shape[1])])
        a=-frequency_scale**2*trap.energy_derivative(z[:3]*L,v,1,stray)/trap.mass*T*T/L
        return np.r_[z[3:],a]
    sol=solve_ivp(rhs,(t[0],t[-1]),z0,method='DOP853',rtol=2e-9,atol=1e-10,
                  max_step=min(float(np.min(np.diff(t))),np.pi/(np.max(initial.omega_rad_s)*T)))
    if not sol.success: raise Rejected('numerical_failure',sol.message)
    final=trap.local(V[-1],target_m,stray);z=sol.y[:,-1]
    omega=final.omega_rad_s*frequency_scale
    dx=final.modes.T@(z[:3]*L-final.position_m);vv=final.modes.T@(z[3:]*L/T)
    n=trap.mass*(vv*vv+omega**2*dx*dx)/(2*hbar*omega)
    out = {'coherent_n_modes':n.tolist(),'coherent_n_total':float(n.sum()),
            'mean_position_m':(z[:3]*L).tolist(),'mean_velocity_m_s':(z[3:]*L/T).tolist(),
            'position_error_m':float(np.linalg.norm(z[:3]*L-target_m)),
            'force_evaluations':sol.nfev,'model':'full Sion pseudopotential centroid; NOT full quantum n'}
    if return_trajectory:
        # sol.t in scaled units, sol.y [6, N]
        times_si = (sol.t * T).tolist()
        # decimate
        if len(times_si) > trajectory_decimate:
            idx = np.linspace(0, len(times_si)-1, trajectory_decimate, dtype=int)
            times_si = [times_si[i] for i in idx]
            traj = (sol.y[:3, idx].T * L).tolist()
        else:
            traj = (sol.y[:3].T * L).tolist()
        out['trajectory'] = {'times_s': times_si, 'mean_positions_m': traj}
    return out
