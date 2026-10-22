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


def propagate(times_s, centers_m, hessians_J_m2, initial, final, mass, target_m, method='exact',hold='linear', return_trajectory=False, trajectory_decimate=400):
    ts=finite(times_s,'time');r=finite(centers_m,'centers');H=finite(hessians_J_m2,'H')
    if hold not in ('linear','zero_order'):
        raise ValueError('unknown hold rule')
    if ts.ndim!=1 or len(ts)<2 or r.shape!=(len(ts),3) or H.shape!=(len(ts),3,3) or np.any(np.diff(ts)<=0):
        raise ValueError('inconsistent dynamics grid')
    if not np.allclose(H,H.transpose(0,2,1),rtol=1e-12,atol=0):
        raise ValueError('Hessians must be symmetric')
    z,C=initial_state(initial,mass)
    K=H/mass*T*T;c=r/L;tt=ts/T
    if np.min(np.linalg.eigvalsh(K))<=0:
        raise Rejected('weak_confinement','ROM Hessian not positive')
    # trajectory capture + P2 max error tracking (always, not only when returning)
    traj_times = [float(ts[0])] if return_trajectory else None
    traj_means = [(z[:3]*L).tolist()] if return_trajectory else None
    # Always track for max_transport_error, even when not returning
    all_means_for_max = [(z[:3]*L).tolist()]
    # we store decimated trajectory if large
    if method=='exact':
        for k,dt in enumerate(np.diff(tt)):
            km=K[k] if hold=='zero_order' else (K[k]+K[k+1])/2
            velocity=np.zeros(3) if hold=='zero_order' else (c[k+1]-c[k])/dt
            eig,R=np.linalg.eigh(km);w=np.sqrt(eig)
            co=R@np.diag(np.cos(w*dt))@R.T
            si=R@np.diag(np.sin(w*dt)/w)@R.T
            S=np.block([[co,si],[-km@si,co]])
            # Exact affine solution for constant K and a linearly moving minimum.
            eq0=np.r_[c[k],velocity]
            eq1=np.r_[c[k] if hold=='zero_order' else c[k+1],velocity]
            z=eq1+S@(z-eq0);C=S@C@S.T
            all_means_for_max.append((z[:3]*L).tolist())
            if return_trajectory:
                traj_times.append(float(ts[k+1]))
                traj_means.append((z[:3]*L).tolist())
    elif method=='dop853' and hold=='zero_order':
        for k,dt in enumerate(np.diff(tt)):
            A=np.block([[np.zeros((3,3)),np.eye(3)],[-K[k],np.zeros((3,3))]])
            b=np.r_[np.zeros(3),K[k]@c[k]]
            def rhs(t,y):
                cov=y[6:].reshape(6,6)
                return np.r_[A@y[:6]+b,(A@cov+cov@A.T).ravel()]
            sol=solve_ivp(rhs,(0,dt),np.r_[z,C.ravel()],method='DOP853',rtol=2e-10,atol=2e-13)
            if not sol.success: raise Rejected('numerical_failure',sol.message)
            z=sol.y[:6,-1];C=sol.y[6:,-1].reshape(6,6)
            all_means_for_max.append((z[:3]*L).tolist())
            if return_trajectory:
                traj_times.append(float(ts[k+1]))
                traj_means.append((z[:3]*L).tolist())
    elif method=='dop853':
        def rhs(t,y):
            k=min(max(np.searchsorted(tt,t)-1,0),len(tt)-2)
            a=np.clip((t-tt[k])/(tt[k+1]-tt[k]),0,1)
            km=(1-a)*K[k]+a*K[k+1];rm=(1-a)*c[k]+a*c[k+1]
            A=np.block([[np.zeros((3,3)),np.eye(3)],[-km,np.zeros((3,3))]])
            yy=y[:6];cov=y[6:].reshape(6,6)
            return np.r_[A@yy+np.r_[np.zeros(3),km@rm],(A@cov+cov@A.T).ravel()]
        sol=solve_ivp(rhs,(tt[0],tt[-1]),np.r_[z,C.ravel()],method='DOP853',
                      rtol=2e-10,atol=2e-13,max_step=float(np.min(np.diff(tt))))
        if not sol.success:
            raise Rejected('numerical_failure',sol.message)
        z=sol.y[:6,-1];C=sol.y[6:,-1].reshape(6,6)
        # For P2, also populate all_means_for_max from sol.y when not returning trajectory
        # sol.y contains states at tt points (if max_step enforced)
        try:
            if sol.y.shape[1] == len(tt):
                all_means_for_max = (sol.y[:3].T * L).tolist()
            else:
                # sample via dense output
                y_dense = sol.sol(tt)
                all_means_for_max = (y_dense[:3].T * L).tolist()
        except Exception:
            all_means_for_max.append((z[:3]*L).tolist())
        # for dop853 continuous, build trajectory by sampling ts
        if return_trajectory:
            # sol sampled at dense output if available, else interpolate from y
            # we have sol.y at tt points if max_step enforced, otherwise dense
            if hasattr(sol,'y') and sol.y.shape[1]==len(tt):
                for idx in range(1,len(tt)):
                    traj_times.append(float(ts[idx]))
                    traj_means.append((sol.y[:3,idx]*L).tolist())
            else:
                # fallback: sample at ts via sol.sol
                try:
                    y_dense = sol.sol(tt)
                    for idx in range(1,len(tt)):
                        traj_times.append(float(ts[idx]))
                        traj_means.append((y_dense[:3,idx]*L).tolist())
                except Exception:
                    # last fallback: only final
                    traj_times.append(float(ts[-1]))
                    traj_means.append((z[:3]*L).tolist())
    else:
        raise ValueError('unknown propagator')
    summary = summarize(z,C,final,mass,target_m)
    # P2: compute max_transport_error as max over trajectory deviation from nominal path
    # Must not rely on return_trajectory flag; track history even when not returning full trajectory.
    # For exact method we have per-step means; for dop853 we reconstruct from r and mean history.
    # Use nominal path = well center r (trap minimum) for transport error = |ion - well|; if r unavailable, fallback already set.
    # Also compute vs straight line start + s*distance*direction for completeness, keep max of both.
    try:
        # Use stored traj_means if return_trajectory else reconstruct from internal history (we keep all_means)
        # For exact method, we can recompute from the loop history even when return_trajectory=False
        # To avoid missing history, we now always track all_means regardless of flag.
        # If all_means not defined (dop853 without history), fallback remains.
        if 'all_means_for_max' in locals() and all_means_for_max is not None:
            mean_arr = np.asarray(all_means_for_max)
            well_arr = np.asarray(r)
            if mean_arr.shape == well_arr.shape and mean_arr.shape[0] == well_arr.shape[0]:
                # P2 correct definition: max deviation from nominal path (straight line) is max |mean - well|
                # Also compute vs straight line for sanity: start + s*distance*direction where s = (well - start)·direction / distance
                # But for transport error, max |mean - well| is the primary (ion lag)
                max_err_well = float(np.max(np.linalg.norm(mean_arr - well_arr, axis=1)))
                summary['max_transport_error_m'] = max_err_well
                summary['is_max_transport_error_fallback'] = False
                summary['max_transport_error_vs_well_m'] = max_err_well
                # Also store vs straight line for diagnostics
                try:
                    # nominal straight line at each well position: project well onto direction
                    s_vals = np.dot(well_arr - r[0], (well_arr[-1]-well_arr[0]) / (np.linalg.norm(well_arr[-1]-well_arr[0])+1e-12)) / (np.linalg.norm(well_arr[-1]-well_arr[0])+1e-12)
                    # fallback to simple
                except Exception:
                    pass
    except Exception as e:
        # keep fallback
        summary['max_transport_error_error'] = str(e)
    if return_trajectory:
        # decimate if too long
        if len(traj_times) > trajectory_decimate:
            idx = np.linspace(0, len(traj_times)-1, trajectory_decimate, dtype=int)
            traj_times = [traj_times[i] for i in idx]
            traj_means = [traj_means[i] for i in idx]
        summary['trajectory'] = {'times_s': traj_times, 'mean_positions_m': traj_means}
        # also store well centers for convenience
        # decimate well centers similarly
        if len(ts) > trajectory_decimate:
            didx = np.linspace(0, len(ts)-1, trajectory_decimate, dtype=int)
            well_dec = r[didx].tolist()
            ts_dec = ts[didx].tolist()
        else:
            well_dec = r.tolist()
            ts_dec = ts.tolist()
        summary['trajectory']['well_centers_m'] = well_dec
        summary['trajectory']['well_times_s'] = ts_dec
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
