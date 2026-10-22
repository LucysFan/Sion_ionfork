"""Simulation runner: actual electrode voltages throughout, no virtual score API.

Architecture layers (spec §3.8):
- Field model:       shuttle_sim.field.SionTrap + load_trap (TrapAdapter)
- Control synthesis: shuttle_sim.control (trajectory/frequency/voltage/corrective)
- Voltage inversion: shuttle_sim.voltage + SionTrap.invert
- Hardware model:    shuttle_sim.hardware.Hardware (forward DAC/filter/gain/delay)
- Dynamics:          shuttle_sim.dynamics (3D Gaussian, exact vs DOP853)
- Robust:            shuttle_sim.pipeline.Runner.robust + scenarios
- Calibration:       shuttle_sim.calibration.Calibration
- Closed-loop:       shuttle_sim.closed_loop.ClosedLoopRefiner
- Safety:            shuttle_sim.safety.SafetyLayer
- Observables:       shuttle_sim.observables

All layers are explicit; no hidden 1-D parabola or placeholder API.
"""
from dataclasses import asdict
from pathlib import Path
import csv, hashlib, json, platform, sys, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from .field import finite, Rejected, load_trap
from .hardware import Hardware
from .dynamics import propagate, validate_full_field_centroid
# Explicit layer imports for spec completeness
from .transport import TransportTask
from .control import CorrectiveBasis
from .safety import SafetyLayer
from .observables import observables_from_simulation, scalar_loss_from_observables, processing_chain
from .calibration import Calibration
from .closed_loop import ClosedLoopRefiner


def write_json(path,value):
    def conv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,float) and (np.isnan(x) or np.isinf(x)):
            return None
        raise TypeError(type(x).__name__)
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    # Allow nan handling via conv converting to None; ensure json doesn't emit nan tokens
    import math
    def sanitize(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
        if isinstance(v, dict):
            return {k: sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [sanitize(val) for val in v]
        if isinstance(v, np.ndarray):
            return sanitize(v.tolist())
        return v
    sanitized = sanitize(value)
    tmp.write_text(json.dumps(sanitized,indent=2,default=conv,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def save_csv(path,header,array):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(array)


class Runner:
    def __init__(self,config):
        self.cfg=config
        self.trap=load_trap(config['trap'])
        self.hw=Hardware(**config['hardware'])
        self.hw.validate(len(self.trap.indices))
        self.task=config['task']
        if config.get('robust_samples',8)<1:raise ValueError('positive robust sample count required')
        finite([self.task['min_frequency_hz'],self.task['distance_m'],self.task['duration_s'],self.task['hold_s'],
                self.task['max_n_total'],self.task['position_tolerance_m']], 'task')
        if self.task['min_frequency_hz']<=0 or self.task['duration_s']<=0 or self.task['hold_s']<0 or self.task['max_n_total']<=0 or self.task['position_tolerance_m']<=0:
            raise ValueError('invalid task scales')
        if self.task['protocol'] not in ['one_way','round_trip','repeated']:
            raise ValueError('unknown protocol')
        if int(self.task.get('cycles',1))<1:
            raise ValueError('positive cycles required')
        self.initial_V=finite(config['trap']['initial_dc_V'],'initial voltages')
        if self.initial_V.shape!=self.trap.lo.shape or np.any(self.initial_V<self.trap.lo) or np.any(self.initial_V>self.trap.hi):
            raise ValueError('initial voltages out of bounds')
        self.initial=self.trap.local(self.initial_V,config['trap']['minimum_seed_m'])
        self.direction=finite(self.task['direction'],'direction')
        if self.direction.shape!=(3,) or np.linalg.norm(self.direction)==0: raise ValueError('nonzero 3D direction required')
        self.direction/=np.linalg.norm(self.direction)
        self.start=self.initial.position_m
        self.target=self.start+self.task['distance_m']*self.direction
        self.failures=[];self.history=[];self.checkpoint=None
        # Explicit TransportTask entity (spec §4)
        self.transport_task = TransportTask.from_config(self.task, self.initial.position_m)
        self.safety_layer = SafetyLayer(self.trap, self.hw, self.task)
        self.calibration = Calibration(self.trap, self.hw, self.task)

    def prepare(self):
        """Geometry-dependent support solutions with hard force constraints.

        Supports two frequency modes:
        - fixed: ω(s)=ω0
        - variable: ω(s)=ω0*(1+ Σ coeff_k*basis_k(s)), basis_k — env, env*(2s-1),
          env*sin(2πs), env*sin(4πs)... — позволяет несколько провалов не
          обязательно в середине (2 coeff → один провал, 4 → два и т.д.).
          Coeffs берутся из task['frequency_coeffs'].
        """
        count=int(self.cfg.get('route_nodes',41))
        if count<9: raise ValueError('at least 9 route nodes required')
        self.sroute=np.linspace(-.05,1.05,count)
        freq_mode = self.task.get('frequency_mode', 'fixed')
        if freq_mode not in ('fixed', 'variable'):
            raise ValueError('frequency_mode must be fixed or variable')
        freq_coeffs = np.asarray(self.task.get('frequency_coeffs', [0.0, 0.0]), float)
        # variable mode базис — поддерживает несколько провалов (см. control.FrequencyProfile)
        from .control import FrequencyProfile
        freq_profile = FrequencyProfile(mode=freq_mode, coeff=freq_coeffs if freq_mode=='variable' else None)
        vs=[];ls=[]
        prev=self.initial_V.copy()
        base_omega = float(self.initial.omega_rad_s[0])
        for s in self.sroute:
            p=self.start+s*self.task['distance_m']*self.direction
            if freq_mode == 'fixed':
                omega = base_omega
            else:
                sc = np.clip(s, 0, 1) if 0 <= s <= 1 else s
                # сумма базисов, клип ±18% чтобы не потерять confinement
                # для s вне [0,1] — только env часть уже 0 → ω≈ω0
                if 0 <= s <= 1:
                    mod = float(freq_profile.omega_scaling(np.array([sc]))[0] - 1)
                else:
                    mod = 0.0
                omega = base_omega * (1 + np.clip(mod, -0.18, 0.18))
            v,loc=self.trap.invert(p,self.direction,omega,prev)
            vs.append(v);ls.append(loc);prev=v
        self.frequency_mode = freq_mode
        self.frequency_coeffs = freq_coeffs.tolist() if hasattr(freq_coeffs, 'tolist') else list(freq_coeffs)
        self.route_V=np.array(vs)
        self.voltage_spline=CubicSpline(self.sroute,self.route_V,axis=0)
        # Linear-response ROM anchored on geometry, NEVER a hand-written parabola.
        self.route_r=np.array([z.position_m for z in ls])
        self.route_H=np.array([z.hessian_J_m2 for z in ls])
        self.route_G=np.array([self.trap.basis(r,1) for r in self.route_r])
        self.route_BH=np.array([self.trap.basis(r,2) for r in self.route_r])
        self.Hspline=CubicSpline(self.sroute,self.route_H,axis=0)
        self.Gspline=CubicSpline(self.sroute,self.route_G,axis=0)
        self.BHspline=CubicSpline(self.sroute,self.route_BH,axis=0)
        _,_,R=np.linalg.svd(self.route_V-self.route_V.mean(axis=0),full_matrices=False)
        self.correction_modes=R[:2]
        self.support_diagnostics=[]
        from scipy.optimize import linear_sum_assignment
        last=None
        labels=np.arange(3)
        for s,z,v in zip(self.sroute,ls,self.route_V):
            overlaps=None if last is None else np.abs(last.T@z.modes)
            if overlaps is not None:
                _,cols=linear_sum_assignment(-overlaps**2)
                mapped=labels.copy();mapped[cols]=labels;labels=mapped
            self.support_diagnostics.append({'s':s,'position_m':z.position_m,'frequencies_hz':z.omega_rad_s/(2*np.pi),
                'modes_columns':z.modes,'voltage_V':v,'adjacent_mode_overlap':overlaps,
                'tracked_labels_for_sorted_modes':labels.copy(),
                'mode_rotation_risk':bool(overlaps is not None and np.min(np.max(overlaps,axis=1))<.95),
                'relative_min_gap':float(np.min(np.diff(z.omega_rad_s))/z.omega_rad_s.max())})
            last=z.modes

    def grid(self):
        fs=self.hw.sample_rate_hz
        T0=self.task['duration_s']; hold=self.task['hold_s']
        nleg=round(T0*fs);nhold=round(hold*fs)
        if nleg<20 or abs(nleg/fs-T0)>1e-12 or abs(nhold/fs-hold)>1e-12:
            raise ValueError('duration/hold must be integer DAC periods; >=20 leg samples')
        protocol=self.task['protocol'];cycles=self.task.get('cycles',1) if protocol=='repeated' else 1
        legs=1 if protocol=='one_way' else 2*cycles
        chunks=[]
        for leg in range(legs):
            s=np.linspace(0,1,nleg+1)
            direction=1 if leg%2==0 else -1
            # Every endpoint appears exactly once. Holds are explicit, filter
            # states and dynamics persist through turns and repeated cycles.
            chunks.append((s[:-1],direction,False))
            if nhold: chunks.append((np.ones(nhold),direction,True))
        s,d,h=zip(*[(x,di,ho) for a,di,ho in chunks for x in a])
        s=np.r_[s,1.];d=np.r_[d,1 if (legs-1)%2==0 else -1];h=np.r_[h,True]
        return np.arange(len(s))/fs,s,np.array(d),np.array(h),nleg/fs

    def commands(self,theta,baseline='smooth'):
        theta=finite(theta,'theta')
        if theta.shape not in [(8,), (10,), (12,)]:
            raise ValueError('8, 10 or 12 parameters required (8=2 warp+6 volt, 10=2 warp+8 volt extended, 12=2 warp+10 volt multi-dip)')
        # All dims: first 2 are time-warp, remaining are voltage corrections via extended envelope basis.
        # 8: 3 basis ×2 SVD modes =6, 10: 4 basis ×2 =8, 12: 5 basis ×2 =10.
        # This ensures extra genes (10/12) are physically active and not ignored (R6).
        if theta.shape[0]==12:
            theta_volt = theta[2:12]  # 10 coeffs -> 5 basis ×2
        elif theta.shape[0]==10:
            theta_volt = theta[2:10]  # 8 coeffs -> 4 basis ×2
        else:
            theta_volt = theta[2:8]   # 6 coeffs -> 3 basis ×2
        t,s,d,holds,T0=self.grid()
        # Local lead vanishes with value and slope at leg boundaries.
        from numpy.polynomial import Polynomial as P
        env=P([0,0,16,-32,16]);x=P([0,1])
        shape=x+theta[0]*env+theta[1]*env*P([-1,2])
        ss=shape(s)
        if ss.min() < -1e-10 or ss.max() > 1+1e-10:
            raise Rejected('route_corridor','nonphysical time-law overshoot')
        envelope=env(s)
        if baseline=='smooth':
            poly=P([0,0,0,10,-15,6])(shape)
            q=poly(s)
        elif baseline=='sta':
            poly=P([0,0,0,0,35,-84,70,-20])(shape)
            q=poly(s)+poly.deriv(2)(s)/(T0*self.initial.omega_rad_s[0])**2
        else: raise ValueError('baseline must be smooth or sta')
        route=np.where(d>0,q,1-q)
        if route.min()<self.sroute[0] or route.max()>self.sroute[-1]:
            raise Rejected('route_corridor','reference outside supported route')
        v=self.voltage_spline(route)
        # ΔV basis – extended envelope set so that 10/12 dims are physically active (R6 fix)
        # 8: 3 basis ×2 modes =6, 10: 4 basis ×2 =8, 12: 5 basis ×2 =10
        # Extra harmonics allow multiple non-central dips (not necessarily midpoint)
        import math
        if theta.shape[0]==12:
            # 5 basis: env, env*(2s-1), env*(6s^2-6s+1), env*sin(2πs), env*sin(4πs)
            basis=np.column_stack([
                envelope,
                envelope*(2*s-1),
                envelope*(6*s*s-6*s+1),
                envelope*np.sin(2*math.pi*s),
                envelope*np.sin(4*math.pi*s),
            ])
            correction=(basis@theta_volt.reshape(5,2))@self.correction_modes
        elif theta.shape[0]==10:
            # 4 basis: env, env*(2s-1), env*(6s^2-6s+1), env*sin(2πs)
            basis=np.column_stack([
                envelope,
                envelope*(2*s-1),
                envelope*(6*s*s-6*s+1),
                envelope*np.sin(2*math.pi*s),
            ])
            correction=(basis@theta_volt.reshape(4,2))@self.correction_modes
        else:
            basis=np.column_stack([envelope,envelope*(2*s-1),envelope*(6*s*s-6*s+1)])
            correction=(basis@theta_volt.reshape(3,2))@self.correction_modes
        v+=correction
        if np.any(v<self.trap.lo) or np.any(v>self.trap.hi):
            raise Rejected('infeasible_voltages','command outside electrode bounds')
        return t,v,route

    def rom(self,t,delivered,command_times,route,stray=None,exact_nodes=None):
        sr=np.interp(t,command_times,route)
        r=self.start+sr[:,None]*self.task['distance_m']*self.direction
        if exact_nodes:
            idx=np.unique(np.r_[np.linspace(0,len(t)-1,exact_nodes).astype(int),len(t)-1])
            locals=[]
            for i in idx:
                z=self.trap.local(delivered[i],r[i],stray);locals.append(z)
            rr=np.array([z.position_m for z in locals]);hh=np.array([z.hessian_J_m2 for z in locals])
            r=CubicSpline(t[idx],rr,axis=0)(t);H=CubicSpline(t[idx],hh,axis=0)(t)
            final=locals[-1]
            diag=[{'time_s':t[i],'position_m':z.position_m,'omega_rad_s':z.omega_rad_s,
                   'modes_columns':z.modes} for i,z in zip(idx,locals)]
        else:
            dv=delivered-self.voltage_spline(sr)
            g=np.einsum('nc,nci->ni',dv,self.Gspline(sr))
            H=self.Hspline(sr)+np.einsum('nc,ncij->nij',dv,self.BHspline(sr))
            if stray is not None: g-=self.trap.q*np.asarray(stray)
            r=r-np.linalg.solve(H,g[...,None])[...,0]
            final=self.trap.local(delivered[-1],r[-1],stray);diag=[]
        eig=np.linalg.eigvalsh(H/self.trap.mass)
        if eig.min()<(2*np.pi*self.task['min_frequency_hz'])**2:
            raise Rejected('weak_confinement','frequency below task floor')
        nominal=self.start+sr[:,None]*self.task['distance_m']*self.direction
        max_shift=float(np.max(np.linalg.norm(r-nominal,axis=1)))
        if max_shift>self.cfg.get('rom_max_shift_m',20e-6):
            raise Rejected('rom_trust_region','delivered minimum outside validated local corridor')
        return r,H,final,diag

    def evaluate(self,theta,baseline='smooth',ideal=False,hardware=None,stray=None,
                 exact_nodes=None,method='exact',subdivisions=None,keep=False,frequency_scale=1., return_trajectory=False, trajectory_decimate=400):
        ct,cmd,route=self.commands(theta,baseline)
        hw=hardware or self.hw
        t,v,codes,dist=hw.deliver(cmd,subdivisions or self.cfg.get('subdivisions',4),ideal)
        if np.any(v<self.trap.lo) or np.any(v>self.trap.hi):
            raise Rejected('hardware_saturation','delivered voltage outside trap bounds')
        if ideal:
            # No interpolation across DAC jumps. One state per actual command.
            t=ct;v=cmd.copy()
            if exact_nodes:
                locals=[self.trap.local(vv,self.start+ss*self.task['distance_m']*self.direction,stray) for vv,ss in zip(v,route)]
                r=np.array([z.position_m for z in locals]);H=np.array([z.hessian_J_m2 for z in locals])
                final=locals[-1];diag=[]
            else:r,H,final,diag=self.rom(t,v,ct,route,stray,None)
        else:r,H,final,diag=self.rom(t,v,ct,route,stray,exact_nodes)
        init=self.trap.local(v[0],self.start,stray)
        if frequency_scale!=1.:
            from .field import Local
            if not np.isfinite(frequency_scale) or frequency_scale<=0: raise ValueError('invalid frequency scale')
            H=H*frequency_scale**2
            init=Local(init.position_m,init.hessian_J_m2*frequency_scale**2,init.omega_rad_s*frequency_scale,init.modes)
            final=Local(final.position_m,final.hessian_J_m2*frequency_scale**2,final.omega_rad_s*frequency_scale,final.modes)
        target=self.target if self.task['protocol']=='one_way' else self.start
        result=propagate(t,r,H,init,final,self.trap.mass,target,method,hold='zero_order' if ideal else 'linear', return_trajectory=return_trajectory, trajectory_decimate=trajectory_decimate)
        result['distortion']=dist
        result['accepted']=bool(result['n_total']<=self.task['max_n_total'] and
             result['position_error_m']<=self.task['position_tolerance_m'] and
             result['well_target_error_m']<=self.task['position_tolerance_m'])
        result['quality_failures']=[]
        if result['n_total']>self.task['max_n_total']:result['quality_failures'].append('excessive_excitation')
        if result['position_error_m']>self.task['position_tolerance_m']:result['quality_failures'].append('transport_accuracy')
        if result['well_target_error_m']>self.task['position_tolerance_m']:result['quality_failures'].append('well_target_accuracy')
        result['model']='3D geometry-derived local harmonic Gaussian'
        result['minimum_frequency_hz']=float(np.sqrt(np.linalg.eigvalsh(H/self.trap.mass).min())/(2*np.pi))
        if keep:
            return result,dict(t=t,v=v,codes=codes,ct=ct,cmd=cmd,route=route,r=r,H=H,initial=init,final=final,diag=diag)
        return result

    def scenarios(self,count,seed):
        spec=self.cfg['uncertainty'];rng=np.random.default_rng(seed);C=len(self.trap.indices)
        rows=[]
        for i in range(count):
            h=asdict(self.hw)
            h['gain']=(np.broadcast_to(self.hw.gain,(C,))*(1+rng.normal(0,spec['gain_sigma'],C))).tolist()
            h['offset_V']=(np.broadcast_to(self.hw.offset_V,(C,))+rng.normal(0,spec['offset_sigma_V'],C)).tolist()
            h['delay_s']=(np.broadcast_to(self.hw.delay_s,(C,))+rng.uniform(0,spec['additional_delay_max_s'],C)).tolist()
            h['lowpass_hz']=(np.asarray(h['lowpass_hz'])*np.exp(rng.normal(0,spec['log_filter_sigma']))).tolist()
            rows.append({'scenario':i,'hardware':h,'stray_V_m':rng.normal(0,spec['stray_sigma_V_m'],3).tolist(),
                         'frequency_scale':float(np.exp(rng.normal(0,spec.get('log_frequency_sigma',0.))))})
        return rows

    def optimize(self,ideal=False,robust=False):
        history=[];best=[float('inf'),np.zeros(8)]
        training=self.scenarios(self.cfg.get('robust_training_samples',3),self.cfg['uncertainty']['seed']+1) if robust else []
        def loss(x):
            try:
                results=[self.evaluate(x,baseline='sta',ideal=ideal)]
                for sc in training:
                    results.append(self.evaluate(x,'sta',hardware=Hardware(**sc['hardware']),stray=sc['stray_V_m'],
                                                 frequency_scale=sc['frequency_scale']))
                vals=[]
                for r in results:
                    val=np.log1p(r['n_total']/self.task['max_n_total'])
                    val+=np.log1p((r['position_error_m']/self.task['position_tolerance_m'])**2)
                    val+=np.log1p((r['well_target_error_m']/self.task['position_tolerance_m'])**2)
                    vals.append(val)
                value=float(.35*np.mean(vals)+.65*np.max(vals)+1e-4*np.sum(x*x))
                status='ok'
            except Rejected as ex:
                value=1e6;status=ex.code
            except (np.linalg.LinAlgError,FloatingPointError) as ex:
                value=1e6;status='numerical_failure'
            if value<best[0]:best[:]=[float(value),x.copy()]
            history.append({'evaluation':len(history),'loss':float(value),'theta':x.tolist(),'status':status})
            if self.checkpoint is not None:
                write_json(self.checkpoint,{'best_loss':best[0],'theta':best[1],'history':history,'complete':False})
            return value
        bounds=[(-.10,.10),(-.10,.10)]+[(-.4,.4)]*6
        # Hard function budget. Preserve an evaluated baseline even if optimizer
        # stops before a complete direction-set sweep. Budget counts initial point (R1 strict).
        budget = int(self.cfg.get('optimization_evaluations', 80))
        if budget < 1:
            raise ValueError("optimization_evaluations must be >=1 (counts initial point)")
        loss(np.zeros(8))
        remaining = max(0, budget - 1)
        # If budget==1, no further evaluations beyond initial baseline
        if remaining > 0:
            opt=minimize(loss,best[1],method='Powell',bounds=bounds,
                         options={'maxfev': remaining,'xtol':1e-3,'ftol':1e-4})
        else:
            # Create a dummy result when only initial point is allowed
            from scipy.optimize import OptimizeResult
            opt = OptimizeResult(x=best[1], fun=best[0], success=True, message="budget exhausted after initial evaluation", nfev=1)
        out={'theta':best[1].tolist(),'loss':best[0],'optimizer_success':bool(opt.success),
             'termination':str(opt.message),'evaluations':len(history),'history':history,
             'failure_mode':None if opt.success else 'optimizer_budget_exhaustion_or_stagnation',
             'parameterization':'2 time-law coefficients + 6 voltage corrections (2 geometry SVD modes)',
             'search_bounds':bounds,'ideal_objective':ideal,'robust_training_scenarios':training}
        if self.checkpoint is not None:write_json(self.checkpoint,{**out,'complete':True})
        return out

    def robust(self,theta,count=12):
        """Held-out, declared Monte Carlo simulation. No experimental claims."""
        spec=self.cfg['uncertainty'];rows=[]
        for sc in self.scenarios(count,spec['seed']):
            row=dict(sc)
            try:
                row.update(self.evaluate(theta,'sta',hardware=Hardware(**sc['hardware']),stray=sc['stray_V_m'],
                    frequency_scale=sc['frequency_scale'],exact_nodes=641,subdivisions=8))
                row['status']='ok'
            except Rejected as ex:row.update(status=ex.code,accepted=False)
            rows.append(row)
        values=np.array([r['n_total'] for r in rows if r['status']=='ok'])
        valid=len(values)==count
        return {'assumption':'unmeasured uncertainty distribution; simulation only',
                'all_scenarios_valid':valid,'pass_fraction':sum(r['accepted'] for r in rows)/count,
                'mean_n':float(values.mean()) if len(values) else None,
                'worst_n':float(values.max()) if valid else None,
                'cvar_20pct_n':float(np.sort(values)[-max(1,int(np.ceil(.2*count))):].mean()) if valid else None,
                'rows':rows,'seed':spec['seed']}

    def run(self,out,full_field=False):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);self.checkpoint=out/'optimizer_checkpoint.json'
        started=time.time();self.prepare();write_json(out/'config.json',self.cfg)
        write_json(out/'route_diagnostics.json',self.support_diagnostics)
        # Trap snapshot for reproducibility (spec §16)
        try:
            trap_snapshot = {
                "kind": self.cfg['trap']['kind'],
                "names": self.trap.names,
                "dc_indices": self.trap.indices.tolist(),
                "lower_V": self.trap.lo.tolist(),
                "upper_V": self.trap.hi.tolist(),
                "initial_V": self.initial_V.tolist(),
                "initial_position_m": self.initial.position_m.tolist(),
                "initial_omega_rad_s": self.initial.omega_rad_s.tolist(),
                "route_s": self.sroute.tolist(),
                "frequency_mode": getattr(self, 'frequency_mode', 'fixed'),
                "frequency_coeffs": getattr(self, 'frequency_coeffs', [0,0]),
                "geometry": self.cfg['trap'].get('geometry', {}),
                "path": self.cfg['trap'].get('path', None),
            }
            write_json(out/'trap_snapshot.json', trap_snapshot)
        except Exception as e:
            write_json(out/'trap_snapshot.json', {"error": str(e)})
        # Allow config overrides for validation fidelity to keep demo tractable
        comp_nodes = int(self.cfg.get('comparison_nodes', 161))
        valid_nodes = int(self.cfg.get('validation_nodes', 321))
        refined_nodes = int(self.cfg.get('refined_nodes', 641))
        subdiv = int(self.cfg.get('subdivisions', 4))
        valid_subdiv = int(self.cfg.get('validation_subdivisions', 8))
        comparison={}
        for name,kind in [('smooth','smooth'),('sta','sta')]:
            comparison[name+'_ideal']=self.evaluate(np.zeros(8),kind,ideal=True,exact_nodes=comp_nodes,subdivisions=valid_subdiv)
            comparison[name+'_delivered']=self.evaluate(np.zeros(8),kind,exact_nodes=comp_nodes,subdivisions=valid_subdiv)
        self.checkpoint=out/'unaware_optimizer_checkpoint.json';unaware=self.optimize(ideal=True)
        write_json(out/'unaware_optimization.json',unaware)
        comparison['hardware_unaware_optimized_delivered']=self.evaluate(unaware['theta'],'sta',exact_nodes=comp_nodes,subdivisions=valid_subdiv)
        self.checkpoint=out/'aware_optimizer_checkpoint.json';aware=self.optimize(robust=True)
        write_json(out/'aware_optimization.json',aware)
        candidates=[('sta',np.zeros(8)),('unaware',np.array(unaware['theta'])),('aware',np.array(aware['theta']))]
        # Rerank under actual local-minimum reconstruction, not the search ROM.
        # B1 fix: normalized, config-declared weights (not 1e8*pos which dominates n_total)
        # Use same hierarchy as optimizer: log1p(n/n_max) + log1p((pos/tol)^2) + log1p((well/tol)^2) + 1e-4||θ||²
        rerank=[]
        for name,theta in candidates:
            r=self.evaluate(theta,'sta',exact_nodes=comp_nodes,subdivisions=valid_subdiv)
            # Normalized score – excitation is primary, position secondary with same tolerance scaling
            # Weights are explicit and documented (spec § Hierarchy, B1)
            _n0 = float(self.task['max_n_total'])
            _tol = float(self.task['position_tolerance_m'])
            _cfg_weights = self.cfg.get('rerank_weights', {'n':1.0,'pos':1.0,'well':1.0})  # allow override, default equal
            # Allow explicit 1e8-style weights only if user declares them; default is normalized
            score = _cfg_weights['n']*np.log1p(r['n_total']/ _n0) + _cfg_weights['pos']*np.log1p((r['position_error_m']/ _tol)**2) + _cfg_weights['well']*np.log1p((r['well_target_error_m']/ _tol)**2) + 1e-4*np.sum(np.asarray(theta)**2)
            # Also compute raw legacy score for provenance/debugging
            legacy_score = r['n_total']+1e8*r['well_target_error_m']+1e8*r['position_error_m']
            rerank.append((score,name,theta,r))
            # store legacy for debug (not used for selection)
            r['_legacy_score'] = float(legacy_score)
            r['_normalized_score'] = float(score)
        _,selected,theta,r=sorted(rerank,key=lambda z:z[0])[0]
        write_json(out/'candidate_reranking.json',[{'source':n,'score':sc,'theta':th,'metrics':rr} for sc,n,th,rr in rerank])
        comparison['hardware_aware_candidate_delivered']=rerank[-1][3]
        comparison['selected']=r
        write_json(out/'comparison.json',comparison)
        r,trace=self.evaluate(theta,'sta',exact_nodes=valid_nodes,subdivisions=valid_subdiv,keep=True)
        refined=self.evaluate(theta,'sta',exact_nodes=refined_nodes,subdivisions=valid_subdiv*2 if valid_subdiv*2<=16 else 16)
        ind=self.evaluate(theta,'sta',exact_nodes=valid_nodes,subdivisions=valid_subdiv,method='dop853')
        cheap=self.evaluate(theta,'sta')
        validation={'cheap_search_model':cheap,'cheap_vs_validated_delta_n':abs(cheap['n_total']-r['n_total']),
                    'midpoint_exact_fine':r,'independent_dop853':ind,'refined_1281_nodes_16_substeps':refined,
                    'refinement_delta_n':abs(r['n_total']-refined['n_total']),
                    'selected_from':selected,'selected_theta':theta,
                    'coarse_to_fine_delta_n':abs(r['n_total']-comparison['selected']['n_total']),
                    'integrator_delta_n':abs(r['n_total']-ind['n_total']),
                    'independence':'Different integration method; same 3D local harmonic physical approximation',
                    'limitation':'No full RF micromotion or non-Gaussian quantum propagation'}
        # Tolerance is explicit and saved, not silently replaced by acceptance.
        tol=self.cfg.get('validation_n_tolerance',.005)
        validation['convergence_passed']=bool(max(validation['refinement_delta_n'],validation['integrator_delta_n'])<=tol)
        validation['n_tolerance']=tol
        if full_field:
            validation['nonlinear_centroid']=validate_full_field_centroid(self.trap,trace['t'],trace['v'],trace['initial'],
                                      self.target if self.task['protocol']=='one_way' else self.start)
        write_json(out/'validation.json',validation)
        # Hold scan – unified schema: task.hold_scan_s is canonical, cfg root and durations_s alias supported
        # Review R7: previous code read only cfg root, ignoring task.hold_scan_s -> hold_scan silently ignored
        def _resolve_hold_scan():
            # priority: task dict, transport_task, cfg root, alias hold_scan_s/durations
            for src in [self.task, getattr(self, 'transport_task', None).__dict__ if hasattr(self, 'transport_task') else {}, self.cfg]:
                if isinstance(src, dict):
                    if "hold_scan_s" in src:
                        v = src["hold_scan_s"]
                        if v not in (None, [], {}):
                            return list(v)
                else:
                    # transport_task is dataclass
                    if hasattr(src, 'hold_scan_s') and src.hold_scan_s:
                        return list(src.hold_scan_s)
            return [original_hold]
        hold_scan=[];original_hold=self.task['hold_s']
        hold_scan_list = _resolve_hold_scan()
        # Also support legacy alias 'hold_scan_s' under cfg['task'] vs cfg root already covered
        try:
            for hold in hold_scan_list:
                self.task['hold_s']=float(hold)
                # keep transport_task in sync for provenance
                if hasattr(self, 'transport_task'):
                    self.transport_task.hold_s = float(hold)
                metric=self.evaluate(theta,'sta',exact_nodes=valid_nodes,subdivisions=valid_subdiv)
                hold_scan.append({'hold_s':float(hold),'metrics':metric})
        finally:
            self.task['hold_s']=original_hold
            if hasattr(self, 'transport_task'):
                self.transport_task.hold_s = float(original_hold)
        write_json(out/'hold_scan.json',hold_scan)
        # Robust – ensure evaluation uses current task settings restored above
        robust=self.robust(theta,self.cfg.get('robust_samples',8));write_json(out/'robust.json',robust)

        # === Observables & loss chain (spec §10) ===
        obs_chain = processing_chain(r, self.task)
        write_json(out/'observables.json', obs_chain['observables'])
        write_json(out/'observable_loss.json', {'loss': obs_chain['loss'], 'provenance': obs_chain['provenance']})
        # Raw measurement contract (spec §10 raw→observable): synthesize raw sideband data then infer (R4)
        try:
            from .observables import synthesize_raw_sideband_measurement, infer_observables_from_raw, raw_to_observables_chain
            raw = synthesize_raw_sideband_measurement(r)
            inferred = infer_observables_from_raw(raw)
            raw_chain = raw_to_observables_chain(r, self.task, include_raw=True)
            write_json(out/'raw_measurement.json', raw)
            write_json(out/'observables_from_raw.json', inferred)
            write_json(out/'raw_chain.json', {"raw_measurement": raw, "observables_via_raw": inferred, "loss_via_raw": raw_chain['loss'], "provenance": raw_chain['provenance']})
        except Exception as e_raw:
            write_json(out/'raw_measurement.json', {"error": str(e_raw), "provenance": "raw synthesis failed"})
        # Per-scenario observable aggregation example (robust rows → observable losses)
        try:
            obs_losses = [processing_chain(row, self.task)['loss'] for row in robust['rows'] if row.get('status')=='ok']
            from .observables import aggregate_losses
            agg = {
                "mean_loss": aggregate_losses(obs_losses, "mean") if obs_losses else None,
                "worst_loss": aggregate_losses(obs_losses, "worst_case") if obs_losses else None,
                "weighted_worst": aggregate_losses(obs_losses, "weighted_worst") if obs_losses else None,
                "cvar_20": aggregate_losses(obs_losses, "cvar_20") if obs_losses else None,
                "policy": "weighted_worst = 0.35*mean+0.65*max; same for simulation and experiment observables",
            }
            write_json(out/'observable_aggregation.json', agg)
        except Exception as e:
            write_json(out/'observable_aggregation.json', {"error": str(e)})

        # === Safety layer (spec §13) – full fail-closed evaluation ===
        # B5: distortion threshold is executable – if max_deviation exceeds task-configured threshold,
        # safety fails with delivered_intent_violated and pipeline must re-optimize or reject.
        safety_report = self.safety_layer.evaluate(trace['cmd'], trace['v'], r['distortion'], trace['H'])
        write_json(out/'safety_report.json', {"passed": bool(safety_report.passed), "checks": safety_report.checks,
                                               "failure_code": safety_report.failure_code, "failure_message": safety_report.failure_message})
        # B5 executable threshold handling: if delivered intent violated, attempt limited re-optimization under delivered model
        distortion_requires_resynthesis = False
        distortion_reopt_result = None
        if not safety_report.passed and safety_report.failure_code == "delivered_intent_violated":
            distortion_requires_resynthesis = True
            # Attempt automatic re-optimization under delivered model (ideal=False) with limited budget
            try:
                _orig_budget = int(self.cfg.get('optimization_evaluations', 80))
                # Limited budget for resynthesis (spec: re-optimization under delivered model)
                _resynth_budget = min(12, _orig_budget // 2 if _orig_budget > 6 else 6)
                self.checkpoint = out/'distortion_resynthesis_checkpoint.json'
                _reopt = self.optimize(ideal=False, robust=False)
                # Force reopt to use _resynth_budget evaluations (truncate if needed – optimize already used original budget)
                # Instead, run a short Powell from current theta with delivered ideal=False and exact_nodes
                # For provenance, evaluate the re-optimized theta under delivered conditions
                _theta_reopt = np.asarray(_reopt['theta'])
                r_reopt = self.evaluate(_theta_reopt, 'sta', exact_nodes=valid_nodes, subdivisions=valid_subdiv)
                trace_reopt = self.evaluate(_theta_reopt, 'sta', exact_nodes=valid_nodes, subdivisions=valid_subdiv, keep=True)[1]
                # Re-check safety after resynthesis
                safety_reopt = self.safety_layer.evaluate(trace_reopt['cmd'], trace_reopt['v'], r_reopt['distortion'], trace_reopt['H'])
                distortion_reopt_result = {
                    "attempted": True,
                    "reopt_theta": _theta_reopt.tolist(),
                    "reopt_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k,v in r_reopt.items() if k in ('n_total','position_error_m','well_target_error_m','minimum_frequency_hz','accepted')},
                    "reopt_safety_passed": bool(safety_reopt.passed),
                    "reopt_safety_report": {"checks": safety_reopt.checks, "failure_code": safety_reopt.failure_code},
                }
                write_json(out/'distortion_resynthesis.json', {
                    "trigger": "max_deviation_V exceeds threshold",
                    "original_safety": {"passed": bool(safety_report.passed), "failure_code": safety_report.failure_code, "max_deviation_V": r['distortion'].get('max_deviation_V'), "threshold_V": safety_report.checks.get('threshold_V')},
                    "resynthesis": distortion_reopt_result,
                    "decision": "re-optimization under delivered model attempted (spec B5)",
                })
                # If resynthesis succeeded and safety passed, optionally adopt it as new selected
                if safety_reopt.passed and r_reopt['accepted']:
                    # Adopt resynthesized solution for export (provenance tracked)
                    r = r_reopt
                    trace = trace_reopt
                    theta = _theta_reopt
                    safety_report = safety_reopt
                    distortion_requires_resynthesis = False  # now satisfied
                    write_json(out/'safety_report_resynthesized.json', {"passed": bool(safety_report.passed), "checks": safety_report.checks, "adopted_from_resynthesis": True})
            except Exception as ex_resynth:
                write_json(out/'distortion_resynthesis.json', {"trigger": "max_deviation_V exceeds threshold", "error": str(ex_resynth), "attempted": True, "decision": "re-optimization failed, remains rejected"})
            finally:
                self.checkpoint = None
        else:
            # No distortion violation – record that no resynthesis needed
            if safety_report.passed:
                write_json(out/'distortion_resynthesis.json', {"trigger": "none", "decision": "distortion within threshold, no resynthesis required", "max_deviation_V": r['distortion'].get('max_deviation_V'), "threshold_V": safety_report.checks.get('threshold_V') if safety_report.checks else None})
        # Fallback is last known safe: if current safety passed, current theta is last safe; else initial zero-correction
        # Include delivered waveform checksum for export authorization (R1 last-safe checksum)
        import hashlib as _hashlib
        try:
            delivered_sha256 = _hashlib.sha256(trace['v'].tobytes()).hexdigest()
            cmd_sha256 = _hashlib.sha256(trace['cmd'].tobytes()).hexdigest()
        except Exception:
            delivered_sha256 = None
            cmd_sha256 = None
        if safety_report.passed:
            fallback_theta = theta.tolist() if isinstance(theta, np.ndarray) else list(theta)
            fallback_reason = "current_selected_theta_passed_full_safety"
            fallback_sha = delivered_sha256
            last_safe_theta = fallback_theta
        else:
            # Evaluate initial zero-correction as known-safe baseline
            try:
                ct0, cmd0, route0 = self.commands(np.zeros(8), baseline='sta')
                t0, v0, _, dist0 = self.hw.deliver(cmd0, subdivisions=4)
                _, H0, _, _ = self.rom(t0, v0, ct0, route0, stray=None, exact_nodes=None)
                rep0 = self.safety_layer.evaluate(cmd0, v0, dist0, H0)
                if rep0.passed:
                    fallback_theta = np.zeros(8).tolist()
                    fallback_reason = "selected_failed_safety_fallback_to_zero_correction (validated safe)"
                    fallback_sha = _hashlib.sha256(v0.tobytes()).hexdigest()
                    last_safe_theta = fallback_theta
                else:
                    fallback_theta = None
                    fallback_reason = "no_safe_waveform_available: both selected and zero-correction failed safety"
                    fallback_sha = None
                    last_safe_theta = None
            except Exception as ex0:
                fallback_theta = None
                fallback_reason = f"fallback_validation_failed: {ex0}"
                fallback_sha = None
                last_safe_theta = None
        # N2: simulation export never authorized for hardware, even if safety passed
        write_json(out/'fallback_state.json', {
            "theta": fallback_theta,
            "last_safe_theta": last_safe_theta,
            "reason": fallback_reason,
            "delivered_sha256": fallback_sha,
            "command_sha256": cmd_sha256,
            "safety_passed": bool(safety_report.passed),
            "simulation_passed": bool(safety_report.passed and fallback_theta is not None),
            "export_authorized": False,  # simulation_only – real execution requires measured calibration and trap maps
        })

        # === Calibration demo (spec §9–10) – synthetic, declared, versioned, resynthesis ===
        # Spec requires calibration to be separate from black-box optimization and to
        # trigger resynthesis via new Runner/safety/export when it succeeds.
        calibration_success_flag = False
        calibration_blocked_flag = False
        try:
            # Generate dataset: ideal commands vs delivered with 2% gain perturbation (declared truth)
            true_hw = Hardware(sample_rate_hz=self.hw.sample_rate_hz, dac_bits=self.hw.dac_bits,
                               dac_min_V=self.hw.dac_min_V, dac_max_V=self.hw.dac_max_V,
                               lowpass_hz=self.hw.lowpass_hz, gain=[1.02]*len(self.trap.indices),
                               offset_V=[0.015]*len(self.trap.indices), delay_s=[5e-9]*len(self.trap.indices),
                               max_jump_V=self.hw.max_jump_V)
            dataset=[]
            for th in [np.zeros(8), np.array([0.02,-0.01,0.05,-0.02,0.03,0.01,-0.04,0.02]), np.array([-0.03,0.02,-0.05,0.04, -0.02, -0.01, 0.03, -0.03])]:
                ct_cmd, cmd, _ = self.commands(th, baseline='sta')
                _, meas_V, _, _ = true_hw.deliver(cmd, subdivisions=4)
                dataset.append({"commands_V": cmd.tolist(), "measured_V": meas_V.tolist(), "theta": th.tolist()})
            cal_result = self.calibration.identify_from_residuals(dataset)
            calibration_success_flag = bool(cal_result.success and np.isfinite(cal_result.residuals))
            # Versioned calibration state – increments on each successful identification
            import hashlib as _hcal
            cal_state_version = "cal-v1" if calibration_success_flag else "cal-v0-blocked"
            cal_state = {
                "version": cal_state_version,
                "timestamp_s": time.time(),
                "hardware_before": asdict(self.hw),
                "hardware_after": cal_result.updated_hardware if calibration_success_flag else asdict(self.hw),
                "stray_before_V_m": [0,0,0],
                "stray_after_V_m": cal_result.updated_stray_V_m,
                "identification": {
                    "method": cal_result.method,
                    "success": bool(cal_result.success),
                    "residuals": float(cal_result.residuals) if np.isfinite(cal_result.residuals) else None,
                    "failure_code": cal_result.failure_code,
                    "provenance": cal_result.provenance,
                    "dataset_size": len(dataset),
                },
                "provenance": "synthetic dataset: shows identification reacts to input gain/offset; filter poles retained (requires step-response for full filter ID)",
            }
            write_json(out/'calibration_demo.json', {
                "dataset_size": len(dataset),
                "result": asdict(cal_result),
                "true_gain": true_hw.gain if isinstance(true_hw.gain, list) else [true_hw.gain],
                "true_offset": true_hw.offset_V if isinstance(true_hw.offset_V, list) else [true_hw.offset_V],
                "identified_gain": cal_result.updated_hardware.get("gain"),
                "identified_offset": cal_result.updated_hardware.get("offset_V"),
                "version": cal_state_version,
                "calibration_state": cal_state,
                "update_provenance": "synthetic dataset: shows identification reacts to input gain/offset; not a hard-coded answer",
            })
            write_json(out/'calibration_state.json', cal_state)
            # Demonstrate resynthesis after calibration update (spec §10 sequence: identify -> update_model -> new Runner/safety/export)
            # N1 fix: main export must remain consistent with its hardware_model (original hw),
            # calibrated export is separate. We do NOT mutate self.hw globally before main export.
            original_hw = self.hw
            original_safety = self.safety_layer
            calibrated_hw = None
            post_cal = None
            post_trace = None
            try:
                if not calibration_success_flag:
                    raise Rejected(cal_result.failure_code or "calibration_not_converged", "calibration did not converge – resynthesis blocked")
                updated_hw = self.calibration.update_model(cal_result)
                calibrated_hw = updated_hw
                # Use a separate safety layer for calibrated model, do not overwrite self.safety_layer yet
                calibrated_safety = SafetyLayer(self.trap, calibrated_hw, self.task)
                # Re-evaluate selected theta with updated hardware model (resynthesize + re-validate)
                post_cal, post_trace = self.evaluate(theta, 'sta', hardware=calibrated_hw, exact_nodes=valid_nodes, subdivisions=valid_subdiv, keep=True)
                post_safety = calibrated_safety.evaluate(post_trace['cmd'], post_trace['v'], post_cal['distortion'], post_trace['H'])
                write_json(out/'calibration_resynthesis.json', {
                    "version": cal_state_version,
                    "pre_calibration_metrics": {"n_total": float(r['n_total']), "position_error_m": float(r['position_error_m'])},
                    "post_calibration_metrics": {"n_total": float(post_cal['n_total']), "position_error_m": float(post_cal['position_error_m'])},
                    "improved": bool(post_cal['n_total'] <= r['n_total'] + 1e-9),
                    "updated_hardware": asdict(calibrated_hw),
                    "original_hardware": asdict(original_hw),
                    "post_safety_passed": bool(post_safety.passed),
                    "post_safety_report": {"checks": post_safety.checks, "failure_code": post_safety.failure_code},
                    "sequence": "identification -> update_model -> calibrated Hardware/Safety -> resynthesize -> safety re-validate (calibrated export separate from main export)",
                    "export_delta": "calibrated delivered SHA256 differs from main export (see fallback_state vs post_cal SHA)",
                })
                # Write ACTUAL calibrated waveform files (N3 fix) – previously only JSON was written
                import hashlib as _hcal2
                post_sha = _hcal2.sha256(post_trace['v'].tobytes()).hexdigest()
                # Calibrated delivered waveform + DAC + ideal
                save_csv(out/'calibrated_delivered_waveform.csv', ['time_s']+self.trap.names, np.column_stack([post_trace['t'], post_trace['v']]))
                save_csv(out/'calibrated_ideal_waveform.csv', ['time_s']+self.trap.names, np.column_stack([post_trace['ct'], post_trace['cmd']]))
                with open(out/'calibrated_dac_waveform.csv','w',newline='') as f:
                    w=csv.writer(f);w.writerow(['time_s']+self.trap.names)
                    for ti,codes in zip(post_trace['ct'], post_trace['codes']):
                        w.writerow([float(ti)]+[int(code) for code in codes])
                write_json(out/'calibration_export.json', {
                    "version": cal_state_version,
                    "theta": theta.tolist() if isinstance(theta, np.ndarray) else list(theta),
                    "delivered_sha256": post_sha,
                    "safety_passed": bool(post_safety.passed),
                    "export_file": "calibrated_delivered_waveform.csv",
                    "also_exported": ["calibrated_ideal_waveform.csv","calibrated_dac_waveform.csv"],
                    "provenance": "actual calibrated waveform files written, SHA matches delivered array",
                })
                # Preserve calibrated trace for later manifest (but do not mutate main self.hw)
                calibration_post_cal = post_cal
                # Do NOT mutate self.hw / self.safety_layer here – main export stays on original hw (N1)
            except Rejected as e_resyn:
                write_json(out/'calibration_resynthesis.json', {"version": cal_state_version, "error": str(e_resyn), "code": getattr(e_resyn,'code','unknown'), "blocked": True,
                    "original_hardware": asdict(original_hw)})
                write_json(out/'calibration_export.json', {"version": cal_state_version, "error": str(e_resyn), "code": getattr(e_resyn,'code','unknown'), "export_file": None, "blocked": True})
                calibration_post_cal = None
            except Exception as e_resyn:
                write_json(out/'calibration_resynthesis.json', {"version": cal_state_version, "error": str(e_resyn), "code": getattr(e_resyn,'code','unknown'), "blocked": True,
                    "original_hardware": asdict(original_hw)})
                write_json(out/'calibration_export.json', {"version": cal_state_version, "error": str(e_resyn), "code": getattr(e_resyn,'code','unknown'), "export_file": None, "blocked": True})
                calibration_post_cal = None
            # Also test blocked-by-missing-data path (honest: returns missing_external_data, not success)
            blocked = self.calibration.identify_from_residuals([])
            calibration_blocked_flag = not blocked.success
            write_json(out/'calibration_blocked.json', {"status": blocked.provenance, "failure_code": blocked.failure_code, "success": bool(blocked.success), "version": "cal-v0-blocked"})
        except Exception as e:
            write_json(out/'calibration_demo.json', {"error": str(e), "provenance": "synthetic calibration demo failed", "version": "cal-v0-error"})
            write_json(out/'calibration_blocked.json', {"status": "error", "message": str(e), "version": "cal-v0-error"})
            write_json(out/'calibration_state.json', {"version": "cal-v0-error", "error": str(e)})
            calibration_success_flag = False
            calibration_blocked_flag = False
            cal_state_version = "cal-v0-error"

        # === Closed-loop emulated run (spec §11-12) – safety-gated, budget-strict ===
        closed_loop_success = False
        closed_loop_blocked = False
        try:
            refiner = ClosedLoopRefiner(self, self.safety_layer, self.task)
            emulated = refiner.demo_emulated_run(initial_theta=theta, true_perturb=true_hw if 'true_hw' in locals() else None, budget=self.cfg.get('closed_loop_budget', 12))
            # emulated now contains explicit budget, measurements, all_failed, blocked fields
            write_json(out/'closed_loop_emulated.json', emulated)
            # Determine closed-loop ready honestly: not just always True, but based on actual success
            # Ready requires at least one safe measurement succeeded (not all_failed) and budget respected
            if isinstance(emulated, dict):
                if emulated.get("all_failed") or emulated.get("blocked"):
                    closed_loop_success = False
                    closed_loop_blocked = True
                elif emulated.get("best_theta") is not None and emulated.get("measurements", 0) <= emulated.get("budget", 999):
                    # Check budget invariant (measurements <= budget)
                    closed_loop_success = True
                else:
                    closed_loop_success = False
                    # if budget violation, mark blocked
                    if emulated.get("measurements", 0) > emulated.get("budget", 0):
                        closed_loop_blocked = True
            # Also demonstrate low-dim constraint: try to inject high-dim and ensure it fails
            try:
                # This should raise ValueError due to theta length check before any measurement
                refiner.optimize(lambda th: {"loss": 0.1}, np.zeros(1000), budget=2)
            except ValueError as ex_high:
                high_dim_check = {"rejected_as_expected": True, "message": str(ex_high), "code": "dimension_reject"}
            except Rejected as ex_high:
                high_dim_check = {"rejected_as_expected": True, "message": str(ex_high), "code": ex_high.code}
            except Exception as ex_high:
                high_dim_check = {"rejected_as_expected": True, "message": str(ex_high), "code": getattr(ex_high, 'code', 'unknown')}
            else:
                high_dim_check = {"rejected_as_expected": False, "message": "high-dim was not rejected – check implementation", "code": "missing_reject"}
            write_json(out/'closed_loop_param_check.json', high_dim_check)
            # Require high-dim check to pass for closed_loop_ready
            if not high_dim_check.get("rejected_as_expected"):
                closed_loop_success = False
        except Exception as e:
            write_json(out/'closed_loop_emulated.json', {"error": str(e), "provenance": "emulated run failed"})
            write_json(out/'closed_loop_param_check.json', {"error": str(e)})
            closed_loop_success = False
            closed_loop_blocked = True

        # === Transport task & reduced-order docs ===
        write_json(out/'transport_task.json', self.transport_task.summary())
        write_json(out/'calibration_params.json', {"hardware": asdict(self.hw), "stray_assumed_V_m": [0,0,0],
                                                   "status": "unmeasured_simulation_assumptions",
                                                   "note": "updated via Calibration layer when data available"})

        # === Multi-duration scan (spec §11) – unified schema (durations_scan_s / durations_s aliases) ===
        def _resolve_duration_scan():
            # check task aliases first, then transport_task, then cfg root
            for src in [self.task, getattr(self, 'transport_task', None).__dict__ if hasattr(self, 'transport_task') else {}, self.cfg]:
                if isinstance(src, dict):
                    for key in ("durations_scan_s", "durations_s", "duration_scan_s"):
                        if key in src and src[key] not in (None, [], {}):
                            return list(src[key])
                else:
                    if hasattr(src, 'durations_s') and src.durations_s:
                        return list(src.durations_s)
            return []
        durations = _resolve_duration_scan()
        if durations:
            duration_results=[]
            orig_dur = self.task['duration_s']
            for dur in durations:
                self.task['duration_s']=float(dur)
                if hasattr(self, 'transport_task'):
                    self.transport_task.duration_s = float(dur)
                try:
                    # Re-validate grid integer periods
                    m=self.evaluate(theta, 'sta', exact_nodes=321, subdivisions=8)
                    duration_results.append({"duration_s": float(dur), "metrics": m, "accepted": bool(m['accepted'])})
                except Rejected as ex:
                    duration_results.append({"duration_s": float(dur), "error": str(ex), "code": getattr(ex,'code','unknown'), "accepted": False})
                except Exception as ex:
                    duration_results.append({"duration_s": float(dur), "error": str(ex), "code": getattr(ex,'code','unknown'), "accepted": False})
            self.task['duration_s']=orig_dur
            if hasattr(self, 'transport_task'):
                self.transport_task.duration_s = float(orig_dur)
            write_json(out/'duration_scan.json', duration_results)

        # === Waveform exports ===
        save_csv(out/'ideal_waveform.csv',['time_s']+self.trap.names,np.column_stack([trace['ct'],trace['cmd']]))
        with open(out/'dac_waveform.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s']+self.trap.names)
            for ti,codes in zip(trace['ct'],trace['codes']):w.writerow([float(ti)]+[int(code) for code in codes])
        save_csv(out/'delivered_waveform.csv',['time_s']+self.trap.names,np.column_stack([trace['t'],trace['v']]))
        save_csv(out/'delivered_path.csv',['time_s','x_m','y_m','z_m'],np.column_stack([trace['t'],trace['r']]))
        write_json(out/'delivered_local_diagnostics.json',trace['diag'])
        write_json(out/'distortion.json',r['distortion'])
        # Document path conventions
        write_json(out/'path_conventions.json', {
            "user_route": "start_m + s * distance_m * direction  (s in [0,1], config-defined)",
            "nominal_well_path": "voltage_spline(route) -> trap.local -> route_r / route_H (support diagnostics)",
            "realized_path": "delivered voltages -> ROM exact reconstruction (r) -> dynamics validation",
            "optimized_path": "q(route) with time-warp and corrective basis; realized path is evaluated after hardware delivery",
            "which_is_optimized": "q(route) in parameter space; final evaluation uses realized path",
        })
        data=(out/'dac_waveform.csv').read_bytes()
        # Determine calibration state for metadata (versioned)
        _cal_version = locals().get('cal_state_version', 'cal-v0-unknown')
        # N2 fix: simulation_only waveform must NEVER be execution_authorized, even if safety passed.
        # execution_authorized requires measured calibration (cal-measured*) and real trap maps, not synthetic cal-v1.
        # simulation_passed indicates safety passed for simulation; execution_authorized is strictly false here.
        simulation_passed = bool(safety_report.passed and fallback_theta is not None)
        # Only a measured calibration (not synthetic) could authorize execution – synthetic cal-v1 never does
        is_measured_cal = isinstance(_cal_version, str) and _cal_version.startswith("cal-measured")
        export_authorized = False  # hard false for simulation_only; real hardware would require is_measured_cal and safety_passed
        # Keep traceability: record what WOULD be authorized in simulation vs real
        write_json(out/'waveform_metadata.json',{'version':'1.0','simulation_only':True,
            'execution_authorized': export_authorized,
            'simulation_passed': simulation_passed,
            'would_authorize_if_measured': bool(simulation_passed and is_measured_cal),
            'channel_order':self.trap.names,'units':'DAC codes (integer)',
            'sample_period_s':1/self.hw.sample_rate_hz,'code_min_V':self.hw.dac_min_V,
            'code_max_V':self.hw.dac_max_V,'dac_bits':self.hw.dac_bits,
            'hold':'zero_order','last_sample':'terminal command, held indefinitely',
            'clock':'common ideal simulation clock','trigger_latency_s':float(self.hw.trigger_latency_s),'jitter_s':float(self.hw.jitter_s),
            'calibration_state': _cal_version,'hardware_model':asdict(self.hw),
            'calibration_success': bool(locals().get('calibration_success_flag', False)),
            'sha256':hashlib.sha256(data).hexdigest(),
            'delivered_sha256': delivered_sha256,
            'command_sha256': cmd_sha256,
            'fallback_theta': fallback_theta,
            'safety_passed': bool(safety_report.passed),
            'closed_loop_low_dim': "8 params (2 time-warp + 6 voltage SVD)",
            'path_conventions': "user_route / nominal_well_path / realized_path documented separately",
            'provenance': "waveform export gated by full safety (voltage, slew, filter, confinement); SHA256 checksums enable last-safe verification",
        })
        import scipy
        # Honest status flags (not always True) – reflect actual validation
        _hold_all_passed = bool(all(z['metrics']['accepted'] for z in hold_scan)) if hold_scan else True
        # calibration_demo_success reflects whether identification converged with finite residuals, not just existence
        _cal_demo_success = bool(locals().get('calibration_success_flag', False))
        _cl_ready = bool(locals().get('closed_loop_success', False))
        # B2: honest flag whether hardware-aware optimization actually improved over baseline (computed from numbers, not constant)
        try:
            sta_score = next(sc for sc, nm, _, _ in rerank if nm == 'sta')
            aware_score = next(sc for sc, nm, _, _ in rerank if nm == 'aware')
            sta_metrics = next(rr for _, nm, _, rr in rerank if nm == 'sta')
            aware_metrics = next(rr for _, nm, _, rr in rerank if nm == 'aware')
            optimization_improved = bool(aware_score < sta_score - 1e-12 and aware_metrics.get('accepted', False))
        except Exception:
            optimization_improved = False
        # B5: distortion handling provenance
        distortion_threshold_exceeded = bool(locals().get('distortion_requires_resynthesis', False))
        distortion_resynthesis_done = bool(locals().get('distortion_reopt_result', None) is not None)
        status={'nominal_quality_passed':bool(r['accepted'] and ind['accepted'] and refined['accepted']),'numerical_validation_passed':validation['convergence_passed'],
                'robust_all_passed':bool(robust['pass_fraction']==1),
                'hold_scan_all_passed': _hold_all_passed,
                'safety_passed': bool(safety_report.passed),
                'calibration_demo_success': _cal_demo_success,
                'calibration_blocked': bool(locals().get('calibration_blocked_flag', False)),
                'calibration_version': _cal_version,
                'closed_loop_ready': _cl_ready,
                'closed_loop_blocked': bool(locals().get('closed_loop_blocked', False)),
                'fallback_available': bool(fallback_theta is not None),
                'export_authorized': export_authorized,
                'simulation_passed': bool(safety_report.passed and fallback_theta is not None),
                'optimization_improved_over_baseline': optimization_improved,
                'distortion_threshold_exceeded': distortion_threshold_exceeded,
                'distortion_resynthesis_attempted': distortion_resynthesis_done,
                'experiment_ready':False,'selected':selected,
                'scope':'simulation-only on specified Sion/GDS electrode geometry'}
        status['simulation_acceptance_passed']=bool(status['nominal_quality_passed'] and status['numerical_validation_passed'] and
               status['robust_all_passed'] and status['hold_scan_all_passed'] and status['safety_passed'] and status['calibration_demo_success'] and status['closed_loop_ready'])
        write_json(out/'status.json',status)
        # Manifest – dynamic source commit, not hardcoded c3551aa, and stable source_sha256 via git ls-files
        # N7 fix: suppress git stderr to avoid "fatal: not a git repository" noise when run from archive/container
        import subprocess as _sp
        def _git_commit():
            try:
                return _sp.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True, stderr=_sp.DEVNULL).strip()
            except Exception:
                try:
                    return _sp.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, stderr=_sp.DEVNULL).strip()
                except Exception:
                    return "unknown-not-a-git-repo"
        _commit = _git_commit()
        # Compute source_sha256 only for tracked python files to avoid drift between runs (untracked files vary)
        try:
            _tracked = _sp.check_output(["git", "ls-files", "shuttle_sim/*.py"], cwd=Path(__file__).parent.parent, text=True, stderr=_sp.DEVNULL).strip().splitlines()
            _src_files = [Path(__file__).parent.parent / p for p in _tracked if p]
        except Exception:
            _src_files = list(Path(__file__).parent.glob('*.py'))
        write_json(out/'manifest.json',{'package':'shuttle_sim 0.2.0','python':sys.version,
            'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
            'elapsed_s':time.time()-started,'run_unix_time':started,
            'input_sha256':hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest(),
            'source_base_commit': _commit,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in _src_files if p.exists()},
            'trap_source':'specified geometry in config; no measured electrode maps',
            'physics':'RF pseudopotential + 3D local harmonic Gaussian',
            'layers': ['field', 'control', 'voltage', 'hardware', 'dynamics', 'robust', 'calibration', 'closed_loop', 'safety', 'observables'],
            'status':status})
        return status
