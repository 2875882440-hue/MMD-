"""Transactional held-pose warmup for the bounded discrete joint kernel.

Not a native character adapter or a production CCD/cloth simulator. The input
pose, constraints and force are held constant. A failed candidate is restored;
no pin release, mass change, threshold relaxation or parameter retry occurs.
"""
import copy
from dataclasses import dataclass, asdict
import math
import numpy as np
from .secondary_attachment_driver import exp_rotation, rotation_vector, _array
from .secondary_contact_constraints import prewarm_convergence


@dataclass(frozen=True)
class WarmupSettings:
    dt: float = 1/60
    max_steps: int = 300
    ramp_steps: int = 30
    window_steps: int = 30
    minimum_full_force_steps: int = 30
    damping_per_second: float = 2.
    projection_iterations: int = 96
    projection_tolerance_m: float = 1e-5
    max_initial_shift_m: float = .05
    force_balance_limit_m_s2: float = .1

    def validate(self):
        for name in ('max_steps','ramp_steps','window_steps','minimum_full_force_steps','projection_iterations'):
            if type(getattr(self,name)) is not int:raise ValueError(name+' must be an integer')
        for name in ('dt','damping_per_second','projection_tolerance_m','max_initial_shift_m','force_balance_limit_m_s2'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(name+' must be finite')
        if not 1/1000<=self.dt<=1/30:raise ValueError('Warmup dt must be between 1/1000 and 1/30 seconds')
        if not 1<=self.max_steps<=600 or not 0<=self.ramp_steps<self.max_steps:raise ValueError('Invalid warmup budget')
        if self.window_steps<2 or self.window_steps*self.dt<.5-1e-12:raise ValueError('Trailing full-force window must cover at least 0.5 seconds')
        if self.minimum_full_force_steps<self.window_steps or self.ramp_steps+self.minimum_full_force_steps>self.max_steps:raise ValueError('Insufficient full-force warmup budget')
        if not 1<=self.projection_iterations<=256:raise ValueError('Invalid projection budget')
        if self.damping_per_second<0 or self.projection_tolerance_m<=0 or self.max_initial_shift_m<=0 or self.force_balance_limit_m_s2<=0:raise ValueError('Invalid warmup settings')
        # Speed limits are the existing 0.02 / 0.1 m/s gate, never configurable
        # here. Extra checks may be stricter, not a bypass around that gate.
        return self


def prewarm_status():
    return {'implementation':'held_pose_joint_prewarm_v1','core_implemented':True,
            'initial_conflict_diagnostics':True,'pin_release':False,
            'full_force_window_required':True,'undamped_balance_probe':True,
            'production_ready':False,'native_model_validated':False}


def _snapshot(frame):return frame.x.copy(),copy.deepcopy(frame.rigid)


def _restore(frame,state):
    frame.x[:]=state[0];frame.rigid=copy.deepcopy(state[1])


def _edges(system,frame):
    return [{'kind':'edge','ids':(a,b),'length':float(np.linalg.norm(frame.source[a]-frame.source[b]))} for a,b in system.edges]


def diagnose_initial_state(system,frame,tolerance_m=1e-5):
    if frame.system is not system.attachment:raise ValueError('Frame belongs to another joint system')
    if not np.isfinite(frame.positions()).all() or not np.array_equal(frame.x[frame.system.fixed],frame.source[frame.system.fixed]):raise ValueError('Invalid prediction or modified protected source vertices')
    if not math.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('Invalid diagnostic tolerance')
    audit=system.audit(frame,_edges(system,frame));conflicts=[]
    for c in system._features(frame):
        depth=-system._value(frame,c)
        if depth<=tolerance_m:continue
        ids=c['ids'];dofs=frame._dofs({'child':ids[0],'target':ids[1:]})
        if not dofs:conflicts.append({'kind':'fixed_contact','channel':c['channel'],'vertices':list(ids),'deficit_m':float(depth)})
    for c in frame.system.constraints:
        residual=float(np.linalg.norm(frame.residual(c)))
        if residual>tolerance_m and not frame._dofs(c):conflicts.append({'kind':'fixed_attachment','part':c['part'],'residual_m':residual})
    return {'source_state_audit':audit,'definite_fixed_conflicts':conflicts,
            'fixed_conflicts_found':bool(conflicts),'all_conflicts_classified':False,
            'pin_release_performed':False}


def _moving_vertices(frame):
    ids=[]
    for v in range(len(frame.x)):
        pid=frame.system.rigid_owner.get(v)
        if (pid is None and frame.weights[v]>0) or (pid is not None and (frame.mass[pid]>0 or np.any(frame.inertia[pid]))):ids.append(v)
    return np.array(ids,dtype=int)


def _range_projection(matrix,value):return matrix@np.linalg.pinv(matrix,rcond=1e-12)@value


def _predict(frame,velocity,linear,angular,acceleration,dt,damping):
    # Gravity is acceleration, independent of mass; locked DOFs receive none.
    free=np.array([v for v in range(len(frame.x)) if frame.weights[v]>0 and v not in frame.system.rigid_owner],int)
    velocity[free]=velocity[free]*math.exp(-damping*dt)+acceleration*dt
    frame.x[free]+=velocity[free]*dt
    for pid,state in frame.rigid.items():
        if frame.mass[pid]>0:
            linear[pid]=linear[pid]*math.exp(-damping*dt)+acceleration*dt
            state['center']+=linear[pid]*dt
        else:linear[pid][:]=0.
        r=state['rotation'];world_inertia=r@frame.inertia[pid]@r.T
        angular[pid]=_range_projection(world_inertia,angular[pid])*math.exp(-damping*dt)
        turn=angular[pid]*dt
        if np.linalg.norm(turn)>.25:raise ValueError('Angular step exceeds the discrete solver range; smaller substeps required')
        state['rotation']=exp_rotation(turn)@r


def _reconstruct(frame,old,dt):
    velocity=(frame.x-old[0])/dt;linear={};angular={}
    for pid,state in frame.rigid.items():
        previous=old[1][pid]
        linear[pid]=(state['center']-previous['center'])/dt
        angular[pid]=rotation_vector(state['rotation']@previous['rotation'].T)/dt
    return velocity,linear,angular


def _speeds(frame,previous_positions,velocity,linear,angular,dt,ids):
    positions=frame.positions();chord=np.linalg.norm(positions[ids]-previous_positions[ids],axis=1)/dt
    instantaneous=velocity[ids].copy()
    for i,v in enumerate(ids):
        pid=frame.system.rigid_owner.get(int(v))
        if pid is not None:instantaneous[i]=linear[pid]+np.cross(angular[pid],positions[v]-frame.rigid[pid]['center'])
    return np.maximum(chord,np.linalg.norm(instantaneous,axis=1))


def _balance_probe(system,frame,acceleration,settings,ids):
    """Full force, zero damping and zero initial velocity, on a scratch state.

    This is a static force-response probe, NOT the measured velocity history.
    A heavily damped freely falling patch must not be labelled settled.
    """
    state=_snapshot(frame);before=frame.positions()
    try:
        v=np.zeros_like(frame.x);linear={p:np.zeros(3) for p in frame.rigid};angular=copy.deepcopy(linear)
        _predict(frame,v,linear,angular,acceleration,settings.dt,0.)
        report=system.solve(frame,iterations=settings.projection_iterations,tolerance_m=settings.projection_tolerance_m)
        residual=float((np.linalg.norm(frame.positions()[ids]-before[ids],axis=1)/settings.dt**2).max(initial=0))
        return {'passed':bool(report['discrete_constraints_converged'] and residual<=settings.force_balance_limit_m_s2),
                'projected':report['discrete_constraints_converged'],'maximum_unbalanced_acceleration_m_s2':residual,
                'limit_m_s2':settings.force_balance_limit_m_s2,'damping_per_second':0.,'full_force':True}
    finally:_restore(frame,state)


def run_prewarm(system,frame,*,settings=None,acceleration=(0.,0.,-9.81),
                initial_vertex_velocity=None,initial_rigid_velocity=None,initial_rigid_angular_velocity=None,cancelled=None):
    cfg=(settings or WarmupSettings()).validate()
    acceleration=_array(acceleration,(3,),'Warmup acceleration')
    diagnosis=diagnose_initial_state(system,frame,cfg.projection_tolerance_m)
    ids=_moving_vertices(frame);original=_snapshot(frame)
    velocity=np.zeros_like(frame.x) if initial_vertex_velocity is None else _array(initial_vertex_velocity,frame.x.shape,'Vertex velocity')
    rigid_ids=sorted(frame.rigid)
    def vectors(values):
        if values is None:return {p:np.zeros(3) for p in rigid_ids}
        if set(values)!=set(rigid_ids):raise ValueError('Explicit velocity is required for every rigid part')
        return {p:_array(values[p],(3,),'Rigid velocity') for p in rigid_ids}
    linear=vectors(initial_rigid_velocity);angular=vectors(initial_rigid_angular_velocity)
    for v in range(len(frame.x)):
        if (frame.weights[v]==0 or v in frame.system.rigid_owner) and np.any(velocity[v]):raise ValueError('Fixed/rigid vertices cannot receive independent velocities')
    for pid in rigid_ids:
        if frame.mass[pid]==0 and np.any(linear[pid]):raise ValueError('Fixed rigid translation cannot receive velocity')
        r=frame.rigid[pid]['rotation'];w=r@frame.inertia[pid]@r.T
        if not np.allclose(_range_projection(w,angular[pid]),angular[pid],atol=1e-12,rtol=0):raise ValueError('Locked rigid rotation cannot receive velocity')
    result={'settings':asdict(cfg),'initial_diagnosis':diagnosis,'initial_projection':None,
            'initial_shift_max_m':0.,'steps_completed':0,'full_force_steps':0,'history':[],
            'prewarm_convergence':{'passed':False,'trailing_frames':0,'reason':'not_measured'},
            'velocity_convergence':None,'force_balance_probe':None,'core_prewarm_passed':False,
            'candidate_committed':False,'rolled_back':False,'physics_accepted':False,
            'original_before_quality_gate_checked':False,'continuous_collision_checked':False,
            'self_collision_checked':False,'reason':None,'measured_moving_vertices':len(ids)}
    def fail(reason):
        _restore(frame,original)
        result['prewarm_convergence'].update(passed=False,reason=reason)
        result.update(reason=reason,rolled_back=True);return result
    try:
        if cancelled and cancelled():return fail('cancelled')
        if diagnosis['fixed_conflicts_found']:return fail('initial_fixed_constraint_conflict')
        if not len(ids):return fail('no_dynamic_degrees_of_freedom')
        before=frame.positions()
        projected=system.solve(frame,iterations=cfg.projection_iterations,tolerance_m=cfg.projection_tolerance_m)
        result['initial_projection']=projected
        if not projected['discrete_constraints_converged']:return fail('initial_joint_projection_failed')
        shift=float(np.linalg.norm(frame.positions()-before,axis=1).max());result['initial_shift_max_m']=shift
        if shift>cfg.max_initial_shift_m:return fail('initial_correction_budget_exceeded')
        # Initial overlap repair is a separately reported static preparation,
        # not a simulated time step. Explicit incoming velocities are retained.
        history=[]
        for step in range(cfg.max_steps):
            if cancelled and cancelled():return fail('cancelled')
            state=_snapshot(frame);previous=frame.positions()
            ramp=min(1.,(step+1)/cfg.ramp_steps) if cfg.ramp_steps else 1.
            scale=ramp*ramp*(3-2*ramp)
            _predict(frame,velocity,linear,angular,acceleration*scale,cfg.dt,cfg.damping_per_second)
            projected=system.solve(frame,iterations=cfg.projection_iterations,tolerance_m=cfg.projection_tolerance_m)
            if not projected['discrete_constraints_converged']:
                result['failed_projection']=projected;return fail('warmup_joint_projection_failed')
            velocity,linear,angular=_reconstruct(frame,state,cfg.dt)
            speeds=_speeds(frame,previous,velocity,linear,angular,cfg.dt,ids)
            if not np.isfinite(speeds).all():raise ValueError('Nonfinite measured warmup speed')
            result['steps_completed']=step+1
            full_force=step>=cfg.ramp_steps
            if full_force:
                result['full_force_steps']+=1;history.append(speeds)
                if len(history)>cfg.window_steps:history.pop(0)
            result['history'].append({'step':step+1,'force_scale':scale,'full_force_window_eligible':full_force,
                'maximum_vertex_speed_m_s':float(speeds.max()),'p95_m_s':float(np.percentile(speeds,95)),
                'maximum_contact_deficit_m':projected['candidate_audit']['maximum_contact_deficit_m'],
                'attachment_residual_m':projected['candidate_audit']['attachment_residual_m']})
            if result['full_force_steps']<cfg.minimum_full_force_steps:continue
            gate=prewarm_convergence(np.array(history),p95_limit=.02,max_limit=.1)
            result['velocity_convergence']=gate
            result['prewarm_convergence']={**gate,'passed':False,'reason':'force_balance_not_verified'}
            if not gate['passed']:continue
            probe=_balance_probe(system,frame,acceleration,cfg,ids);result['force_balance_probe']=probe
            if not probe['passed']:return fail('full_force_equilibrium_not_reached')
            result['prewarm_convergence']={**gate,'passed':True,'force_balance_verified':True}
            result.update(core_prewarm_passed=True,candidate_committed=True,
                          final_vertex_velocity=velocity.tolist(),
                          final_rigid_velocity={p:v.tolist() for p,v in linear.items()},
                          final_rigid_angular_velocity={p:v.tolist() for p,v in angular.items()})
            return result
        if history:
            gate=prewarm_convergence(np.array(history),p95_limit=.02,max_limit=.1)
            result['velocity_convergence']=gate;result['prewarm_convergence']={**gate,'passed':False}
        return fail('warmup_budget_exhausted')
    except Exception:
        _restore(frame,original)
        raise
