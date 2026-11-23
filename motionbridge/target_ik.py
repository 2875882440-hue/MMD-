"""Body-relative endpoint goals for different target proportions (NumPy only)."""
import numpy as np
from .refinement import fit_two_bone


def hand_goal(source_pose, names, target_heads, side, target_lengths):
    """Blend ordinary arm motion with head-relative and bimanual relationships.

    Head-relative hands must not inherit the source character's head proportions.
    Only source geometry drives the blend; no video names or frame rules.
    """
    p=np.asarray(source_pose)
    j={n:names.index(n) for n in [side+'Arm',side+'ForeArm',side+'Hand','Head','Chest','LeftHand','RightHand']}
    a,b,c=[j[side+n] for n in ['Arm','ForeArm','Hand']]
    source_length=np.linalg.norm(p[b]-p[a])+np.linalg.norm(p[c]-p[b])
    scale=sum(target_lengths)/max(source_length,1e-6)
    ordinary=target_heads['shoulder']+(p[c]-p[a])*scale
    head_distance=np.linalg.norm(p[c]-p[j['Head']])/max(source_length,1e-6)
    near_head=np.clip((.8-head_distance)/.35,0,1)
    head_goal=target_heads['head']+(p[c]-p[j['Head']])*scale
    goal=ordinary*(1-near_head)+head_goal*near_head
    hand_distance=np.linalg.norm(p[j['LeftHand']]-p[j['RightHand']])/max(source_length,1e-6)
    together=np.clip((.4-hand_distance)/.25,0,1)*(1-near_head)
    chest_goal=target_heads['chest']+(p[c]-p[j['Chest']])*scale
    return goal*(1-together)+chest_goal*together


def chain_ik(start,middle,end,goal,rest_lengths,weight=1.):
    goal=np.asarray(end)*(1-weight)+np.asarray(goal)*weight
    return fit_two_bone(np.asarray(start),goal,np.asarray(middle),*rest_lengths,np.asarray(middle))
