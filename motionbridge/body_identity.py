"""Experimental COCO-17 anatomical identity continuity, independent of clip/time.

Second-order dynamic programming chooses raw or whole-body L/R assignment.
It preserves measured coordinates/confidences; it does not infer unseen joints.
The first frame anchors the labels. Prolonged occlusion remains ambiguous.
"""
import numpy as np

COCO_FLIP = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])

def stabilize_coco_identity(keypoints, boxes):
    points = np.asarray(keypoints, float)
    boxes = np.asarray(boxes, float)
    if points.ndim != 3 or points.shape[1:] != (17,3) or boxes.shape != (len(points),3):
        raise ValueError('Expected keypoints[T,17,3] and crop boxes[T,3]')
    if not np.isfinite(points).all() or not np.isfinite(boxes).all() or (boxes[:,2]<=0).any():
        raise ValueError('Identity tracking needs finite observations and positive crop sizes')
    if len(points) < 3:
        return points.copy(), {'enabled':True,'swapped_frames':[],'reason':'too_short'}
    normalized = (points[...,:2]-boxes[:,None,:2])/boxes[:,None,2:]
    candidates = np.stack((normalized,normalized[:,COCO_FLIP]),axis=1)
    # Torso labels are more useful for whole-body identity than fast-moving wrists.
    weights = np.array([0.,.1,.1,.1,.1,2.,2.,.5,.5,.15,.15,2.,2.,.5,.5,.4,.4])
    confidence = np.minimum(points[...,2],points[:,COCO_FLIP,2])
    back = np.zeros((len(points),2,2),dtype=np.int8)
    cost = np.full((2,2),np.inf)
    for b in range(2):
        delta = candidates[1,b]-candidates[0,0]
        cost[0,b] = float(np.sum(delta**2*weights[:,None]))*.05 + .00005*b
    for t in range(2,len(points)):
        new = np.full((2,2),np.inf)
        w = weights*np.clip(np.minimum.reduce(confidence[t-2:t+1]),0.,1.)
        for b in range(2):
            for c in range(2):
                options = []
                for a in range(2):
                    velocity = candidates[t,c]-candidates[t-1,b]
                    acceleration = velocity-(candidates[t-1,b]-candidates[t-2,a])
                    energy = np.sum(np.minimum(acceleration**2,.04)*w[:,None])
                    energy += .05*np.sum(np.minimum(velocity**2,.04)*w[:,None])
                    options.append(cost[a,b]+energy+.00005*c)
                best = int(np.argmin(options))
                new[b,c], back[t,b,c] = options[best], best
        cost = new
    states = np.zeros(len(points),dtype=int)
    states[-2], states[-1] = np.unravel_index(np.argmin(cost),cost.shape)
    for t in range(len(points)-1,1,-1):
        states[t-2] = back[t,states[t-1],states[t]]
    result = points.copy()
    swapped = np.flatnonzero(states)
    result[swapped] = points[swapped][:,COCO_FLIP]
    return result, {'enabled':True,'method':'coco17_second_order_identity_v1',
                    'swapped_frames':swapped.tolist(),'anchor':'first_frame_raw',
                    'limitations':'Identity continuity is a hypothesis; prolonged occlusion, incorrect initial labels and symmetric motion remain ambiguous.'}


def stabilize_arm_observations(keypoints, boxes, fps=30.):
    """Resolve elbow/wrist pair identity, then mark short isolated excursions.

    Run after whole-body identity. Torso and legs are never reassigned here.
    Interpolation is only a hypothesis: confidence is reduced, not fabricated.
    """
    points = np.asarray(keypoints, float)
    boxes = np.asarray(boxes, float)
    if points.ndim != 3 or points.shape[1:] != (17,3) or boxes.shape != (len(points),3):
        raise ValueError('Expected COCO17 observations and crop boxes')
    if not np.isfinite(points).all() or not np.isfinite(boxes).all() or (boxes[:,2]<=0).any() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid arm observations')
    result = points.copy()
    count = len(points)
    report = {'enabled': True, 'method': 'distal_pair_identity_and_short_excursion_v1',
              'elbow_swaps': [], 'wrist_swaps': [], 'interpolated': [],
              'note': 'Corrections are hypotheses, not visibility or accuracy. Interpolated confidence is reduced.'}
    if count < 3:
        return result, report
    normalized = (points[...,:2]-boxes[:,None,:2])/boxes[:,None,2:]
    candidates = np.repeat(normalized[:,None,7:11],4,axis=1)
    for state in range(4):
        if state & 1: candidates[:,state,[0,1]] = candidates[:,state,[1,0]]
        if state & 2: candidates[:,state,[2,3]] = candidates[:,state,[3,2]]
    back = np.zeros((count,4,4),dtype=np.int8)
    costs = np.full((4,4),np.inf)
    for b in range(4):
        costs[0,b] = .05*np.sum((candidates[1,b]-candidates[0,0])**2)+.0008*(bool(b&1)+bool(b&2))
    for t in range(2,count):
        new = np.full((4,4),np.inf)
        for b in range(4):
            for c in range(4):
                velocity = candidates[t,c]-candidates[t-1,b]
                options=[]
                for a in range(4):
                    acceleration = velocity-(candidates[t-1,b]-candidates[t-2,a])
                    energy=np.sum(np.minimum(acceleration**2,.09))+.08*np.sum(np.minimum(velocity**2,.09))
                    options.append(costs[a,b]+energy+.0008*(bool(c&1)+bool(c&2)))
                best=int(np.argmin(options));new[b,c]=options[best];back[t,b,c]=best
        costs=new
    states=np.zeros(count,dtype=int)
    states[-2],states[-1]=np.unravel_index(np.argmin(costs),costs.shape)
    for t in range(count-1,1,-1): states[t-2]=back[t,states[t-1],states[t]]
    for bit,pair,label in [(1,[7,8],'elbow_swaps'),(2,[9,10],'wrist_swaps')]:
        frames=np.flatnonzero(states&bit)
        result[np.ix_(frames,pair)] = points[np.ix_(frames,pair[::-1])]
        report[label]=frames.tolist()
    # A large excursion followed by a return within <=100ms can be a duplicate
    # detection on the other hand or sleeve. Do not smooth sustained fast motion.
    norm=(result[...,:2]-boxes[:,None,:2])/boxes[:,None,2:]
    maximum=max(1,min(3,round(fps*.1)))
    for j in [7,8,9,10]:
        f=1
        while f<count-1:
            repaired=False
            for gap in range(1,maximum+1):
                end=f+gap
                if end>=count: break
                a,b=norm[f-1,j],norm[end,j]
                entry=np.linalg.norm(norm[f,j]-a)
                exit_step=np.linalg.norm(norm[end-1,j]-b)
                net=np.linalg.norm(b-a)
                if min(entry,exit_step)<.18 or net>.10 or min(result[f-1,j,2],result[end,j,2])<.55:
                    continue
                for k in range(f,end):
                    alpha=(k-f+1)/(gap+1)
                    # Interpolate measured image pixels, not normalized crop positions.
                    result[k,j,:2]=(1-alpha)*result[f-1,j,:2]+alpha*result[end,j,:2]
                    result[k,j,2]=min(result[k,j,2],result[f-1,j,2],result[end,j,2])*.25
                    report['interpolated'].append([k,j])
                f=end;repaired=True;break
            if not repaired: f+=1
    return result,report
