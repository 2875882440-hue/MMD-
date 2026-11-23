"""Optional local hand observations and conservative anatomical wrist association.

Hand classifier scores are not body visibility or a physical accuracy score.
Only image-plane wrists are corrected; no finger animation is claimed.
"""
import itertools
import numpy as np


class HandObserver:
    def __init__(self, model_path):
        import mediapipe as mp
        self.mp = mp
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=.5, min_hand_presence_confidence=.5))

    def close(self):
        self.detector.close()

    def observe(self, rgb, pose, confidence):
        height, width = rgb.shape[:2]
        indices = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
        visible = [i for i in indices if confidence[i] >= .35 and np.isfinite(pose[i, :2]).all()]
        if len(visible) < 4:
            return []
        points = pose[visible, :2] * [width, height]
        # Include head-top and raised hands even when a body wrist is misplaced.
        margin = max(60., np.linalg.norm((pose[11, :2]-pose[12, :2])*[width,height]) * .8)
        x0, y0 = np.maximum(0, np.floor(points.min(axis=0) - margin)).astype(int)
        x1, y1 = np.minimum([width,height], np.ceil(points.max(axis=0) + margin)).astype(int)
        if min(x1-x0, y1-y0) < 40:
            return []
        regions = [(x0,y0,x1,y1)]
        for wrist in [15,16]:
            if confidence[wrist] >= .35 and np.isfinite(pose[wrist,:2]).all():
                cx,cy = pose[wrist,:2]*[width,height]
                radius = max(60.,height*.115)
                regions.append((int(max(0,cx-radius)),int(max(0,cy-radius)),int(min(width,cx+radius)),int(min(height,cy+radius))))
        hands = []
        for x0,y0,x1,y1 in regions:
            if min(x1-x0,y1-y0)<40:
                continue
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
            for landmarks, categories in zip(result.hand_landmarks, result.handedness):
                xy = np.array([[p.x*(x1-x0)+x0, p.y*(y1-y0)+y0] for p in landmarks]) / [width,height]
                span = np.linalg.norm((xy[9]-xy[0])*[width/height,1])
                score = float(categories[0].score)
                if np.isfinite(xy).all() and .006 < span < .18 and np.all((xy[0] >= 0) & (xy[0] <= 1)) and score >= .65:
                    hands.append({'wrist':xy[0].tolist(), 'score':score,
                                  'handedness':categories[0].category_name})
        unique = []
        for hand in sorted(hands,key=lambda h:-h['score']):
            if all(np.linalg.norm((np.array(hand['wrist'])-h['wrist'])*[width/height,1])>.018 for h in unique):
                unique.append(hand)
        return unique[:4]


def _associate_direction(image, confidence, candidates, fps, width, height):
    """One-to-one assignment using body attachment and short temporal tracks.

Never label by screen-left/screen-right. Ambiguous assignments abstain instead
of pulling both arms to the same hand. Reset across unobserved intervals.
"""
    out = np.asarray(image, float).copy()
    support = np.zeros((len(out), 2))
    corrections = np.zeros((len(out), 2))
    previous = [None, None]
    velocity = [np.zeros(2), np.zeros(2)]
    last_seen = [-10000, -10000]
    aspect = np.array([width/height, 1.])
    ambiguous = 0
    for frame, hands in enumerate(candidates):
        hands = hands[:4]
        if not hands:
            continue
        torso = max(.08, np.linalg.norm(((out[frame,11,:2]+out[frame,12,:2])-(out[frame,23,:2]+out[frame,24,:2]))*.5*aspect))
        costs = np.full((2, len(hands)), 100.)
        for side, (wrist, elbow) in enumerate([(15,13),(16,14)]):
            if confidence[frame, elbow] < .35:
                continue
            for k, hand in enumerate(hands):
                point = np.asarray(hand['wrist'])
                distance = np.linalg.norm((point-out[frame,wrist,:2])*aspect)/torso
                reach = np.linalg.norm((point-out[frame,elbow,:2])*aspect)/torso
                if distance > 1.25 or reach > 1.5:
                    continue
                cost = .65*distance + .12*abs(reach-.55) + .15*(1-hand['score'])
                if frame-last_seen[side] <= max(1, round(fps*.25)) and previous[side] is not None:
                    predicted = previous[side] + velocity[side] * (frame-last_seen[side])
                    cost += .6*np.linalg.norm((point-predicted)*aspect)/torso
                costs[side,k] = cost
        choices = []
        for pair in itertools.product(range(-1,len(hands)), repeat=2):
            if pair[0] >= 0 and pair[0] == pair[1]:
                continue
            cost = sum(.8 if k < 0 else costs[s,k] for s,k in enumerate(pair))
            choices.append((cost,pair))
        choices.sort()
        best = choices[0][1]
        # An almost-tied permutation is not evidence for which wrist is which.
        if len(choices)>1 and choices[1][0]-choices[0][0] < .04 and best != (-1,-1):
            ambiguous += 1
            continue
        for side,k in enumerate(best):
            if k < 0 or costs[side,k] >= .75:
                continue
            point = np.asarray(hands[k]['wrist'])
            if frame-last_seen[side] == 1:
                velocity[side] = np.clip(point-previous[side], -.04, .04)
            else:
                velocity[side] = np.zeros(2)
            previous[side], last_seen[side] = point, frame
            wrist = 15+side
            weight = .85 * np.clip((hands[k]['score']-.6)/.3,0,1)
            corrections[frame,side] = np.linalg.norm((point-out[frame,wrist,:2])*aspect)*weight
            out[frame,wrist,:2] += weight*(point-out[frame,wrist,:2])
            support[frame,side] = weight
    return out, support, {'enabled':True, 'method':'cropped_hand_wrist_temporal_assignment_v1',
        'candidate_frames':sum(bool(h) for h in candidates), 'associated_wrist_samples':int((support>0).sum()),
        'ambiguous_frames_skipped':ambiguous, 'mean_image_correction':float(corrections[support>0].mean()) if np.any(support>0) else 0.,
        'note':'Independent image wrist observations only; original body confidence unchanged; no finger animation or true depth.'}


def associate_wrists(image, confidence, candidates, fps, width, height):
    """Offline forward/backward agreement prevents one-way identity drift.

    Weak body wrists require independent temporal agreement. Unsupported frames
    remain body observations; a tracking weight is not detector confidence.
    """
    forward, fw, report = _associate_direction(image, confidence, candidates, fps, width, height)
    backward, bw, _ = _associate_direction(image[::-1], confidence[::-1], candidates[::-1], fps, width, height)
    backward, bw = backward[::-1], bw[::-1]
    result, support = np.asarray(image, float).copy(), np.zeros_like(fw)
    conflicts = 0
    for side, wrist in enumerate([15, 16]):
        both = (fw[:, side] > 0) & (bw[:, side] > 0)
        gap = np.linalg.norm((forward[:, wrist, :2] - backward[:, wrist, :2]) * [width / height, 1], axis=1)
        agree = both & (gap < .025)
        conflicts += int(np.count_nonzero(both & ~agree))
        result[agree, wrist, :2] = (forward[agree, wrist, :2] + backward[agree, wrist, :2]) * .5
        support[agree, side] = np.minimum(fw[agree, side], bw[agree, side])
        # One-way evidence is used conservatively only with a visible body wrist.
        for observed, weights, other in [(forward, fw, bw), (backward, bw, fw)]:
            valid = (weights[:, side] > 0) & (other[:, side] == 0) & (confidence[:, wrist] >= .6)
            result[valid, wrist, :2] = (observed[valid, wrist, :2] + image[valid, wrist, :2]) * .5
            support[valid, side] = weights[valid, side] * .5
    report.update(method='bidirectional_wrist_agreement_v2', associated_wrist_samples=int((support > 0).sum()),
                  bidirectional_conflicts=conflicts)
    corrections = np.linalg.norm((result[:,[15,16],:2] - image[:,[15,16],:2]) * [width / height, 1], axis=-1)
    report['mean_image_correction'] = float(corrections[support > 0].mean()) if (support > 0).any() else 0.
    return result, support, report


def image_observation_weights(confidence, hand_support):
    """Separate fusion weights; NEVER overwrite raw body detector confidence."""
    body = np.asarray(confidence, float)
    support = np.asarray(hand_support, float)
    if support.shape != (len(body), 2) or not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
        raise ValueError('Invalid independent hand support')
    weights = body.copy()
    weights[:, [15, 16]] = np.maximum(weights[:, [15, 16]], support)
    return weights
