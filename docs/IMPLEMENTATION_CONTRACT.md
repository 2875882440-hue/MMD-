# Internal implementation contract — v0.6.0

Local-only app at http://127.0.0.1:8765; single serial worker; CPU MediaPipe.

## v0.6.0 experimental world-motion engine

- Upload accepts `engine=mediapipe|gvhmr_experimental`, default mediapipe; unknown values return 422, incomplete local GVHMR installation returns 409 before creating a job. Reprocess accepts optional engine Form field, otherwise preserves the original engine. Both create new jobs and preserve old input/output.
- GVHMR effective settings explicitly override sample_fps=30, max_seconds=32, root_mode=estimated_world_translation, feature_precision=fp16, keypoint_identity=temporal, smoothing=0 and all three MediaPipe refinement toggles=false. Frontend disables irrelevant controls and shows limits. Oversized duration is rejected, never silently truncated.
- Health adds `engines.gvhmr_experimental` containing installed/missing/device/fps/max_seconds/experimental/runtime_checked/character_retarget. File presence is not a CUDA runtime test. Legacy top-level CPU/features describe the default engine. No automatic downloads or fallback on inference failure.
- `motionbridge.gvhmr_pipeline.process_video` has the same callable contract as the CPU pipeline. Normalizes source to 30 FPS, runs six owned sequential worker processes in `.venv-gvhmr`, checks stage telemetry and neutral NPZ, and exports motion.json/quality.json/pose.mp4/overlay.mp4/pose_keypoints.json. Cancellation terminates and waits for the owned worker. App queue remains single-worker. Torch is not imported into production Python.
- Two-dimensional previews are corrected ViTPose COCO17 converted to BODY18, pixel coordinates in pose_keypoints.json. They are detector observations, not projections of the final 3D animation. Confidence in 3D remains unknown; detected_ratio=null. Quality includes telemetry and identity hypotheses, never accuracy percentages.
- New world mode bypasses all old fixed-root alignment and direction fitting. Viewer, BVH and Blender/FBX use the same 22-joint positions/rotations. Viewer fits the full trajectory with a fixed camera, adds no artificial ground offset, and initially faces the first predicted body heading.
- The inspected Nahida preset supports this mode through one fixed mean-leg-length scale and initial stage alignment; all subsequent pelvis deltas (including vertical motion) are transferred. It bypasses old direction cleanup, per-frame grounding and MediaPipe endpoint IK. Residual model jumps and foot/height drift are preserved and explicitly warned about. This is not general rig retargeting or measured contact recovery.
- Saved character audit checks every world root against expected scaled positions as well as finite bones, original source hash and packed textures. No angular speed limit is asserted for an unfiltered world model. `continuity_limit_applicable=false`, `continuity_passed=null` do not certify smooth motion. Blender/FBX report likewise uses null for an inapplicable limit.

## v0.5.1 additions (supersede older descriptions below)

- Projection optimization starts from the previous fitted directions and strengthens temporal regularization to reduce depth-branch switching. Per-frame maximum speed alone is not a jitter criterion.
- When enhancement, refinement and stabilization are enabled, `temporal.finish_motion` runs AFTER projection and source contact IK. A centered Gaussian matrix average (sigma 65ms), projected onto proper SO(3) rotations by SVD, smooths all source world tracks and root height; fixed-length FK is rebaked. Terminal hand rotations inherit final forearms because palm roll is not observed. `temporal_finish` is additive in motion and quality JSON, with method, duration and mean position adjustment. Existing schema 1.3 readers still consume baked tracks unchanged.
- Source contact/projection metrics describe their own stages before this final filter; they are not final endpoint accuracy or final slip. Smoothing can move previously locked endpoints, soften brief poses and change direction at sequence boundaries. It does not recover hidden geometry.
- Only motions carrying enabled `temporal_finish` activate the new target behavior: accumulate the entire target IK result, smooth rotations and root lift with a centered 45ms window, inherit corrected elbow orientation at neutral wrists, then bake. Target `mean_support_horizontal_step_m` is evaluated from these final matrices; pre-limit goal residual remains pre-filter. `target_temporal_finish` identifies this stage. Old saved motions retain their earlier export behavior.
- No new UI/API parameter is required. Existing stabilization/enhancement switches apply. New uploads and reprocess use 0.5.1; historical jobs are not rewritten. Large-file download and actual generated Blender output were checked through the local HTTP API.
- Temporal regression now compares saved Blender rig trajectories across versions: adjacent angular-velocity reversals above 2 degrees per frame, angular acceleration and position second differences. These supplement full-frame round-trip audits and visual samples; neither low jitter nor low file error establishes capture accuracy.

## v0.5 additions (supersede older descriptions below)

- Existing `enhance_motion` now also enables bidirectional wrist association, independent image observation weights, detail-aware image smoothing and image-constrained three-link arm fitting. No new upload field is required. Health adds `image_constrained_arms:true`. Disable enhancement to retain the prior body-only processing path; historical saved jobs are not upgraded implicitly.
- `landmarks_to_motion(..., hand_support=None)` accepts optional finite T×2 weights in [0,1]. These are geometric/temporal fusion support, NOT calibrated detector probabilities. Raw body confidence is preserved. New frame `observation_weights[J]` is separate from `confidence[J]`; arm IK and enhanced target fitting use it. Missing support defaults to zero. No hand-derived depth confidence is invented.
- Hand proposals are associated in forward and reverse time with up to 0.25s short tracks. Disagreement abstains. One-direction evidence requires visible body wrists and is weakened. This is not long-occlusion identity recovery. `hand_assistance` reports actual final correction, associated samples, forward ambiguity and bidirectional conflicts.
- `observations.npz` retains `world`, `image`, `confidence`, `hand_support`, adding `image_weights`, `raw_body_image`, `raw_world`, `assisted_image`. Raw arrays can contain NaNs from missed detections; processed arrays remain finite. `world` is body-confidence interpolation/smoothing; `image` uses independent weights and preserves some coherent fast observed motion. Cache schema is additive; old files remain readable.
- `projection.py` calibrates weak perspective from visible shoulder/hip anchors, then fits clavicle/upper-arm/forearm directions to observed 2D points with fixed segment lengths, a weak monocular depth prior and previous-direction regularization. This is a per-frame nonlinear fit with temporal prior, NOT a full sliding-window or anatomical whole-body optimizer. Insufficient anchors fall back; unsupported endpoints do not pull the arm. Corrected rotations are parallel-transported across frames, then speed-limited before final FK baking. Contact IK follows and preserves schema 1.3.
- New `projection_refinement` in motion/quality contains camera coverage, rate-limit count, mean projected elbow/wrist residual before/after in units of image height. The residual compares detector observations, not ground truth. Contact pelvis shift occurs afterward; metrics describe this stage. Rotation stability metrics describe the earlier stabilization stage; actual final speeds are audited by Blender export.
- Enhanced projected source motions activate target shoulder mapping and use the midpoint of the actual left/right eye bones as a face-height proxy for the source ear midpoint. The MMD external gaze control `両目` is deliberately excluded. `reference_landmarks` records chosen rest points and caveats; head bone tip is not a measured scalp landmark. This version does not add calibrated palm/sole geometry or interactive reference editing. Old source motions retain the earlier target profile behavior.
- Raw observation storage and manually labeled review files are never read as per-video corrections by production code. Source videos, old jobs and original Blender projects are retained. Manual screen-space review is approximate and is not 3D accuracy certification.

## v0.4 additions (supersede older defaults below)

- Upload adds explicitly parsed `enhance_motion="true"|"false"`, default true, invalid values 422 before creating a job. Reprocess enables it alongside refinement and stabilization. Disabling enhancement retains v0.3 behavior; disable all three switches for initial direction-only processing.
- Health adds `contact_ik:true` and `hand_assistance:bool` (optional local model readiness). `root_motion`, `fingers`, and arbitrary `custom_retarget` remain false.
- Optional official pinned Hand Landmarker runs on CPU on upper-body and wrist crops. Deduplicated candidates use conservative one-to-one wrist attachment and temporal prediction; nearly tied identities abstain. Only image wrists are corrected. Handedness classification scores are not physical accuracy or pose visibility. Original body confidence remains unchanged. Missing model produces an explicit warning and body-only fallback; corrupt model fails processing rather than silently claiming assistance.
- `observations.npz` adds `hand_support[T,2]`. Its `image` includes assisted wrists when enabled; it is not an untouched raw detector export. `world` is pre-refinement, as before. `hand_assistance` summary is stored in motion and quality JSON, with candidate/associated/skipped counts.
- Enhanced rotation stability permits higher speed only with consistent adjacent 3D rotations, body confidence and corroborating projected angular motion. Ordinary caps remain 360/720 deg/s; absolute enhanced caps are torso 720 / limbs 2160. This is a heuristic, not a guarantee that a confident estimate is correct. Stored cap fields describe maxima. Five-frame robust filtering can still change timing.
- With enhancement AND valid refinement, schema 1.2 adds `foot_contacts`: T×2 blend weights, exclusive-end contact segments, locked counts, pre/post support horizontal step, pelvis adjustment and rate-limited counts. Contact hypotheses require visible confident low/slow feet for >=0.18s, break on large displacement, and blend over ~0.12s. Raised, fast or uncertain feet are released; no mandatory grounded foot. Per-segment median anchors, fixed-length leg IK, <=0.12m vertical pelvis correction and final leg angular caps are applied. Some anchors are unreachable; rate limits can weaken exact locks. No scene translation or measured jump recovery is inferred.
- Contact output `root_mode="in_place_contact_vertical"`; horizontal root remains in place, vertical root can change. Positions and baked rotations remain FK-consistent for web, BVH, Blender and FBX. Old schemas and jobs are readable unchanged. If refinement is unavailable, `foot_contacts.enabled=false` with reason.
- Target Nahida enhancement reads source settings and contact weights. Arm goals blend ordinary length-scaled arm motion with head-relative and close-bimanual chest-relative positions. Leg contact anchors are fitted in the target proportions, with D-bone counterparts adjusted. Fixed bone lengths and final world angular caps remain enforced. Reports distinguish target-goal residual BEFORE rate limiting from actual final support drift; neither is capture accuracy. Finger/face/cloth physics are unchanged.
- `quality.json` stores hand and contact diagnostics (contact weights remain in motion JSON). Existing endpoint_fit metrics precede stabilization and contact IK; they are not final endpoint errors. Full-frame export validation uses the baked final tracks and checks continuity against configured maxima.
- Installation: `scripts/download_hand_model.py` verifies pinned official version 1 using exact length, GCS ETag MD5, SHA-256 record and ZIP member CRC; Setup invokes it optionally. No model auto-download occurs during video processing.

## API

- GET /api/health → {version, model_ready, blender_ready, blender_path, device:"CPU", limits:{max_seconds:60,max_upload_mb:300}, features:{...}}
- GET /api/characters → {characters:[{id,label,ready,reason}]}. The only implemented preset is `nahida`; readiness requires Blender plus both locally configured scene files. Paths are configured under `config.json → characters.nahida.{scene_path,weights_scene_path}`, never supplied by browser requests.
- GET /api/jobs → {jobs:[job,...]} newest first
- POST /api/jobs multipart: file, smoothing (float 0..1, default .55), sample_fps (int 12..30, default 24), model ("lite" or "full", default "full"), exports_3d (bool, frontend explicitly sends true/false), refine_motion ("true"/"false", default "true"). Returns job (202). The refinement field is explicitly parsed so omitted fields remain default-on with current FastAPI checkbox handling.
- Additional upload field `stabilize_rotations` is explicitly parsed `"true"/"false"`, default `"true"`, independently switchable in the UI. Invalid values return 422 before job creation.
- GET /api/jobs/{id} → job
- POST /api/jobs/{id}/cancel → job
- POST /api/jobs/{id}/export-blender → job (202); run Blender after initial completion, preserving existing results.
- POST /api/jobs/{id}/reprocess → NEW job (202); copy the original upload, preserve processing settings, enable current refinement, keep the old job and artifacts unchanged. Reject running/demo jobs and missing originals with 409. New job records reprocessed_from and software_version. This runs inference again; export-blender alone does not upgrade historical poses.
- Reprocessing in v0.3.0 enables both refinement and rotation stabilization. Reprocess and export-blender reject character jobs with 409; return to their source job instead.
- POST /api/jobs/{id}/retarget multipart `character=nahida` → NEW job (202), `kind:"retarget"`, `source_job_id`, `character`. Requires a completed non-character source with motion.json and a ready preset; unknown presets 422, missing setup/source 409, full shared queue 429. Copies source motion and optional FBX into private input; generates a reference FBX if absent. Original job/files/scenes remain unchanged. Uses the same serial worker, cancellation, restart-interruption handling and job status API as capture jobs.
- GET /api/jobs/{id}/motion → motion JSON when ready
- GET /api/jobs/{id}/files/{filename} → safe file response (video byte ranges supported)
- GET /api/jobs/{id}/download → ZIP of outputs, excludes source video
- POST /api/demo → job with synthetic motion explicitly marked demo; exercises previews and exports WITHOUT claiming video inference.

job = {id,name,status:"queued|processing|completed|failed|cancelled",stage,progress:0..100,created_at,updated_at,error:null|string,warnings:[],settings:{...},meta:{fps,frames,duration,detected_ratio},files:[{name,url,size,label}],is_demo:bool}
Source video available as file source.mp4 only when normalized; uploaded source stored under input with safe extension.
New jobs record software_version. Legacy jobs may omit it. Health features includes motion_refinement:true.
Health additionally advertises rotation_stabilization:true and character_presets:true; custom_retarget remains false (no arbitrary uploaded rig support). Character outputs: Nahida_MotionBridge_Dance.blend, character_preview.png, retarget_report.json, saved_project_audit.json, character_export.log, character_audit.log. Parent source motion.json and normalized source.mp4 are copied for source preview; the UI explicitly labels these as SOURCE motion, not the custom character. The target appears in a separate still preview and Blender download. ZIP excludes source.mp4 and private input files.
files are updated after processing; UI locates pose.mp4, overlay.mp4, motion.bvh, motion.json, motion.blend, motion.fbx, mannequin.mp4 (optional) by name.

## Motion file

JSON object with schema_version:"1.0", fps:float, num_frames:int, joint_names:[string], parents:[int], rest_positions:[[x,y,z],...], frames:[{time:float,positions:[[x,y,z],...],confidence:[float,...]}], source:{width,height,fps,duration}, settings:{...}, warnings:[], coordinate_system:"Y_UP_RIGHT_HANDED_METERS", root_mode:"in_place", is_demo:bool.
Positions in meters, Y up, hips centered at origin. rest_positions are T-pose absolute joint positions, same origin.
Optional contacts/confidence fields must not be required by the UI. Visualize pelvis at rest-height for ground display (do not change exports implicitly).

v0.2 additive fields: solver (direction_v1 or endpoint_ik_v2), positions_are_solved (true for corrected videos), refinement (enabled, solver, image_alignment and endpoint_fit diagnostics). Legacy JSON without these fields retains the original direction solver. Refined frames contain fixed-length FK joint positions shared by the browser, BVH and Blender; observation confidences are unchanged. Display floor offset is -min(rest Y)+0.045 in browser and Blender. Joint center positions match, while the browser and Blender use different simple meshes.

v0.3 stabilized documents use schema_version:"1.1" and add world_rotations[T,J,3,3] (source-coordinate proper rotation matrices) and rotation_stabilization diagnostics. Positions are rebaked from these rotations and fixed rest offsets, with unchanged observation confidence. solve_rotations validates shape/finiteness, orthogonality/determinant, and FK agreement with stored positions, then uses the tracks directly; it must not re-solve or filter them again. Malformed/stale tracks fail explicitly. Schema 1.0 without tracks retains its previous solver behavior.

Stabilization uses parallel transport for single-child segments, a centered 5-frame quaternion medoid, and angular speed caps of 360 deg/s for torso/head and 720 deg/s for limbs. It can change joint targets, attenuate fast movement and introduce lag when a speed cap engages. It does not infer true axial roll, floor contact or camera/scene translation. Quality JSON includes rotation_stabilization. Existing endpoint-fit diagnostics describe the pre-stabilization IK stage, not final position error or capture accuracy.

Refinement fits a torso-calibrated weak-perspective image-plane prior, preserves estimated depth, estimates bone lengths from confident frames, and fits reachable wrist/ankle targets with two-bone IK. It does not restore metric scene translation or implement foot locking, detailed finger poses, arbitrary rig retargeting or anatomical joint-angle limits. Image alignment can fall back when torso calibration is unavailable. Diagnostics measure consistency with estimated observations, not motion accuracy against ground truth.

observations.npz contains gap-filled/smoothed world[T,33,3], image[T,33,3] and original confidence[T,33] arrays before refinement. Load with allow_pickle=False. This cache is a separate downloadable output; never treat it as corrected FK or ground truth. It does not exist in old tasks. quality.json also includes refinement diagnostics.

## Python interfaces

motionbridge.pipeline.process_video(input_path:Path, output_dir:Path, settings:dict, progress:callable, cancelled:callable) -> dict
progress(percent:float, stage:str); cancelled()->bool. Throws RuntimeError on unusable input; InterruptedError on cancellation. Produces source.mp4 (normalized, capped duration enforced), pose.mp4 (OpenPose BODY-18 layout), overlay.mp4, motion.json, quality.json. Returns {motion:dict,meta:dict,warnings:list}. Model files are in project/models, names pose_landmarker_lite.task / pose_landmarker_full.task. No automatic model downloading inside processing. Low-confidence frames must not silently become good measurements. Must reject >60 sec, zero usable poses, non-video.

motionbridge.exporters.export_bvh(motion:dict, output_path:Path) -> None
motionbridge.exporters.export_blender(motion_path:Path, output_dir:Path, blender_path:str, progress:callable|None=None, cancelled:callable|None=None) -> dict
Exports motion.blend, motion.fbx plus blender_verification.json. Blender script consumes motion.json and uses shared rotation solver via project sys.path. Built-in simple rigged mannequin, not arbitrary character auto-rig. Exporter owns skeleton names, rest pose and conversion helper:
motionbridge.exporters.landmarks_to_motion(world_landmarks:np.ndarray[T,33,3], image_landmarks:np.ndarray[T,33,3], confidences:np.ndarray[T,33], fps:float, source:dict, settings:dict) -> dict
Input MediaPipe coordinates (X image-right, Y down, Z away), already gap-filled and smoothed; helper converts [x,-y,-z]. Confidence preserved. Body-only approximate 3D, fixed root, no face/finger capture.

motionbridge.stability.stabilize_motion(motion:dict) -> NEW dict. Requires SciPy during processing only; baked tracks can be read with NumPy in Blender. Idempotent for valid stabilized files.

motionbridge.retarget.export_character(root, blender, profile, motion_path, output, cancelled, progress) -> audit dict. Calls retarget_nahida.py and then audit_retarget_review.py using --disable-autoexec / --python-exit-code / CPU 4 threads. Process timeout 600 seconds per stage; cancellation terminates only the launched worker. Completion requires successful process exits, a passed saved-project audit, and both valid target .blend and preview PNG. New stabilized source tracks bypass the script's legacy cleanup to avoid filtering twice. Existing old motions use its legacy cleanup before target mapping.

Blender/FBX verification now evaluates every frame's joint positions and deformation rotations and checks speed limits for stabilized motions. It supplements the existing keyframe/skin/sampled round-trip checks; it is not motion accuracy against the uploaded video.

Frontend owns only motionbridge/static/ (index.html, app.js, styles.css), may use local vendor/three.module.js. Chinese UI, accessible controls, honest error/cancel states, no fake capture quality. Page title MotionBridge. Root will copy vendor. No npm build required.

## Compatibility

### v0.6.3 unilateral foot-floor collision

With GVHMR and `world_foot_contacts=true`, processing now runs `constrain_world_floor` AFTER the unchanged world contact refinement. This default-on path applies to every new upload and reprocess, not to video names or timestamps. MediaPipe processing is unchanged. Disabling the existing world-contact switch also disables this new stage. `motion_pre_collision.json` is the downloadable result immediately before collision constraints (after selected denoising and contact refinement); `motion_raw.json` keeps its prior meaning.

Source constraints use conservative bounds of the built-in mannequin's foot/toe spheres and ellipsoids against the assumed flat Y=0 model reference, with 1mm clearance. Only penetrating feet receive vertical endpoint IK with fixed leg lengths. Unreachable residual penetration receives a nonnegative, 0.35m/s Lipschitz root lift envelope. No foot is pulled downward; root X/Z and root/upper-body rotations remain unchanged. Lower-limb rotations and, if needed, root Y may change. `floor_collision` in motion/quality records the assumed plane, geometry proxy, before/after minima, corrected samples and every root lift. Invalid inputs and failed projection are explicit errors. Baked positions/rotations remain shared by browser/BVH/Blender/FBX.

For a newly constrained source, the configured Nahida profile evaluates actual foot-weighted mesh vertices after its existing target IK and sole fitting. It tests output frames and midpoints, then applies only the positive residual root lift needed for those samples, with the same velocity-bounded envelope. This profile depends on the configured rig and topology-preserving skinning. Empty foot selection or changed topology fails; arbitrary imported rigs are not supported. The expected root trajectory in the saved-project audit includes explicit collision offsets, not an unrecorded pose edit.

`retarget_report.target_mesh_floor_collision` describes evaluated mesh sampling and lift; `saved_project_audit.foot_floor_collision` independently reopens the saved project and verifies all frames plus half-frames. Blender export verification additionally evaluates foot-weighted geometry in the ACTUALLY REIMPORTED FBX at every output frame. Below-floor geometry makes these checks fail. The tolerance is 1e-5m. These checks do not certify arbitrary subframes, full-body/cloth self-collision, terrain, or capture accuracy.

### v0.6.2 additive world-motion finishing

The existing `world_arm_correction` switch now controls BOTH its unchanged pre-inference 2D identity/jump-point correction and new post-inference local rotation denoising. The latter runs before contacts, filters shoulder/arm/forearm rotations in parent coordinates using a centered 50ms SO(3) window, and attenuates filtering for coherent fast rotation. It does not filter root/torso/leg tracks or impose a global speed cap. `world_stability` in motion/quality reports identifies the method and changes. Root translation and rotation are bitwise preserved by this step; short gestures may soften. Terminal wrist local rotations follow the filtered forearm hierarchy. Existing controls and default MediaPipe processing remain available and unchanged.

`motion_raw.json` remains the model output after selected 2D correction, now before BOTH local denoising and contact correction. Do not describe it as untreated observations. Disabling the arm switch disables the new denoising as well. No new HTTP form fields are required.

`world_foot_contacts` retains v0.6.1 support detection and slow source-height correction, and adds bounded height adjustment to stationary ankle anchors using the same toe-plane estimate. New optional `foot_contacts.ground_weights[T,2]` is a separate image-supported ground-height hypothesis that permits horizontal foot travel. It is not observed confidence and is not used for horizontal locking.

The Nahida profile uses `ground_weights`, when present, for target-only sole baseline fitting after leg IK: vertical offset bounded to 12% target leg length, velocity bounded to 30% leg length per second, short support fade, long unsupported intervals return to zero. Horizontal root path, facing, source capture and arm orientations are unchanged by this target fit. The new optional target report `target_ground_fit` stores every vertical offset; expected root positions in the structural audit include that explicit correction. Older motion files without ground_weights retain their old target path. No universal rig support or measured floor truth is implied.

Stage telemetry in quality.json includes `reused_from_cache`; preflight always runs. Cached-stage timings/VRAM are historic measurements and must not be reported as fresh inference measurements. Same-input regression may reuse unchanged v0.6.1 model stages after hash/config checks, but still runs the actual current postprocessing, previews and Blender/FBX export.

### v0.6.1 GVHMR corrections

Upload adds `world_arm_correction` and `world_foot_contacts` multipart strings, each `"true"|"false"`, default true; invalid values return 422 before job creation. Reprocess accepts optional versions of both fields: omission preserves prior values (legacy missing fields default true), explicit values override for the NEW job. The UI sends its current selections on both upload and reprocess. MediaPipe behavior is unchanged; these switches only affect GVHMR.

GVHMR effective settings use `keypoint_identity=temporal_local` when arm correction is enabled, otherwise `temporal`. The local method runs after whole-body identity repair. It separately pairs elbows/wrists, repairs only short isolated excursions, and reduces interpolated confidence. It does not impose blanket angular speed limits. `quality.json.identity.arms` records decisions.

`motion_raw.json` is downloadable model output BEFORE world contact refinement, but AFTER whichever arm correction was selected for inference. It is not untouched raw ViTPose or ground truth. Optional `world_contact_refinement` reports estimated floor, contact counts, height correction, bounded-foot adjustments and same-support-frame diagnostics. Optional `foot_contacts` contains weights/segments for target IK. Schema 1.1 and fixed-length FK remain valid. `refine_world_contacts` returns a new dictionary, is idempotent, preserves input root X/Z and root rotation, and modifies slow Y drift plus leg rotations. No reliable support means no fabricated floor correction. Flight frames receive only the slow root-height offset, not forced foot locking; small jumps can still be misclassified.

Nahida world mode consumes these weights for bounded target leg IK, with proportional root transfer. Original in-place/MediaPipe pipeline is unchanged. No generic rig auto-retargeting is added. Complete outputs retain the same frames across browser/BVH/Blender; structural conversion verification remains separate from motion accuracy.

Explicitly resumed internal GVHMR processing always runs preflight input/settings-hash validation before reusing successful stages with all required nonempty outputs. Failed or incomplete stages rerun. This is not an automatic HTTP resume feature; server restart still marks interrupted jobs failed and normal reprocess creates a new job.

### GVHMR file adapter (used by the explicit experimental HTTP engine)

`motionbridge.global_motion.gvhmr_to_motion(path:Path, source:dict|None) -> dict` reads a non-pickled NPZ produced by `scripts/run_gvhmr_probe.py`. Required fields: world_body_pose[T,63], world_global_orient[T,3], world_transl[T,3], joints_world[T,22,3], rest_joints[T,22,3], parents[22], scalar fps. Coordinates retain upstream right-handed Y-up meters. Rest shape must be constant; pelvis must equal native rest pelvis plus transl; shared FK must reproduce all joints. Violations fail instead of re-solving poses.

Output remains schema 1.1 with the full topologically ordered 22-joint SMPL-X body hierarchy, an additional Spine2, baked world_rotations, solver=gvhmr_smplx_world and root_mode=estimated_world_translation. It bypasses MediaPipe image alignment, root resetting and direction fitting. Confidence is unknown (confidence_available=false, zero placeholders), never a measured score. Blender uses zero extra floor shift for this root mode and frames the whole estimated trajectory. Only the inspected Nahida profile has a separate world-root adapter; arbitrary rigs remain unsupported.

Schema 1.0 remains valid. Disable BOTH refinement and stabilization for legacy direction_v1 output without corrected positions. New capture corrections are generic: there are no clip names, dance timestamps or per-user pose edits in capture/solver code. Character mapping is deliberately preset-specific. Models remain Lite/Full; Heavy is not a released selectable option.

### v0.6.4 arm/torso proxy avoidance

Upload adds `world_body_collision: "true"|"false"` (default true); reprocess accepts the same optional form field, inheriting stored settings when omitted. Invalid values return 422. Effective GVHMR settings default old jobs without this field to true. MediaPipe ignores this option. The browser sends its independent “手臂与躯干避碰” toggle on upload and reprocess; it is independent of arm denoising and ground toggles.

`body_collision.constrain_body_collision(motion)` runs after existing optional world contact/floor stages. It preserves root position/orientation and all non-arm rotations, uses fixed-length arms and torso-relative corrections bounded to 25 degrees per segment. It smooths correction tracks rather than the input animation. `motion_pre_body_collision.json` retains this step's input and is downloadable. Final motion and quality JSON contain `body_collision`, including per-side proxy deficits, unresolved frame counts (>1mm proxy deficit), measured maximum adjustment and configured limit. These are geometric proxy statistics, never accuracy or true mesh penetration measurements. Idempotence avoids double application to final motion documents; reprocessing starts again from retained inference output.

The configured Nahida profile applies the same NumPy solver to target proportions before keyframe baking, then retains the v0.6.3 evaluated shoe-floor correction. `retarget_report.target_body_collision` records target correction metrics. Reopening the saved project recomputes its arm/torso proxy; `saved_project_audit.body_collision.saved_solver_consistent` checks baking consistency, NOT collision absence. Remaining overlaps are allowed and explicitly reported because a bounded edit takes precedence over forcing an extreme pose. No head, inter-arm, mesh, garment, cloth dynamics, arbitrary rig or terrain collision is implemented here.

### v0.6.5 arm/head proxy avoidance

Upload adds `world_head_collision: "true"|"false"` (default true). Reprocess accepts the optional field, preserving a stored choice or defaulting missing legacy settings to true. Invalid values return 422. This is independent of arm/torso and ground toggles and applies only to GVHMR. The browser sends “手臂与头脸避碰” for both upload and reprocess.

`head_collision.constrain_head_collision` follows the optional torso stage and writes downloadable `motion_pre_head_collision.json` before solving. Source head geometry matches the built-in mannequin head ellipsoid; source root/head/torso/legs are unchanged. The shared NumPy `avoid_arm_head` solver preserves arm bone lengths, caps each additional segment correction at 20 degrees, and backtracks any correction that worsens the pre-existing torso proxy minimum or aggregate penetration, or worsens the head proxy aggregate penetration, with a 1e-6m numerical tolerance. Backoff weights are smoothed downward. Remaining contacts, rejected frames and actual correction sizes are reported under `head_collision.sides` in motion and quality JSON. These are geometric proxy statistics, not capture accuracy or actual mesh intersection depths.

Nahida calibrates a rest AABB-inscribed ellipsoid from >50% head-weighted vertices in the configured mesh/material `Nahida_mesh/颜`. Missing or degenerate geometry fails clearly rather than inventing a calibration. Head pose transforms animate this proxy. Hair, accessories, fingers and non-ellipsoidal parts of the face are not covered. The report stores calibration in `target_head_collision.profile`; the same 20-degree solver and torso safeguard run before keyframe baking and shoe-floor correction. Saved-project audit recomputes head/torso proxies at integer frames, using the head stage's final torso metric when enabled. The earlier body-stage report retains its own stage result and is not overwritten. Shoe verification still covers integer/half frames. A consistency pass is not a zero-intersection claim. Configured Nahida is still the only supported target profile; arbitrary-character auto-calibration is not implemented.

### v0.6.6 image arm refinement and configured surface clearance

For GVHMR, `world_arm_correction=true` additionally runs camera-aware shoulder-relative elbow/wrist fitting after upper-body denoising and before contacts/collisions. It uses retained camera/world SMPL alignment and reliable 2D observations, fixed arm lengths, a 12-degree segment correction limit and 6cm wrist limit. A confidence envelope fades corrections around unreliable observations. Root, torso and leg tracks are unchanged by this stage. Invalid/mismatched camera inputs fail explicitly. `motion_pre_image_alignment.json` is downloadable; `image_arm_refinement` in motion/quality describes observation agreement, NOT capture accuracy. Reprocessing runs this for all new material without filename/frame-specific rules. MediaPipe is unchanged.

`POST /api/jobs/{job_id}/retarget` adds `surface_clearance: "true"|"false"` (default true). Invalid values return 422. The value is retained in the new character job's settings. Only the configured Nahida target is supported; arbitrary-character mapping remains deferred. This switch controls two target stages:

- Rigid front-hair/accessory polygons with head weight >95% repel sampled arm segments with an 8-degree correction cap. Existing head/torso safeguards take priority. The rest geometry and binding of these ornaments are preserved; the audit compares their geometry hash. Unsigned surface distances do not certify volume exclusion.
- Flexible long hair/garments are projected against animated torso/limb capsules, preserving original rest attachments and capping per-vertex world movement at 3cm. Corrections are smoothed and reprojected, then inverse linear skinning embeds one absolute shape per output frame in the new Blender file. Shared toon outlines reuse the same mesh/shape data; incompatible skinning fails. Singular skin inverses are skipped and reported. Existing source files/caches are never modified.

`retarget_report.target_accessory_collision` and `surface_clearance` retain before/after proxy statistics and remaining overlap. The saved-project audit reopens actual evaluated garment vertices and checks all integer frames plus existing integer/half-frame shoe checks. These validate embedding and proxy consistency, not collision-free geometry. Shape corrections live in the target `.blend`; the source skeleton FBX does not contain Nahida garment shapes. No cloth simulation, cloth self-collision, cloth-floor constraint, detailed finger capture or new model training is provided. New uploads benefit through the same processing and target export paths; old jobs are preserved and must be reprocessed/regenerated to adopt changes.

### v0.6.7 topology preservation for flexible surfaces

The same `surface_clearance` target switch now also runs `surface_topology.regularize_surface_tracks` after the existing capsule projection and temporal correction, before inverse-skinning/baking. No new form field, source solver, source bone motion or character mapping is introduced. New character generation applies it to new material; old jobs stay intact.

The solver proposes mesh-edge length relaxation toward the current uncorrected skinned pose, with a 10% + 0.5mm deadband. It smooths only the additional correction using a frame-rate-scaled temporal window. Extra movement is capped at 1cm, keeping the prior 3cm total cap (0.5 micrometer numeric tolerance). Per-vertex backtracking prevents increased capsule penetration beyond 0.1 micrometer and prevents new >1mm samples. Per-frame backtracking rejects increased maximum edge-length change, sum of squared edge-length changes or number of edges changed by >5mm. These priorities can retain significant deformation; no collision-free or strain-free claim is made.

Each flexible object's `topology` report records proposal/backoff frames, extra displacement, and before/after edge metrics. Saved-project audit independently evaluates the saved Basis through its rig and compares actual edge metrics with the solver. Corrections remain embedded absolute shapes, shared with compatible toon outlines. The constrained solve can add small cloth motion even though skeleton animation is untouched; temporal diagnostics and visual review must accompany geometry metrics. It is not physical cloth, mesh self-collision, recognition improvement or arbitrary-character support. Surface-enabled character export allows 900 seconds for baking; cancellation remains active.


### v0.6.8 coherent surface contact

The existing `surface_clearance` switch additionally runs `surface_directed.coherent_surface_tracks` after v067 topology correction. Recognition, rig animation, feet, source FBX and form fields are unchanged. New character generations use the new path; old jobs are not overwritten. Source FBX remains a skeleton export and does not contain target corrective shapes.

Rest triangle orientation supplies a welded surface-side direction field. An 8cm bounded candidate is expanded in time using a contact-magnitude envelope, with frame-rate-scaled approach/release smoothing. Capsule projection and posed-edge relaxation then allow up to 12cm total displacement for this metre-scaled target profile. This is a maximum budget, not a uniform expansion. Connected welded components are accepted or retained for the entire sequence, so fallback does not split edges or alternate per frame.

Per-component and combined gates prevent new >1mm contact samples, larger maximum penetration, increased peak/squared-sum/>5mm edge-length changes, and increased final vertex acceleration P99/peak beyond a 0.1m/s^2 small-amplitude tolerance. `gate_version=actual_vertex_motion_v2` checks rendered-geometry motion, not correction magnitude alone. Both sets of temporal metrics are reported; correction P99 can rise while final geometry motion improves. These statistics do not guarantee every frame improves or certify physical correctness.

`surface_clearance.objects[].coherent_contact` adds before/candidate/after metrics, component decisions and rejection reasons. `baseline_topology` retains the v067 pass; `topology` final edge fields reflect the chosen embedded shapes. Saved audit additionally checks actual displacement against reported movement and the 12cm budget, and final vertex acceleration P99/peak against the solver. Fixed Nahida mapping, capsule proxy limitations and nonphysical cloth behavior remain.

`retarget.character_timeouts` uses the input motion's positive frame count N. Surface-enabled baking allows max(1800, 300 + 2*N) seconds; disabled baking keeps 600 seconds. Saved-project audit allows max(600, ceil(120 + 0.5*N)) seconds. Both calls retain the original cancellation callback and terminate only their own Blender subprocess on cancellation/timeout. The measured 901-frame bake took about 1208 seconds; the new budget prevents longer supported clips inheriting a fixed 1800-second ceiling, but is not a runtime guarantee on every machine. No HTTP fields change.

### v0.6.9 actual triangle contact

Saved geometry verification now uses all flexible vertices at every integer frame, with a 10um Euclidean error limit against `surface_solver_reference.npz` (float32 pre-embedding targets, SHA in report). Missing/corrupt references fail. This sidecar is for audit, not playback; corrections remain embedded in the standalone .blend, and the ZIP includes the sidecar. Already generated v069 files without that snapshot use explicitly labelled `legacy_saved_absolute_key_lbs` verification against independently skinned stored absolute keys, with `original_solver_track_available=false`. This does not reconstruct the lost producer reference. Contact threshold/crossing aggregate differences remain diagnostic; actual saved contacts are recomputed, not inferred from producer counts. Other displacement/temporal/edge/capsule/foot safeguards remain hard checks.

`POST /api/jobs/{job_id}/retarget/retry-validation` returns 202 for a failed/cancelled configured-character task with a complete, matching checkpoint; rejects incompatible status/source records/configuration/input hashes with 409 and a full queue with 429. It reuses the same job ID and .blend, stores previous error/history, and schedules only saved-project audit (and a missing preview), never rebakes or recopies the motion. Old audit JSON/log are copied under outputs/validation_history. The .blend SHA must remain unchanged; failed re-audits cannot promote the task. The frontend exposes “重试工程验证” only when a failed/cancelled character task has a saved .blend and report. Existing “生成角色工程” retains independent-job behavior.

The same `surface_clearance` switch appends `blender_triangle_contacts.mesh_contact_tracks` after v068, before inverse-skin corrective shape embedding. No new request fields or character mappings. It uses the configured Nahida mesh/material triangles (`颜`, `前髮`, `头饰`, `体`, `短裤`, `体饰`, `肌2`) and welded flexible components. Rigid ornaments are colliders and keep their geometry; source recognition and rig animation are unchanged.

Welded vertices, edge midpoints and triangle centers query nearest current faces. Rest-overlap allowance is computed against the currently queried triangle; ambiguous boundary projections and degenerate normals are skipped. Rest signed order constrains different flexible components, excluding ambiguous original attachments. Body queries use a 3cm radius, layers 1.5cm, requested gap 1.5mm. This is a local surface constraint, not a watertight-volume claim.

Proposals are sampled at `stride=max(1,round(fps/10))`, including endpoints, then interpolated/smoothed at full rate. Incremental edge regularization and capsule support planes precede a whole-sequence spatial support ceiling per welded vertex. These weights stay constant over time and diffuse only downward. Whole-component strength backoff and full-frame checks follow; original >1mm capsule-contact and actual-vertex P99/peak acceleration guards remain. Total movement remains 12cm, extra movement versus v068 <=3cm. Each edge allows <=0.5mm extra absolute length error, plus 5um numerical tolerance. Aggregate edge energy is reported and may increase within this explicit per-edge budget; do not imply v068's zero energy-increase rule remains identical.

`surface_clearance.objects[].triangle_contact` records before/after local contact metrics, proper crossing counts, support weights, strength decisions, rejected components and final motion metrics. `triangle_pair_method=proper_noncoplanar_crossing_v2`: BVH only supplies candidates; vectorized bidirectional edge/face tests confirm interior straddling with >10um separation. Coplanar/grazing/shared-boundary touches do not count as proper crossings. Original rest crossings are excluded from additional-crossing counts. These are integer-frame diagnostics, not continuous collision certification or capture accuracy.

Body guards retain local sample/maximum/sum/energy and proper-crossing protections. Layer pairs guard total proper crossings, maximum local depth, summed depth and squared depth energy; the 1mm sample count and single-frame pair peak remain diagnostic, since shallow contact redistribution can change those counts. Mixtures after component fallback are checked again. Saved-project audit evaluates actual saved garment vertices against actual saved collision geometry; all-point reference error is the export-fidelity gate, and threshold aggregate differences remain diagnostic, alongside the prior hard shape/edge/shoe checks. Earlier `baseline_topology` and `coherent_contact` remain stage records; final `topology`/`triangle_contact.after_motion` describe the chosen output.

Timeouts now scale for added geometry work: surface bake max(2400,600+4*N) seconds; disabled bake 600 seconds; audit max(900,ceil(180+N)) seconds. Cancellation still terminates only the job's own subprocess. No cloth dynamics, same-component self-collision, cloth-floor constraint, arbitrary-avatar mapping or recognition-model improvement is introduced. Research caches never enter the production path.


## 0.7.0-dev.1 隔离开发与模型物理检查
开发实例8766，默认任务目录data/physics-jobs，工程外任务路径拒绝。health新增instance_id/project_root/pid和secondary_motion=false、secondary_profile_inspection=true，原动作API保留。当前没有衣发仿真API和网页开关。scripts/inspect_secondary_model.py --profile rin --output <本副本output目录内JSON> 仅读取原生PMX数据；模型路径必须在当前副本内，主骨骼保护必需，enabled=true明确拒绝。输出包含原生刚体/关节参数、候选名称分类、主骨骼动态冲突和局限，不进行Blender仿真、单位转换或动作改写。


## 0.7.0-dev.2 角色物理配置接口（CLI，不是仿真API）

`profiles/secondary_motion/<id>.json`的schema 2包含model.path/scale/sha256、primary_bones、candidate_name_tokens、protected_surface_policy=exclude_and_report、regions（唯一id/role/materials/deformation_policy）、bindings（blend_asset/armature/mesh/bone_aliases）及floor（当前Z=0m、组15）。enabled只接受false。schema 1仍可用旧inspect CLI读取，完整prepare要求schema 2。

`scripts/prepare_secondary_model.py --profile rin --output-dir <新子目录>`：只允许本开发根output/secondary_motion内的新子目录。写model-plan.json及summary.json；不读视频、不运行Blender/求解、不修改源资产。模型指纹、分区/别名/单位/依赖引用非法时失败，不输出半成品配置。plan包含原刚体关键字段和Blender参数、全部三角面/蒙皮、分区顶点及连续pin_weights、显式排除列表、锚点连通分量、地面掩码风险。pin=1表示固定，pin=0表示可活动；这是蒙皮推导的配置，不是布料求解质量保证。

`scripts/bind_secondary_model_blender.py --plan <plan> --output-dir <新子目录> [--scene <同角色工程>] [--preview]`：须Blender background、--disable-autoexec、--python-exit-code。精确校验材质槽/面/顶点/静态位置/原生骨/原蒙皮，拒绝源已启用刚体世界；仅增加MBP_*_region、MBP_*_pin诊断组和只读身份属性，在新目录保存configured-model.blend及binding-validation.json。--scene仅用于动作保持回归，不是按视频生成配置。保存重开后核验原曲线/原蒙皮/形态键、受保护骨骼所有帧世界姿态及新组权重；--preview在保存后临时染色，不改变保存工程的原材质。

单位约定：position=(x,z,y)*scale；rotation=(-rx,-rz,-ry)，Euler YXZ；盒半边长交换YZ，胶囊radius/柱高不交换；角限位取负后上下互换。质量/阻尼保持作者值，弹簧系数只匹配MMD导入，不保证SI校准。mask置位代表排除，接触资格须双方允许；实际碰撞/关节附加排除、地面与仿真未实现，solver_ready=false。

当前没有衣发HTTP/API/UI启用入口；health.secondary_motion仍false。现有主动作接口、默认算法与参数保持。角色配置可复用同角色新视频；不保证任意角色/格式直接匹配。


## 0.7.0-dev.3 实验发束CLI（不改变主动作API）

`profiles/secondary_motion/rin.json.hair_dynamics`：method=spring_bones_v1，显式骨名包含/排除、body_collider_ids、explicit_hair_body_override、substeps、collision_iterations、warmup_seconds、drag_per_second、rest_acceleration、gravity_m_s2、驱动加速度滤波/边界、max_joint_speed_rad_s、contact_joint_speed_rad_s、inertia_strength、collision_margin_m、strand_radius_scale。所有源PMX参数仍保留。enabled=false仍表示准备器/工作台不自动启用，实验求解只走独立命令。

`scripts/simulate_secondary_hair.py --profile rin --scene <本副本blend> --output-dir <新子目录> --enable-hair`：显式启用；路径限制在本副本，输出只能output/secondary_motion下的新目录。启动独立Blender --disable-autoexec，失败保留日志/目录；1小时超时或Ctrl+C仅结束本次进程。不导入motionbridge.app、不重算识别/身体动作。已有发束曲线或重复MBH约束明确拒绝，尚无叠加/重算策略。

输出before.blend、after.blend（相同相机）、model-plan/profile/hair-nodes JSON、hair-simulation.npz和physics-validation.json。原Action/蒙皮/形态键保持；84个新旋转目标Action及COPY_ROTATION仅施加给安全发束。预热max tip speed>=0.04m/s、原动作/48骨矩阵变化、保存旋转/骨长/角限位超阈或发束无可测运动即失败。核心验收passed不等于干净碰撞/无拉伸。

`scripts/render_secondary_hair_compare.py --pair-dir <输出目录>`须Blender执行，可选--frames及按源SHA核验的--resume；逐帧前/后使用相同保存相机/渲染设置。`scripts/compose_secondary_hair_compare.py --pair-dir <输出目录>`按实际fps合成无音频对照并全解码；拒绝旧视频覆盖。`scripts/audit_secondary_hair_surface.py`独立重开前后工程核对1799主矩阵采样/原曲线/原蒙皮形态键，采样真实表面及解析代理、边长和速度；不是真实三角面连续碰撞证明。

大幅水平移动的比较应先用`scripts/frame_secondary_hair_compare.py --pair-dir <输出目录> [--follow-bone 腰]`生成comparison子目录。只能由受保护身体骨驱动双方同一XY平移相机，方向与正交倍率固定；按双方全部帧的变形包围盒共同拟合并保留源工程。渲染器逐帧比较相机矩阵和倍率SHA，而非只比较初始相机。此步骤只改变派生比较工程的相机动作，不改变物理或源动作。对comparison重新保存审核及渲染合成。

单向角弹簧不向身体回传力，角度使用指定父关节坐标系XYZ限位；固定原骨段长度。保持原掩码，单独显式覆盖头发/身体代理接触。骨长和角限位优先；碰撞加深更新局部回退且保留残留指标。非完整PMX/Bullet求解：跨束/重复关节、原生线性六自由度、发束自碰撞/衣物/发饰未解。无衣发HTTP/UI启用入口，health.secondary_motion仍false，不宣称FBX/其他角色通过。


## 0.7.0-dev.4 实验衣物/耦合发束流程（CLI）

入口（开发副本内同角色工程，新输出目录必需）：
```powershell
.\.venv\Scripts\python.exe -B scripts/simulate_secondary_surface.py --profile rin --scene <本副本同角色.blend> --output-dir output/secondary_motion/<全新名字> --enable-surface --compare
```
实际三角面门限依次尝试耦合版、保留原头发版、整段布片回退版；每个候选均验证，最后仍失败则保留日志并拒绝成功发布。本轮Rin的三种候选实际均被拒绝；不能把实验视频当已通过的新版默认效果。

默认UI/HTTP不自动启用，不改变动捕/重定向或主动作接口。角色配置`surface_dynamics`包含真实身体材质、饰品骨标记、真实布料材料/固定引导/求解质量、碰撞间距、自碰撞、地面高度与预热；所有物理参数有限且有范围校验。源指纹/绑定不符、新目录不存在、显式启用为硬前置条件。

阶段：capture_secondary_surface_inputs -> solve_secondary_cloth_native -> solve_secondary_hair_coupled -> assemble_secondary_surface_cache -> audit_secondary_surface（必要时guard_secondary_cloth_components整段回退，再全样本审核）；可选frame_secondary_surface_compare -> verify_secondary_surface_pair/cache -> render_secondary_hair_compare -> compose_secondary_surface_compare。低层脚本供诊断，仍拒绝覆盖旧输出。日志在logs/，阶段失败保留产物；暂不提供重入/增量恢复。主流程无需导入motionbridge.app。

原body actions/48 protected bones/raw skin/shape keys只读。衣物为明确mask的MDD动态表面层（保护body/boots），硬件独立四元数对象+GN隐藏原渲染面，原raw mesh数据不删除。MB4_SourceVertex/Face保持实际表面审计映射。发束只新增MB4旋转目标层，保留原MBH/源Action。

surface-validation.passed只表示数据保持、实际缓存回读与硬件形状验证；hair-validation.passed只表示数值/预热/原动作保护，并非残差为零。surface-audit分整数/半帧真实proper crossing与地面深度，collision_quality_passed不因回读通过而为true。camera/video/cache/protection报告各司其职，详见VALIDATION_SECONDARY_DEV4.md。

发束胶囊自碰撞/原生18跨束范围为离散迭代，真实身体BVH只做局部三角查询；衣物为真实面原生Cloth。自由6DOF硬饰/发片mesh自碰撞/双向衣发/连续CCD未完成，不能宣称完整碰撞或跨DCC导出通过。

## 0.7.0-dev.5 缓存、导出与运行控制（CLI）

既有simulate_secondary_surface.py的scene/profile/output-dir/--enable-surface保持；新增--resume、--no-cache。只写本副本output/secondary_motion；首次须新目录，resume只接管自己的run.json/control.json，拒绝仍存活同PID创建时间的运行；变更参数生成新revision，旧产物保留。运行结果分开报告computation_completed与physics_result_published；质量拒绝仍退出3，不能当程序崩溃，也不能导出成功就升级物理。

共享缓存data/secondary-cache，manifest记录schema/stage/key、逐文件bytes/SHA与来源；校验通过才恢复。缓存key含源/依赖、模型配置、Blender版本build和相关代码；头发/衣物分支选择性失效。legacy dev4采用必须显式seed_secondary_cache.py --input-dir <dev4 case> --adopt-audited-dev4并保留来源，不修改旧产物。--no-cache只禁共享缓存，--resume仍校验复用当前已完成阶段。

secondary_run_control.py --run-dir <运行目录>读取状态，加--cancel写本run_id取消令牌，仅终止本次拥有的子进程。未完成阶段重启，旧半成品保留；无帧级续算/断电保证，复制和SHA期间非即时取消。secondary_cache_control.py [--stage <名称>] [--key <64位SHA>] [--verify]列举/校验；--restore <新output目录>须同时stage/key且不覆盖已有目录。本版不提供删除和自动容量清理。

export_secondary_motion.py --scene <本副本blend> --input-dir <对应case> --output-dir <新目录> [--formats blender,skeleton,rigid,alembic,fbx-character] [--render-check] [--resume] [--no-cache] [--take N] [--experimental]。默认blender,skeleton,rigid,alembic；fbx-character须含skeleton且在其后，逐文件FBX指纹参与依赖。--take为短测不冒充全帧；--adopt format=<本副本已原生验证目录>逐个验证源SHA/完整采样/源未改，不能把外源或短测升级。物理质量拒绝须--experimental才允许格式验证；manifest.format_validation_passed与physics_quality_approved独立，后者不会被格式流程置true。最终delivery采用独立复制和清单；只修改自有清单，不覆盖已变动的旧交付。

Blender包复制外部缓存为相对路径、打包图像，保留原Action/保护48骨/蒙皮/形态键；发现链接库明确拒绝本版的便携声明。骨架FBX不含衣物顶点/形态键，bind/rest不同要按世界姿态接回原模型；fbx-character仅同源读回验证。硬饰FBX独立物体轨迹，不是自由刚体求解；Alembic含全部可见变形且无骨骼/形态键，已含硬饰勿叠加。材质JSON只供手动重连，C4D运行时未验证。本轮不新增HTTP路由/启动服务，UI留第6步，主动作API/默认参数保持。

## 0.7.0-dev.6 工作台接口（显式可选）

原主动作HTTP路径/默认参数保持。health.features.secondary_workbench=true，secondary_default_enabled=false；原secondary_motion=false保留为不自动处理身体任务的旧开关。Rin角色列表由独立命名适配器补入，不修改原纳西妲算法；POST /api/jobs/{id}/retarget character=rin使用完整Rin原蒙皮，独立衣发随后处理，Rin不启用旧纳西妲surface_clearance。角色输出增加Rin_MotionBridge_Dance.blend。未新增任意绑定适配。

- GET /api/secondary/catalog：角色能力与源工程ID，default_enabled=false/full_collision_accepted=false。不接受客户端任意绝对路径；静态源在profiles/secondary_sources.json，已完成Rin角色任务自动发现。
- GET /api/secondary/runs 与 /runs/{id}：独立任务/真实阶段/缓存命中/已有帧进度/保护/完整碰撞/实际选择模式/布片计数/预览和允许文件。
- POST /runs：JSON {source_id,profile="rin",enabled=false,mode="hair"|"surface",preview=true}。enabled严格布尔，未开/源配置不符/已有活跃衣发任务拒绝。与主动作共用单worker队列，状态在DATA/secondary-jobs、产物在本开发根output/secondary_motion/dev6/workbench。
- POST /runs/{id}/cancel、/resume：合作取消仅拥有进程；恢复拒绝源改变/仍活跃子进程，阶段级复用，旧尝试保留。服务重启标记未完成为interrupted，不自动重复运算。
- POST /runs/{id}/refresh-preview：完成物理任务复用已校验阶段，仅更新同相机预览；指纹含前后工程和渲染器SHA，半成品/旧预览不作为当前结果，失败保留新尝试。
- POST /runs/{id}/export：JSON {formats:["blender","skeleton",...],experimental:false}。支持blender/skeleton/rigid/alembic/fbx-character（后者须skeleton）。完整collision_quality_passed不是true时，须明确experimental=true；仅选择统计门限通过不能免除此要求。导出新任务/新交付，format_validated独立于physics_quality。
- GET /runs/{id}/files/{允许文件ID}：只服务该任务输出中的允许文件，拒绝路径逃逸。GET /runs/{id}/log：只读该任务及最近原生阶段日志，不读任意路径。
- GET /runs/{id}/download：已完成交付的清单SHA/大小校验后流式ZIP；不创建第二份数GB临时ZIP，固定小队列有界缓冲，客户端关闭即停止归档。不是断点续传或自动清理。

physics_quality来自实际surface-audit.collision_quality_passed，选择候选的统计不恶化门限另以selection_gate_passed标记。完成计算或门限选中不升级完整碰撞验收；前后关键帧也不是准确率证明。Web预览为五组实际渲染图片，图片成对加载/切换，不是实时3D衣发或连续视频。当前只验收Rin和3秒新源窗口；双模型按用户选择延期。

## 0.7.0-dev.7 残余维修第一批（实验）

主动作API/识别/根轨迹保持。surface_dynamics新增contact_pin_collar=4、contact_edge_iterations=48、contact_max_strain=0.05、attachment_bind_scale=0.2、prewarm_p95_limit_m_s=0.02、prewarm_max_limit_m_s=0.1、maximum_extra_edge_strain=0.25、shape_projection_iterations=128、shape_projection_passes=4；均有限且有边界。绑定距离按模型包围盒对角线计算，非任意模型保证。

cloth-validation.prewarm_convergence保存末尾约1秒每帧P95的最大值及所有顶点最大速度；surface-validation.garment_deformation_gate保存全部实际衣料/柔性滚边边长相对输入的额外误差（整数帧）。管线选择须真实相交不恶化+预热通过+形变通过；缺测也拒绝。完整collision_quality仍独立，不能由选择门限升级。contact_priority、triangle_attachment_binding、garment_ground_constraints、rigid_continuous_floor_envelope为诊断字段。

POST /api/secondary/runs/{id}/refresh-preview严格只调用渲染器，不依赖物理stage缓存复用；operation=preview_only持久化，失败/取消恢复继续只渲染，成功恢复原终态和物理质量。物理完成后预览失败的恢复也遵循同样规则。不改已有result/源/曲线。预览完整校验工程SHA、渲染器SHA和完成标记。

render_secondary_workbench_preview.py新增--all-frames用于连续诊断，默认五关键帧保持；preview.json增加fps。网页仍非实时3D衣发。报告VALIDATION_SECONDARY_REPAIR.md，限制和失败候选明确保留。

## 2026-11-07 饰品耦合诊断与止损 · 0.7.0-dev.8

run_pipeline acceptance_policy=before_strict_single_candidate_v1。成功判据为attachment_repair_gate与原有simulation_quality_gate合取。before为本次capture输入，不接受旧失败候选作替代基线。

预热通过之前不运行hair/bundle；失败时latest.quality包含selected=null、last_experimental_candidate=null、stopped_stage="cloth"、stop_reason、automatic_retry=false、reference="captured_input_before"。命令行仍退出3，computation_completed=false，physics_result_published=false。

预热通过后最多一个bundle-coupled；失败不再自动走bundle-preserved/cloth-guarded/bundle-guarded。历史缓存键保留以读取旧数据，但不再作为自动重试路线。

工作台对应状态quality_rejected、result=null、protection_passed=null、selection_gate_passed=false、physics_quality=rejected；error显示拒绝原因。无结果时预览/导出返回409。原hair模式与有结果任务的严格只渲染预览保持。HTTP路由无新增，既有status含义不升级。

## 0.7.0-dev.9 / 表面分类审核契约（2026-11-09）

profile 新增可选 surface_segmentation={schema_version:1,required_for_surface_dynamics:boolean,review:object|null}。Rin 启用且 review=null。未声明的新旧 profile 保留旧行为，不可据此声称任意角色已分类。
compile_profile 的 plan.surface_segmentation 包含 inventory、review_gate 及 required_for_surface_dynamics。inventory 用原始源面/顶点索引，不拆原网格；同材质共享边分片，坐标小数6位的重复只是候选。geometry_sha256 覆盖位置、面、材质、蒙皮；model_sha256 绑定整个资产。prepare 的 summary 追加 surface_segmentation_review。
review 格式：schema_version=1、model_sha256、geometry_sha256、parts、overlaps。parts 每项为 id/category/face_ids/evidence；inner_cloth/outer_cloth 还需非负整数 layer。类别含 protected_surface、inner_cloth、outer_cloth、hair、rigid_ornament、flexible_trim、display_overlay。全部源面必须唯一覆盖，保护表面不得改成动力学；刚性部件不得隐式吞并保护连接边界。overlaps 每项为 group_id/kind/evidence；kind 为 display_duplicate/separate_layers/authored_overlap，所有重合组须明确记录。审核通过仅表示结构和记录完备，不证明分类人工判断正确、物理面映射/附着已实现或碰撞通过。
required_for_surface_dynamics=true 时，run_pipeline 在任何 RunSession/Blender 仿真前编译检查；capture CLI 在打开/保存场景前检查。旧缓存及低层研究 CLI 不视为已转换成新分类求解器。分类结果尚未驱动物理，后续步骤必须实现映射与联合求解，不能仅填 review 就声称完成维修。
GET /api/secondary/catalog 的 profile 新增 surface_reason、surface_segmentation（审核摘要/null）。未通过时 ready 和 hair 可保持 true，surface=false；POST /api/secondary/runs 的 surface 请求返回409及具体理由、不排队。原 hair 异步成功码仍202。界面显示 surface_reason；没有新分类编辑界面。
本次保护与验证边界见 SEGMENTATION_REPAIR_PROGRESS_20261109.md；424 项测试，动态碰撞未重新验证。

## 0.7.0-dev.10 / 碰撞附着声明（2026-11-10）

可选 profile.secondary_contact_rules 必须恰含 schema_version=1、policy="model_contacts_v1"、bindings=null|object；未知字段/策略拒绝。Rin启用声明但bindings仍null。compile_profile输出plan.secondary_contact_rules，包含definition、definition_sha256、rules_defined、asset_bindings、blockers、solver_implemented=false、simulation_ready=false；prepare summary同步摘要。
policy_definition定义8类36组对称关系。身体/地面不参与位移修正，dynamic类别必须再次受attachment DOF限制；硬饰不允许独立顶点修正。显示副本仅在原生映射证实后走主面。seam只可局部声明，所有pair_rules.raw_audit_excluded=false；原始before验收不变。
bindings固定字段：schema_version/model_sha256/geometry_sha256/review_sha256/attachments/aliases/seams。attachments每个动态部件恰一项，part_id/mode/anchors/limits/evidence，骨座另有bone_id，表面模式另有target_part。mode=source_root|bone_socket|surface_rigid|surface_seam；limits包含translation_m/rotation_rad，固定源根/骨座须0，其他非负有限且角度<=pi。非表面anchor只有vertex_id；表面anchor另有target_face_id/barycentric/rest_offset_m，源偏移一致误差<=1e-6米。刚体源/目标至少3个不共线点，seam至少2点，source_root至少1点。部件/面/点必须属于对应源资产；源骨座仅可指向protected_bone_ids。共享保护顶点不能通过衣料附着移动。
aliases逐项group_id/display_face/canonical_face/evidence；必须来自review中display_duplicate关系、匹配蒙皮，唯一真实主面并覆盖全部显示副本。原生几何/形态键/动画等价尚未自动核实，结构合法只返回structurally_valid=true、asset_mapping_verified=false、requires_native_geometry_and_motion_proof=true。
seams逐项child_part/target_part/face_pairs/max_depth_m/max_slip_m/evidence，必须依附已有表面连接、处于显式源锚点和目标面的邻域；深度/滑移配置上限0–0.002米，仅限待审核配置，不改变运行默认或质量门限。不得整材质忽略或重复同一面例外。
require_review分类通过后仍调用require_contact_policy；声明存在就拒绝当前旧求解器。future solver不是可以编辑simulation_ready标志打开的功能。run_pipeline/catalog同时识别声明，即使未声明surface_segmentation也会编译并阻止执行。HTTP协议保持：当前surface不可用/启动409，hair可用时异步202；没有新增前端或分类编辑器。
stage capture代码指纹包含新模块；policy/profile内容变化会改变对应计划/缓存依赖。467测试通过；绑定实例化、连续CCD、链式铰链/自由刚体、新联合求解及动态回归尚未实现。

## 0.7.0-dev.11 / 附着驱动核心（2026-11-10）

新增secondary_attachment_driver.AttachmentSystem(plan,review,bindings)。构造时重新检查分类/绑定结构、几何指纹及硬件共享顶点；源数据复制并保持。begin_frame(source_positions,vertex_inverse_mass,rigid_inverse_mass,rigid_inverse_inertia,bone_rest,bone_current)创建独立帧状态。所有质量/惯量需显式提供，不对固定骨座施力。
source_root把指定源根逆质量置0；body/boots同样保护。bone_socket使用current@inverse(rest)相对骨座变换，固定模型缩放相消；相对变换每个奇异尺度误差<=1e-6后只去数值正交误差，负行列式/真正缩放拒绝，骨矩阵不修改。surface_rigid使用明确子参考点建立源姿态，硬件只存rotation/center；surface_seam只连接明确顶点。原始偏移转换到源三角局部框架，当前框架变化参与目标侧导数。
project_attachments提供有限预算的局部质量加权双向位置约束投影，包含刚体逆惯量和旋转偏移的数值Jacobian；每步限制以当前源帧姿态为基准。无自由度/限位冲突/迭代不足报告attachment_converged=false；坏输入/退化几何/固定点被预测器改变则异常，不回退猜测。局部转角数值步长最多0.25rad并同倍率作用两侧；不是摩擦/外力积分或完整接触求解。
返回attachment_converged、initial_residual_m、maximum_residual_m、iterations_used、source_vertices_unchanged，以及始终false的collision_checked/physics_accepted。source_vertices_unchanged指固定源点及输入隔离，不代表所有二次运动顶点都不动。positions()输出副本，不能直接当作已经通过碰撞的可发布表面。
compile_contact_policy增加attachment_driver状态，core_implemented=true但binding_structure_checked与实际配置有关，native_model_binding_accepted/production_ready均不自动升为true；Rin当前bindings=null。require_contact_policy继续禁止旧solver。capture/bundle代码指纹覆盖新模块。旧assemble_secondary_surface_cache.py无新计算路径，reject_legacy_assembly在读取缓存/创建输出前拒绝；历史结果读取导出仍可用。
原始before联合验收、保护、预热/形变门限全部保持。495测试及原生工程夹具隔离验证见ATTACHMENT_DRIVER_PROGRESS.md；真正Rin绑定/联合碰撞/动态回归未验收。

## 0.7.0-dev.12 / 离散联合接触核心（2026-11-10）

新增secondary_joint_contacts.JointContactSystem(plan,review,bindings,rules,floor_z=0,floor_gap=0,max_feature_tests=20000)。内部构造第三步AttachmentSystem，复核原分类/绑定与几何指纹；当前明确拒绝aliases、seams或display_overlay，不忽略未实现映射。rules每项恰含part_a/part_b/side_a/side_b/clearance_m，两个side为±1整数，分别表示A相对B有向法线/B相对A有向法线的所在侧；clearance_m为显式正有限值。不同物理部件对必须恰好完整覆盖，不支持同一部件自碰撞；跨接触部件共享顶点拒绝，要求原生接缝适配。面片对展开前检查15×pair_count的枚举预算，不静默采样。侧向必须由适配器明确提供，当前不是任意闭合/折叠表面的接触生成器。

begin_frame沿用第三步独立源帧/逆质量/逆惯量/源骨矩阵接口；solve(frame,iterations=96,tolerance_m=1e-5)预算1..256。顺序为附着一次、柔性边长、重新生成接触与地面，最后对实际候选重新审核全部约束。数值Jacobian覆盖自由顶点及刚体平移/旋转，整体反作用；硬件不逐点推开。真实三角点/面与包含端点的有限边接触、精确最近距离及非共面交叉审核；源输入复制，固定点检查，刚体限位保持。不能证明前后帧连续无穿透，无弯曲/摩擦/速度积分或大型BVH性能承诺。

返回discrete_constraints_converged/candidate_committed/rolled_back/iterations_used/initial_prediction_audit/candidate_audit/reason及固定false的physics_accepted/continuous_collision_checked/self_collision_checked/original_before_quality_gate_checked。initial_prediction_audit是预测输入，不是用户验收的原始before；不得拿它替代原before质量门槛。candidate_committed只表示保留内存帧状态，不生成可发布文件。预算不足/冲突将x和整块rigid状态恢复到调用前；内部异常也恢复后抛出；输入本身无效直接拒绝。audit的残差为米、交叉为显式面片对计数，只在声明范围有效。

compile_contact_policy新增joint_solver状态；discrete_core_implemented=true但production_ready/native_model_binding_accepted/continuous_collision_implemented/self_collision_implemented=false；整体solver_implemented=false、simulation_ready=false仍表示完整生产求解未验收。blockers用production_joint_solver_and_native_mapping_not_accepted替代旧的joint_contact_solver_not_implemented。require_contact_policy和旧装配拒绝保持，Rin review/bindings仍null；HTTP状态和界面未改。capture/bundle代码指纹覆盖新模块。534测试和原生隔离证明详见JOINT_CONTACT_PROGRESS.md。

## 0.7.0-dev.13 / 固定姿态预热核心（2026-11-10）

新增secondary_prewarm.WarmupSettings、diagnose_initial_state(system,frame,tolerance_m)、run_prewarm(system,frame,settings,acceleration,initial_vertex_velocity,initial_rigid_velocity,initial_rigid_angular_velocity,cancelled)。只接受第四步同一AttachmentSystem的帧；固定源姿态、源参考和质量/惯量，不读取或改写主动作文件。

WarmupSettings默认dt=1/60、max_steps=300、ramp_steps=30、window_steps=30、minimum_full_force_steps=30、damping_per_second=2、projection_iterations=96、projection_tolerance_m=1e-5、max_initial_shift_m=0.05、force_balance_limit_m_s2=0.1。dt范围[1/1000,1/30]、max_steps<=600、投影1..256，尾段至少0.5秒、总预算足够完成渐入及完整载荷；非有限值和非法预算拒绝。原速度门限固定P95<=0.02/max<=0.1m/s，不提供放宽入口。新设置为受控核心参数，未写入Rin旧默认。

diagnose_initial_state返回输入审核和明确无自由度的接触/附着冲突，all_conflicts_classified=false。source_state_audit只是传入的未预热帧，可能含预测修正；不是用户验收原始before。先诊断，再尝试一次联合投影，报告initial_shift_max_m，超初始修正预算拒绝。无自动pin release、扩大限位、质量修改、最近邻回退或参数重试。

重力对自由顶点和刚体平移为加速度，速度先作指数阻尼再加重力，旋转在允许惯量自由度内积分；未含外力矩/陀螺/摩擦。预测后联合投影并按实际位移/相对旋转重建速度，初始静态投影不伪装成仿真帧且不清空显式输入速度。最大单步角增量0.25rad，超过拒绝而不截断。完整载荷窗口测量所有允许移动顶点，刚体速度同时包含角速度叉半径与位移，固定身体不稀释P95。

速度通过后，以独立临时状态执行全力、零阻尼、零初速度的一步静态受力响应，记录maximum_unbalanced_acceleration_m_s2；检查后精确恢复候选。它不是速度历史替代品，也不代表CCD。失败即停止整次预热，不延长预算重试。取消callback只应返回布尔值；输入数组均复制。

返回core_prewarm_passed/candidate_committed/rolled_back/reason、initial_diagnosis/initial_projection/initial_shift_max_m、steps_completed/full_force_steps/history、velocity_convergence、prewarm_convergence、force_balance_probe和measured_moving_vertices。成功另返final_vertex_velocity/final_rigid_velocity/final_rigid_angular_velocity。velocity_convergence仅保留原始速度检查，未测量为null；prewarm_convergence始终为字典，只有完整预热与受力检查成功才passed=true。任何失败会把passed设false并保留可用速度数字和reason；旧prewarm_stop_gate可直接安全拒绝，不会对null调用get或误放行受力失败。初始、任一步投影失败/预算耗尽/受力不平衡/取消回滚整次位置与刚体状态；内部异常也恢复后抛出。

physics_accepted/original_before_quality_gate_checked/continuous_collision_checked/self_collision_checked始终false。core_prewarm_passed不是工作台可导出权限。compile_contact_policy新增prewarm状态(core_implemented=true，production_ready/native_model_validated=false)，capture/bundle代码指纹包含新模块；主生产门控、HTTP和前端保持，Rin仍review=null/bindings=null。旧before联合验收、保护、预热/形变门限未改。最终574项测试及原生验证见PREWARM_PROGRESS.md。

## 0.7.0-dev.14 / 回归证据复核（2026-11-12）

新增secondary_regression.regression_readiness(plan)、review_measurements(meta,audit,cloth,surface)、review_archived_case(root,case,pair)与markdown_report(report)。是只读证据核对，不是仿真执行器。regression_readiness始终反映当前新原生适配缺失，不因输入simulation_ready等声明而启用；返回native_execution_allowed=false、blockers、segmentation、joint_solver、prewarm、sequence、next_action及三项未开始状态。

review_measurements核对连续完整整数/半帧网格（2*N-1，frame_stride=1）、6通道逐帧非负整数计数及总数/有交叉样本数/最大值、地面深度/超1mm样本一致性、输入/cloth/surface帧数FPS与源身份、48骨摘要及动作/蒙皮/81形态键/身体保护记录。Rin范围固定48/81，不宣称任意模型通用保护配置。骨矩阵摘要来自既有报告，不代替重新测量。检查失败返回evidence_valid=false及具体原因；原始attachment_repair_gate、0.02/0.1m/s预热阈值、至少0.5秒窗口和0.25/5e-6形变上限保持，拒绝仅信passed标记或放宽声明。

输出evidence_valid/numeric_regression_passed、reasons、channels（before_pairs/after_pairs/delta_pairs/percent_change/worsened_samples/largest_increases）；零基线的百分比为null。采样位置clip_seconds相对meta.frame_start，不等于原长视频时间；合计改善不隐藏局部恶化，但不把逐帧零恶化偷换为原验收标准。publication_accepted/new_simulation/visual_verified始终false。数字通过也不能发布。

review_archived_case仅接受本开发工程output/secondary_motion下捕获范围内的报告，读取input、cloth-validation、surface-audit与surface-validation；保存各文件SHA、检查源文件当前指纹。baseline_reused_from须在同一捕获范围且原before/保护/源指纹一致。不会重新读取候选Blend几何，因此是历史报告核对，不是候选几何再验收。读写错误直接失败，不伪造通过。

GET /api/secondary/catalog模型项新增regression_preflight（有分类/接触声明且成功编译时为摘要，其余null）。沿用同一次compile_profile，不另跑物理；原hair/surface及409/202行为保持，无新HTTP写入口、无前端变更。

scripts/review_secondary_regression.py --output <新validation子目录> 固定检查当前Rin和现有xu-jiaqi90/momo-full的result-repair2-stable；输出review.json/REPORT.md，禁止覆盖已存在目录或写到validation外。命令不调用Blender/子进程/识别/仿真；当前报告成功生成也返回3，表明新版原生回归尚未放行。现阶段没有可被编辑标志解锁的仿真命令。后续必须先实现真实模型适配再扩展执行流程。597测试及真实存量核对见REGRESSION_READINESS.md。

## 0.7.0-dev.15 / 真实模型源面接入（2026-11-12）
surface_segmentation 新增可选 review_file={path,sha256}，与非空内嵌 review 互斥；compile_profile 在副本上解析项目内文件，校验字节SHA，调用原 review_gate，不修改调用方配置。编译结果包含解析后的 review。source_surface_map={path,sha256} 引用模型自有映射，包含 schema_version=1/model_sha256/geometry_sha256/review_sha256、aliases、parts。
aliases 每行 group_id/display_face/canonical_face/vertex_pairs，三个对应点须是源三角顶点双射；只接受已审核 display_duplicate 且同蒙皮组，不同蒙皮禁止合并。部件 parts 每行 part_id/source_bone_ids/native_rigid_ids/native_joint_ids/attachment_reference，必须覆盖全部审核部件且对应实际源蒙皮和作者关节；attachment_reference 固定 immutable_source_skin_and_authored_graph。该引用表不是 secondary_contact_rules.bindings，也不能启用求解。
SourceSurfaceMap 校验真实几何摘要及完整归属，生成 physical_face_ids/physical_triangles/canonical_vertex/fixed_vertices；check_positions 在每次使用时拒绝分离超过2e-6m或非有限源点；expand_delta(source_positions,canonical_delta) 返回独立数组，输入必须为源顶点数×3、有限且非主点和受保护主点增量为0。源网格/UV/蒙皮/形态键/原始审计表面不删改。summary 明确 contact_solver_connected=false、dynamic_attachment_constraints_complete=false、simulation_ready=false。
BlenderSourceSurfaceAdapter(mesh,rig,plan,review,manifest) 检查实际网格拓扑/材质面编号、骨架静态位置、蒙皮、所有形态键的显示等价；当前只接受启用且指向声明骨架的ARMATURE修改器，拒绝其他启用的源网格修改器和活动刚体世界。capture() 获取当前姿态的 source_positions/physical_face_ids/physical_triangles/source_bone_world/display_alias_error_m，每次核对源点/面编号和别名，物理三角朝向来自Blender实际求值结果。不会修改源场景或运行动力学。
inspect_secondary_source_blender.py --profile <项目内配置> --scene <项目内原场景> --output <validation中新或空目录> [--render]：只读源模型验证，输出adapter-validation.json及可选分类图；不是仿真或导出入口。源增量1mm检查只是映射测试，不是施力。
regression_preflight 新增 source_surface_map 状态；存在源几何接入时将缺项细化为 native_surface_joint_and_contact_adapter_pending，native_execution_allowed 仍为false。工作台实际catalog确认 Rin ready/hair=true、分类accepted=true、surface=false。未修改前端/启用入口，未放宽任何质量门限。

## dev.16 / 连接候选接口（2026-11-14）
profile.source_connections={path,sha256}引用schema_version=1/model_sha256/review_sha256/rigid_joints/stitches。compile_connections(plan,review,manifest,config)验证真实单骨硬饰、原生运动学父端和无弹簧关节，输出source_root_vertex_ids、cloth_root_vertex_count（保守候选数量，并非已确认衣根）、rigid_joints、stitch_vertex_pairs等；complete/simulation_ready=false。动态父级禁止隐式固定。
project_source_stitches(source,predicted,inverse_mass,connections,mapping,iterations=32)复制输入，先恢复固定源根，再进行逆质量加权双向同点缝合，预算1至256；固定冲突/最终误差>2e-6m拒绝，失败不写回输入。成功通过SourceSurfaceMap回传主/副面增量，保留身体/靴。
Blender专用project_authored_joint(predicted,parent_delta,parameters)使用原生YXZ坐标系和Euler限位投影独立刚体变换，校验刚性矩阵，不改源骨。此模块未接整模接触/时间积分，不能用投影通过替代动力学验收。当前真实候选因根策略过宽失败，不能启用生产。

## dev.17 / 显式固定段（2026-11-14）
source_connections文件schema_version升为2，新增roots:[{part_id,vertex_ids,evidence}]；旧自动蒙皮固定模式不再接受。每段至少2个不重复源索引，合法柔性类别、部件唯一、点在本部件以1e-6m坐标键统计的单次边界边上；不做源网格焊接。缺段/无依据/越界/重复拒绝。显式固定段与身体/靴保护合并，再闭合到既有显示别名。
编译输出schema_version=2、explicit_root_parts、cloth_root_vertex_count及source_root_vertex_ids。Rin配置当前7段109点；schema/点数不是整模物理接受标记，complete/simulation_ready仍false。缝合投影和原2e-6m门限未改变。roots按资产配置，不依赖视频帧；未声明的其他部件连接继续未完成。

## dev.18 / 原生动态关节集合（2026-11-14）
profile.articulated_connections={path,sha256}引用schema_version=1/model_sha256/dynamic_rigid_ids/joint_ids。compile_articulated验证全部触及动态集合的原生关节都列出，外部端只能是受保护骨的原生运动学刚体；未支持弹簧和无锚点集合拒绝。输出bodies/joints/dynamic_rigid_ids/fixed_rigid_ids，individual_render_links_bound/complete_physics/simulation_ready=false。
Blender专用project_island(compiled,predicted,inverse_metrics,max_iterations=96,tolerance=2e-6)：所有状态为将静态世界点映射至当前世界的4x4刚性变换，inverse_metrics逐动态刚体提供6个非负有限平移/旋转逆度量。保留原生关节帧及YXZ限位，对全关节残差求数值Jacobian并按双方逆度量分配修正，旋转绕各源刚体中心。成功返回独立状态及线性/角度残差、迭代次数；不收敛抛错，输入不改。最大256迭代，不允许放松2e-6门限；未接弹簧、碰撞、速度积分、渲染网格或生产开启逻辑。单位逆度量仅为本轮连接验证设定，不能当物理惯量标定。

## dev.19 / 共点衣面硬饰接口（2026-11-14）
profile.surface_mounts={path,sha256}引用schema_version=1/model_sha256/review_sha256/mounts。每件声明part_id/evidence/anchors，点为rigid_vertex_id/cloth_part/cloth_vertex_id。compile_mounts验证完整源共点关系（1e-6m键、实际距离≤2e-6m）、合法内外衣归属、非共线朝向、无保护表面及刚体间重复主点；禁止只保留一层。输出按主点去重的连接，simulation_ready/full_physics_accepted/collision_checked=false。
project_mounts(source,mapping,compiled,fixed_vertices,cloth_inverse_mass,rigid_inverse_metrics,predicted_offsets=None,iterations=128,tolerance=2e-6)对独立数组求解硬饰整体DOF与衣点，刚性参考由原模型静态形状向当前源输入做拟合（报告拟合误差）。逆度量各硬饰为平移/旋转两个非负值，仅诊断度量并非真实惯量。保护根不动，最多256迭代、不得放宽2e-6门限。成功回传原拓扑数组并检查硬饰/衣点连接；失败抛错且输入不变。
返回连接残差、衣料响应、刚体响应及源刚性拟合误差；passed只代表连接收敛，full_physics_accepted始终false。尚未接布料边长/弯曲或接触，真实候选首帧已发现形变超限，不得用passed启用生产或绕过原验收。

## dev.20 / 源索引联合衣饰约束（2026-11-14）
compile_profile在存在surface_mounts时生成coupled_surface摘要，记录真实边/弯曲距离/衣点/非流形边数量。simulation_ready与full_physics_accepted始终false，默认工作台不启用。
MountedClothFrame(source,mapping,compiled,connections,inverse_mass,rigid_metrics,predicted_offsets=None)：复制不可变源；仅已声明衣料节点和五件配置硬饰有自由度，源根/保护面固定。cloth_structure依据已核对physical_face_ids和canonical_vertex构建部件内边与共享边两侧对顶点距离。非流形拒绝；不猜测焊接UV断边。目标长度为当前源姿态，不是静态T姿态。弯曲距离不能判别折叠翻面，不代表有符号角度弯曲或材料参数。
SourceTriangleContacts(mapping,mounts,connections,gap=...,floor_z=...,body_gap=None,pair_provider=...,pair_budget=100000)：所有物理主面参与三角接触检查，body_gap未传时等于gap；真实资产调用分别读取原surface_settings的body_gap_m/self_gap_m/floor_z_m。small-fixture aabb_pairs仅供小规模，sweep_aabb_pairs为完整三轴膨胀包围盒扫描；超过预算抛错且不抽样。原始交叉、无朝向零距离接触拒绝，返回分类计数及至多24条定位例子；例子数量不代表计数采样。raw_render_faces_audited=false，不能替代原始所有显示面审计。
solve_coupled_surface(frame,contacts,iterations=128,tolerance=2e-6)：最多256轮，门限不得放宽；同一状态修正连接/结构/接触/地面，每条接触在最新状态重算残差，末轮后重新扫描与复查。有限预算失败返回(source.copy(), report)，恢复frame进入调用时的预测状态；异常也恢复状态。candidate_committed仅代表局部离散约束收敛，full_physics_accepted/continuous_collision_checked/integration_checked始终false。源扫描失败或预算不足不能当无碰撞。工作台、预热、原before三通道门槛不绕过。
真实验证限制：早期BVH候选第1帧迭代0次被源交叉拒绝；最终AABB版本仅合成及只读首帧检查，真实检查超候选预算。没有最终整模仿真、保存回读或导出通过证据。

## dev.21 / 流式接触与初始退出（2026-11-14）
iter_aabb_blocks(points,triangles,gap,block_size=4096)显式float64、稳定轴排序，唯一归属后入扫描的面；块间保持活动集，单次最多4096对，活跃集亦分块匹配，不积累整帧集合。保留旧有限全列表函数作为测试/旧接口，不提高其预算。
StreamingSourceContacts继承旧映射/同点规则，绑定一个MountedClothFrame。scan只读完整生成和结算所有块（即使已发现阻断也完整统计）；返回audit.scan_complete、candidate_pairs、pair_blocks、maximum_block_pairs/constraints、交叉分类/退出决策和blocking_reasons。constraints总返回空列表，不能把它当没有接触；消费者应调用project_all逐块处理，旧solve_coupled_surface已显式调用该方法并最终再scan。后块阻断/异常撤回此前块全部修正，不能发布块级结果。
exit_policy为schema_version=1/geometry_sha256/rules；精确physical源面moving_face/target_face、side±1、max_depth_m正且≤.002、evidence非空。geometry_key覆盖静态源点和三角索引。只支持固定目标、另一侧有实际可动自由度；缺方向、双动态、目标退化、固定点改变/阻碍或深度超限拒绝。固定/固定且坐标与不可变源完全相同只记录原有基线；所有交叉仍在原计数中。明确有符号规则在脱离近距候选后仍复查，禁止退出后翻回错误侧。未知零距离接触不猜法线。
全部源面仍需旧raw审计/原before三通道门槛；分块通过非物理通过。当前真实快照607610对全量证明通过，未执行新真实施力、CCD、预热或导出。native-source-scan.json保留首次独立参考精度失败，coverage-proof.json为同一数据库经一致float64独立核对的结论，不得混淆为第二次扫描。

## dev.22 / max_depth_m语义修正（2026-11-16）
StreamingSourceContacts._exit使用规则侧的有符号距离s，penetration=max(0,-min(s))与max_depth_m比较，不含gap；判定后values=s-gap仍为真实约束残差。所有范围、源保护、方向审核、固定目标要求不变。gap改变不能改变相同几何的深度准入。此修复不表示共享刚体端点的衣料结构投影已支持；真实局部候选在该保护处拒绝，未提交结果。

## dev.23 / 共享结构端点（2026-11-16）
MountedClothFrame.project_structure 对每个颜色组拆分普通衣边与刚体端点边；后者以point()求值，距离梯度[-n,n]交给project_scalar聚合同一刚体的平移/旋转梯度，逐条执行。影子x数组的刚体点不独立改变。边和弯曲距离使用同一机制，目标长度与门限保持；塌缩且目标大于2e-6m时拒绝，不猜方向。刚体内部边可能与源姿态长度不一致，当前最后审计仍拒绝；没有自动重设目标、删边或生产启用。原回退与128轮验证预算保持。

## dev.24 / 结构参考与可行性预检查（2026-11-16）
MountedClothFrame初始化新增reference/reference_binding/reference_issues；由source_pose_with_authored_rigid_attachment_reference_v1策略构建，无预测偏移和最终模拟输入。边/弯曲距离以该reference为目标。audit的edge_error_m/bend_distance_error_m为结构残差；edges_over_25_percent/max_edge_relative_change继续相对不可变source边长，新增source_edge_error_m便于区分。
solve_coupled_surface在contacts.scan之前调用check_reference，报告reference_preflight；失败reason=structural_reference_infeasible、iterations_used=0、rolled_back=true，返回source副本。预检查覆盖已知刚体内不变量及附着路径容差下的源形变必要条件，不承诺完整可行性。多刚体参考或固定点参考冲突保守拒绝；不自动改锚点、权重、预算、门限、源拓扑。未经真实整体质量验收不启用工作台。

## dev.25 / 显式附着参考模式（2026-11-16）
surface_mounts schema_version=1增加可选motion_mode；未指定为rest_rigid，另一合法值source_skinned_attachment必须伴随非空motion_mode_evidence。compile_mounts校验并传递，文件SHA继续校验。未知模式拒绝，不按材质/骨数自动推断。
mount_reference供MountedClothFrame与project_mounts共用。rest_rigid按模型rest拟合；source_skinned_attachment以不可变当前source建立局部形状，单位旋转与中心位置作为初态，再由原整体DOF受力。预测偏移不改变参考，原25%源形变保护不变。
报告mount_references逐件包含motion_mode、reference_source_fit_error_m、source_rigid_fit_error_m、constant_static_shape_across_frames、reference_from_simulated_clothing。source_rigid_fit_error_m仍记录静态模型拟合误差，即使新模式参考误差为0也不隐藏旧模型偏差。新模式不是跨帧恒形刚体；调用方只允许输入原源姿态，未提供时序积分/历史状态参考API。
本轮只在实验配置与求解入口启用新语义，完整衣物UI保持未放行，候选失败回退原source。

## dev.26 / 明确缝合的结构参考
build_reference对不含刚体的已有连接等价组，以质量加权源位置建立共同reference；权重为正逆质量的倒数。有零逆质量点时用其源参考，不相容零逆质量点报告incompatible_immovable_stitch_reference并由既有预检查拒绝。reference用于边与弯曲目标，source不变，既有源形变25%保护不变。没有新增邻近焊接、配置删点或接触排除；没有改变零距离缝合目标。

## dev.27 / 实验附着区域校验
compile_regions(config, source_ids, faces)返回本片pin数组；要求fixed/driven/free分区互斥完整且非空，有evidence，每个三角连通组件有fixed点，固定权重1、自由0、驱动在(0,1)。不自动找身体最近点，不改源模型。rin-sleeve-lab.json独立实验配置production_enabled=false，不被默认Rin配置引用；当前只用于验证脚本。

## 2026-11-19 dev.28 工作台 Blender 插件入口

新增 mode=mmd_tools，通过 /api/secondary/catalog 声明安装和回退配置。POST /api/secondary/runs 只接受已有工程 ID。每次显式启用，双进程完成作者物理烘焙与独立 MDD 回读，通过后才发布。成功返回 actual_provider=mmd_tools；回退返回 actual_provider=internal 和 fallback_reason。插件不可用时可回退到内置 hair（默认）或 surface，或配置为直接报错。导出保留原有显式确认。参数、目录结构与验收范围见 BLENDER_PLUGIN_BRIDGE.md。
