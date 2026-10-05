# Prior-art check: photometric-stereo normal supervision for Gaussian Wrapping

This report came from a web research run on 2026-09-24. It is reference material, not instructions. Some statements about the code were later checked against the repository, and the corrections come first.

## Gap check (2026-10-01)

The two gaps the report names ("Not covered" below) were checked again from a cloud session. Neither is fully closed, and this says how far each got.

- **The code has not moved.** `git ls-remote https://github.com/diego1401/gaussianwrapping HEAD` gives `11e3b6fb5ca6f0a54e2b5587a09693488f3655af`, the commit CLAUDE.md pins (2026-08-28). No new commit on the default branch in five weeks, so nothing in the code says a normal prior was added.
- **Issues, pull requests and forks: still unread.** GitHub's API for a repository the session does not own needs it attached with Dawud's GitHub credentials; that was asked for and refused by the session's permission check, and the issue pages were not fetched another way. Dawud can read them in a browser (`https://github.com/diego1401/GaussianWrapping/issues?q=`, `/pulls?q=`, `/forks`) or with `gh issue list -R diego1401/GaussianWrapping --state all`. Incidentally, web search shows a fork `PeterZs/GaussianWrapping`, an account that mirrors many research repos; it was not opened.
- **"Cited by": no citation index was reachable.** The session's egress policy blocks Google Scholar, Semantic Scholar's API, OpenAlex and arXiv (HTTP 403 on CONNECT). Web search was used instead, which is not a citation index: a citing paper that does not use these words would be missed. Searched: "Gaussian Wrapping" with 2604.07337, "From Blobs to Spokes" follow-ups, multi-view photometric stereo with Gaussian splatting and watertight meshes, near-light calibrated PS with Gaussian splatting, PS with Gaussian splatting for garments and fabric, and Gaussian Wrapping at ECCV 2026.
- **Result: still nothing supervises Gaussian Wrapping, or any watertight Gaussian mesh method, with PS normals, and nothing combines PS with Gaussian splatting for fabric.** New since the report, none of which changes the verdict:

  | Work | Date | What it is | Why it is not the combination |
  |---|---|---|---|
  | "From Dark Flash Images to Relightable 3D Scenes with Photometric Stereo Priors" (Frontiers of Computer Science 20, 2009713) | 8 Aug 2026 | NeRF / 3DGS-style reconstruction of dark scenes from a camera-mounted flashlight; a PS problem per image group gives supervision priors, and the flashlight's angular and distance attenuation, pose and intensity are modelled and optimised | The closest new item: near-light PS used as a training prior, with a light model like ours. But a moving co-located flash, relighting of scenes, no watertight mesh and no fabric. Abstract only |
  | CoMVS-GS (arXiv 2608.18413) | 2026 | Multi-view stereo depth and 3DGS supervising each other for surface meshes | MVS depth, not PS normals |
  | AnyGS2Mesh (arXiv 2609.03304) | Sep 2026 | Feed-forward meshes from 3D Gaussians | No PS |
  | Cloth-HUGS (arXiv 2604.15875) | Apr 2026 | Separate Gaussian layers for body and clothing of a performer | No PS; deformation and appearance |
  | GS-2M (arXiv 2509.22276) | 2025 | Material-aware Gaussian splatting meshes; roughness supervised from multi-view photometric variation | Photometric variation across views, not PS normals |

- **What is left before L_PS:** the issues, PRs and forks (Dawud, in a browser), a real "cited by" list (Google Scholar from an ordinary browser), and the ECCV 2026 workshop proceedings. Sources for this section: https://arxiv.org/abs/2604.07337, https://link.springer.com/article/10.1007/s11704-026-51767-9, https://pith.science/paper/2608.18413, https://arxiv.org/pdf/2609.03304, https://arxiv.org/abs/2604.15875, https://arxiv.org/pdf/2509.22276, https://github.com/PeterZs/GaussianWrapping.

## Corrections after reading the code (GaussianWrapping @ `11e3b6f`, 2026-08-28; checked 2026-09-25)

- **Which config is actually used.** `train.py` defaults to `--normal_field_config default_regular_densification`, not `default`. With that config:
  - The oriented-normal field starts at iteration **20,001**.
  - Flip-clone densification runs from **22k to 26k**, every 1k iterations, on the top **5 %** of normal errors.
  - Training is 30k iterations, and extraction hardcodes `--iteration 30000`.
  - So "add L_PS after densification ends" leaves only 4k iterations unless training is extended.
- **Where alignment happens.** `regularization/regularizer/normal_field.py::compute_normal_field_regularization` renders the per-Gaussian oriented normals in the **world** frame (`colors_precomp=gaussian_normals`). It aligns them with normals from the rendered **median depth** (`align_with_rendered_median_depths: true`, weighted by 0.6 and then by 0.05). It converts camera-frame depth normals to world with `view_to_world_transform`. A PS term would sit next to this and use the same conversion.
- **What drives densification.** Densification selects Gaussians by comparing oriented normals with median-depth normals (`densification_normal_to_use: median_depth`) and takes a fixed quantile. A PS term therefore would not change *how many* Gaussians are cloned. It would change *which* ones, favouring regions where the PS normals pull away from depth. The report's concern still holds in this form.
- **What `--depth_order` really does.** It computes DepthAnythingV2 priors at start-up from the training images; nothing is precomputed externally, though the checkpoint must sit in `submodules/Depth-Anything-V2/checkpoints/`. Its weight is 1 from iteration 3k, 0.1 from 7k and 0.01 from 15k, and it is switched off at 15k, so the config's 1e-3 and 1e-4 steps never apply. It is a useful pattern for adding a new prior (a per-camera prior list, a YAML config and a loss term in `train.py`). It is not a normal hook.
- **Post-processing is on by default.** Both end-to-end scripts post-process during extraction unless you pass `--no_postprocess`. Post-processing keeps only the largest connected component. The raw `…_searched.ply` mesh is always written first.
- **GW ignores the camera principal point.** It hardcodes Cx = (W−1)/2 and Cy = (H−1)/2. Any PS supervision must use the same centred camera model, or images cropped so the principal point is centred.
- **Texture refinement changes only colours.** It optimises vertex colours, not geometry.
- **GitHub issues and PRs are still unread.** Neither this run nor the check could read them (robots.txt and API access were blocked). Check them before building L_PS.

---

# PS-Normal Supervision for Gaussian Wrapping: Prior-Art Check, Compatibility, Design Proposal and Assessment

**Verdict: (c) nothing found that supervises Gaussian Wrapping's oriented Gaussians with photometric-stereo (PS) normals. There is (b) closely related work, but all of it is built on 3DGS or 2DGS, none of it produces watertight meshes, and none targets fabric.**

Confidence that the exact combination is still unpublished as of 24 September 2026 is moderate to high, about 80%. Two gaps limit it: the six GitHub issues and the pull requests on the GaussianWrapping repo could not be read, and only the abstract of the Ju et al. JSTSP paper was available.

## TL;DR
- **Not found anywhere checked.** The search covered the paper, project page, README, authors' pages (Gomez, Guédon, Gong), known citing works, and general PS + Gaussian Splatting searches. The nearest relatives are:
  - PS-GS (Chen et al.): 2DGS with uncalibrated PS normals on rendered normal maps.
  - PSGS (Ju et al.): 3DGS with normal-guided initialisation and cloning.
  - GS-PS (Ducastel et al.): single view.
- **It is buildable.** Gaussian Wrapping already renders an alpha-blended map of per-Gaussian *oriented* normals, N(p). A PS term can be a second cosine loss on that same N(p), after converting the PS normals from camera to world coordinates and adding confidence weights and sign handling. PS-GS's loss (Eq. 10) depends on the representation only through "a rendered normal map", so it transfers. Note that `--depth_order` is a depth prior, not a normal hook.
- **The payoff over baking is real but narrow, and mostly reasoning rather than evidence.** Training-time supervision can move surface positions and the occupancy isosurface where multi-view stereo is weak, such as low texture or repetitive patterns like fabric weave. Baking a UV normal map cannot do that. Below mesh resolution, and wherever the geometry is already right, baking is as good and far cheaper. Build it if the geometry itself matters (simulation, measurement); bake if you only need appearance.

## Key findings

### How the stated understanding of Gaussian Wrapping holds up (established, from the paper, project page and README)
- **Correct: learnable oriented normal per Gaussian.** The project page says: "We address this by endowing each Gaussian with a single learnable oriented normal n_i ∈ S²—the only additional parameter in our framework". The occupancy/vacancy field comes from Objects as Volumes, and meshes are extracted with Pivot-Based Marching Tetrahedra or Primal Adaptive Meshing (PAM).
- **Needs a precision: "normal-alignment loss against depth gradient".** Eq. 4 is L_N = Σ_p 1 − N(p)·∇D(p).
  - N(p) is the *rendered* (alpha-blended) oriented normal at pixel p, not each Gaussian's own normal.
  - D is depth rendered "as the exact 0.5-isosurface of our geometric field" via binary search.
  - So the loss works per pixel and reaches individual Gaussians only through the blending weights. It is self-supervised: there is no external signal.
- **Missing from the summary: densification is driven by the same loss.** Gaps "cause L_N to rise locally. Every K iterations we propagate per-pixel errors back to individual Gaussians via their blending weights, then clone high-error Gaussians with flipped normals". This is the main coupling a PS term would disturb.
- **Missing: it works as a plug-in regularizer.** The paper plugs it into RaDe-GS and reports "consistent improvement across all T&T scenes", calling it "an effective drop-in regularizer". The code ships two rasterizers: RaDe-GS ("Best looking meshes!") and the authors' median-depth one ("faster, better metrics").
- **Correct: `--depth_order`.** The README says it adds "depth-order regularization using a pre-trained monocular depth model. It is not used in the paper, but can yield better results."
- **Other status:**
  - Accepted at ECCV 2026.
  - Code public since 7 April 2026; README last updated 1 May 2026.
  - At the time of checking: about 231 stars, 15 forks, 6 issues (content not read).

### Related work (established unless noted)

| Work | Date | Base representation | PS setup | How normals enter the loss | Target materials | Mesh extraction | Code | Link |
|---|---|---|---|---|---|---|---|---|
| Gaussian Wrapping (Gomez, Guédon et al.) | arXiv 8 Apr 2026; ECCV 2026 | 3DGS + learnable oriented normal | None | Rendered oriented normal vs. rendered depth gradient (Eq. 4) | General scenes (DTU, T&T, Mip-NeRF 360) | Pivot MTet / PAM, watertight | Public | https://arxiv.org/abs/2604.07337 |
| PS-GS (Chen, Liang, Guo, Cheng, Zhao, Weng) | arXiv 24 Jul 2025; AAAI 2026; C&G 2026 | 2DGS (surfel normal t_u×t_v) + deferred PBR | **Uncalibrated** PS (self-calibrating network, Chen et al. 2019) | L_{n,r} = Σ(N_r − T_c2w(N_e)) on alpha-blended rendered normal map (Eq. 8, 10), in both stages | Objects (DiLiGenT-MV, PS-NeRF synthetic) | Not a focus; evaluated on normal MAE | No repo found | https://arxiv.org/abs/2507.18231 |
| PSGS, "Photometric Regularization for 3DGS in Multi-View Surface Projection" (Ju, Zhao, Xiao, Dong et al.) | IEEE JSTSP (Xplore 11192617); exact date unconfirmed, likely late 2025 | 3DGS | Not confirmed from abstract | Normal-guided initialisation, adaptive cloning where normals vary strongly, visibility-aware occlusion handling | DiLiGenT-MV | Not stated; reports PSNR (+2.18 dB), ~150× faster rendering | No repo found | https://ieeexplore.ieee.org/document/11192617/ |
| GS-PS (Ducastel, Quéau, Tschumperlé) | Jul 2025 (arXiv 2507.06684) | 2DGS-style planar Gaussians, **single view** | **Calibrated** Lambertian | Solves PS itself: photometric L1 with Lambertian shading + depth-gradient vs. splatted-normal loss (Eq. 8) | Lambertian DiLiGenT objects | None (normal maps) | Paper only | https://arxiv.org/abs/2507.06684 |
| SuperNormal (Cao & Taketomi) | CVPR 2024 | Neural SDF (hash grid) | PS normals from SDM-UniPS | Multi-view normal integration | Objects | Marching cubes | Public | https://arxiv.org/abs/2312.04803 |
| Brument et al., "Multi-view Surface Reconstruction Using Normal and Reflectance Cues" (RNb-NeuS follow-up) | IJCV 2025 | SDF | PS normals + reflectance | Joint reparametrisation of normals and reflectance | Objects | Implicit surface | See paper | https://arxiv.org/abs/2506.04115 |
| Scale-encoded multi-view normal integration (Yang et al.) | arXiv Mar 2026; CGF | Neural SDF | Normal maps (MVPS setting) | Normal integration | Objects | SDF | See paper | https://arxiv.org/abs/2603.20337 |
| Adjacent: AmbiSuR (Li et al.) | ICML 2026 | PGSR-based GS | None (monocular priors) | Photometric disambiguation | General | Mesh | Public | https://arxiv.org/abs/2605.12494 |
| Adjacent fabric: PGC; Gaussian Garments; GarmentGS; ClothingTwin | 2025 | Mesh + Gaussians / 2DGS | **No PS** | Point-cloud normals (GarmentGS) or none | Garments | Mesh | Mixed | e.g. https://arxiv.org/html/2503.20779 |

Note: two different papers are called "PS-GS". Do not conflate them.

### Answers to Part 1
1. **The exact combination:** nothing found. The search covered the paper, README, project page, Guédon's and Gong's publication lists (the newest Guédon work is Surflo, a flow-matching model unrelated to PS), and the X announcement thread.
2. **Close relatives:** only PS-GS (2DGS) supervises a Gaussian surface representation with PS normals during training. No PS-supervised SuGaR, GOF, RaDe-GS, MILo or GGGS was found.
3. **The general case:** no Gaussian-Splatting mesh method was found that uses *calibrated* or known-lighting multi-view PS normals to supervise geometry during training. Ducastel et al. is calibrated but single-view with no mesh. PS-GS is uncalibrated. The PS input in Ju et al. is unconfirmed.
4. **Fabric:** nothing combines PS normals with Gaussian Splatting for cloth. The garment Gaussian Splatting methods use multi-view RGB, point clouds or physically based rendering (PBR), never PS.
5. **After PSGS or Gaussian Wrapping:** the September 2026 paper "Integrating Multi-view Multi-light Surface Reconstruction into Cultural Heritage Workflows" (arXiv 2609.15833) wraps UniMS-PS, SDM-UniPS and LINO-UniPS as Meshroom nodes. Judging from the snippet, it is a pipeline paper, not Gaussian Splatting supervision. The full text was not read.

**Searched:** arXiv (papers and full text), the GitHub repo, DeepWiki, the project page, the authors' homepages, X, Pith, IEEE Xplore, ScienceDirect, ResearchGate, Papers with Code mirrors and paper-note sites.
**Not covered:** GitHub issues, PRs and forks; Google Scholar "cited by" lists; OpenReview; ECCV 2026 workshop proceedings.

## Details

### Part 2: Compatibility (the equations are established; the compatibility judgement is the researcher's analysis)
- **PS-GS (Eq. 8, 10):**
  - The normal map is N = Σ w_i n_i with w_i = T_i α_i / Σ T_i α_i.
  - The loss compares N_r with T_c2w(N_e), i.e. in world space.
  - The only 2DGS-specific part is where n_i comes from (Eq. 4, n = t_u × t_v). In Gaussian Wrapping, n_i is the learnable oriented normal, which the rasterizer already blends.
  - So the loss does not depend on the representation. Caveats: the norm is unspecified, and uncalibrated PS carries a sign/GBR ambiguity.
- **Ducastel et al. (Eq. 8):**
  - Its L_n compares a depth-gradient normal with the splatted normal, the same idea as Gaussian Wrapping's own L_N.
  - The PS part is a Lambertian re-rendering loss that needs calibrated lights and single-view z=0 initialisation.
  - It is compatible only as an inverse-rendering add-on.
- **PSGS (Ju et al.):**
  - Its initialisation likely rotates each Gaussian's shortest scale axis. That conflicts with Gaussian Wrapping, which keeps the normal as a separate parameter, so you would initialise n_i instead.
  - Its cloning rule would compete with Gaussian Wrapping's flip-clone densification.
  - Its equations were not verified.
- **Rasterizer:** Gaussian Wrapping's CUDA rasterizers already output N(p) and D(p). A PS loss needs no kernel changes unless you want per-Gaussian (unblended) supervision.

### Part 3: Proposed design (the researcher's own proposal)
**a. Loss term.**
- **Coordinate transform:**
  - Take the PS normal n^c in camera coordinates.
  - Convert it to the codebase's convention. COLMAP/3DGS use x right, y down, z forward. Many PS tools use an OpenGL-style frame, which needs n^c ← diag(1,−1,−1) n^c.
  - Rotate to world coordinates: n^w = R_c2w n^c.
  - Check this on one frame against Gaussian Wrapping's ∇D normals: the median cosine should be above 0.9.
- **What to supervise:** the rendered map, with L_PS = Σ_p c(p) (1 − N(p)·n^w(p)). Optionally, add a per-Gaussian term using the most-visible Gaussian per pixel for thin structures.
- **Weights:** c(p) = M(p) · clamp((n^w·v)/τ, 0, 1) · conf_PS(p) · 1[D(p) consistent], with τ ≈ cos 75°. Drop pixels with accumulated alpha below about 0.5 and pixels at depth discontinuities.
- **Unreliable PS normals:**
  - Mask shadowed and saturated pixels per light.
  - Use the PS fit residual, or the variance across light subsets, as confidence.
  - Down-weight concavities, where interreflections bias normals.
  - Use a robust penalty (Cauchy or Huber).
  - Optionally fit a per-view rotation to absorb calibration error.
- **Sign:** use the *signed* cosine, never |cos|. The sign defines inside and outside for the occupancy field. For uncalibrated PS, fix the concave/convex flip before training.

**b. Where to hook in.** `--depth_order` is a template (optional precomputed per-view prior, config file, weight), not a normal hook. Add a parallel `--ps_normals` path that loads per-view normal and confidence maps and adds L_PS next to L_N.

**c. Conflicts with existing parts.**
- Keep L_N: L = … + λ_N L_N + λ_PS L_PS. L_PS pulls N toward the PS normals, and L_N pulls D toward N. That chain is how PS moves the geometry.
- The flip-clone rule keys on high L_N. Either gate densification on L_N alone and exclude L_PS from the error map, or freeze L_PS during the densification window.
- Mask PS near thin structures, where the normals are unreliable.
- Ramp λ_PS up from 0 once densification ends.

**d. When to run it.** Run stock Gaussian Wrapping first, then a joint fine-tuning phase with L_PS on and densification off or restricted. PS-GS used a similar two-stage scheme.

### Part 4: Honest assessment
**Evidence-backed:**
- PS normal supervision beats self-supervision on normal accuracy: in PS-GS's ablation, removing the normal loss raises mean angular error from 5.57° to 7.02°.
- SDF methods report fine relief from multi-view normal integration.
- Garment work (PGC) shows fabric detail can live in appearance layers rather than geometry.

**The researcher's reasoning:**
- **What training-time supervision can do that baking cannot:** shift surface positions and the 0.5-isosurface where multi-view stereo is ambiguous; correct low-frequency shape errors, such as bulges on textureless panels, which a normal map cannot fix; and give the watertight mesh correct mid-scale relief (folds, seams).
- **What it probably cannot do:** create thin structures, fix silhouettes, change topology much, or recover relief finer than the density of Gaussians and mesh vertices.
- **Where baking is as good:** well-textured objects, render-only use, and weave-scale detail.
- **Risk:** on fabric, PS itself is weak. Sheen, fuzz, subsurface scattering and interreflections inside folds bias the normals. With training-time supervision those biased normals change the geometry; with baking they only damage a texture.

## Recommendations (from the report)
1. Run a cheap check first. Reconstruct with stock Gaussian Wrapping and measure where the mesh normals disagree with the PS normals above mesh resolution. If the disagreement is only high-frequency, bake and stop.
2. If there is low- or mid-frequency disagreement, implement L_PS on N(p): signed cosine, confidence weights, staged schedule, and densification gated on L_N only. Ablate on DiLiGenT-MV before moving to fabric.
3. Use calibrated or universal PS (SDM-UniPS, UniMS-PS) with confidence maps. Avoid uncalibrated PS unless the GBR/sign ambiguity is resolved first.
4. Before committing, check the GaussianWrapping issues, PRs and forks, the Scholar "cited by" lists and the ECCV 2026 workshops.

## Unexplored vs. solved
**Solved:**
- Supervising alpha-blended Gaussian normal maps with PS normals (PS-GS, on 2DGS).
- PS-guided initialisation and cloning for 3DGS (Ju et al.).
- Calibrated single-view PS with Gaussian Splatting (Ducastel et al.).
- Fine relief from multi-view PS normals in SDFs.
- Baking normal maps onto meshes.
- Watertight thin-structure meshes from Gaussian Splatting (Gaussian Wrapping).

**Unexplored (no evidence found):**
- PS supervision of oriented Gaussians together with the Objects-as-Volumes occupancy field.
- How it interacts with the L_N-driven flip densification.
- Calibrated multi-view PS in any Gaussian-Splatting-to-watertight-mesh pipeline.
- A controlled comparison of training-time PS supervision against baking.
- Any of this applied to fabric.

## Sources
1. https://arxiv.org/abs/2507.18231 and https://arxiv.org/pdf/2507.18231
2. https://ieeexplore.ieee.org/document/11192617/
3. https://diego1401.github.io/BlobsToSpokesWebsite/
4. https://arxiv.org/abs/2604.07337 and https://arxiv.org/html/2604.07337
5. https://github.com/diego1401/GaussianWrapping
6. https://deepwiki.com/diego1401/GaussianWrapping
7. https://anttwo.github.io/
8. https://arxiv.org/pdf/2507.06684
9. https://www.researchgate.net/publication/383985861 (Gaussian Garments)
10. https://en.papernotes.org/CVPR2025/3d_vision/pgc_physics-based_gaussian_cloth_from_a_single_pose/
11. https://arxiv.org/html/2505.02126v1
12. https://arxiv.org/pdf/2609.15833
13. https://arxiv.org/abs/2312.04803
14. https://arxiv.org/pdf/2507.23162
15. https://arxiv.org/pdf/2506.04115
