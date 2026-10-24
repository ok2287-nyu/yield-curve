# Design decisions

| Decision | Choice | Why |
|---|---|---|
| Frame rate | 5 Hz (every 5th of 25 Hz frames), dt = 0.2 s | 25 Hz is mostly redundant |
| Window | 10 warm-up frames (2 s) + 20 predicted (4 s); new window every 1 s; built at load time from tracks.csv | Median track is 15.5 s; PPUU's 20+20 would drop most tracks |
| Ego tracks | class == car, at least 25 frames (5 s) | Trucks and bicycles behave differently |
| Image | 3 x 64 x 64, 0.75 m/px (48 m square) | Small first (TA); a car is about 6 x 3 px |
| Ego placement | centre column, row 43: about 32 m ahead, 16 m behind, 24 m each side | Traffic ahead matters more than behind |
| Frame | ego-centric, translated and rotated (ego faces up) | Same as PPUU's code (`traffic_gym.py`, `_get_observation_image`) |
| Channels | red = road (aerial photo, grayscale), green = other road users, blue = ego | PPUU layout; photo first, Lanelet2 map later |
| Car drawing | filled rectangles at true size, 4x supersampled then area-averaged (soft edges); every class, incl. pedestrians, bicycles, motorcycles | Sub-pixel accuracy for 6 x 3 px cars. VRU sizes in tracksMeta are real (smallest: pedestrian 0.57 x 0.84 m), checked Oct 23 |
| Actions | (longitudinal acceleration m/s^2, curvature 1/m) | Curvature is speed-independent and integrates safely at low speed; PPUU's code uses curvature too (`map_i80.py`) |
| Curvature below 0.5 m/s | set to 0 | Undefined when stopped |
| Clipping | accel +/- 6 m/s^2, curvature +/- 0.3 1/m | Data lies inside; clipped 1 sample in rec 17 |
| Kinematics | midpoint rule (see kinematics.py) | Extraction and integration are exact inverses |
| Ego motion | kinematic update, not predicted by the world model | Ego must obey the action; the model only learns how others respond |
| State for the model | image + normalised ego speed; no world heading, no 2-D velocity | Velocity always points up in the ego frame; heading would label the roundabout arm |
| Loss | per-pixel-mean MSE on the green channel (rollout + warm-up, equal weight) + beta * KL, beta = 1e-6, KL summed over the 8x8x8 latent | PPUU's convention; z-dropout, not beta, regularises z. Revisit BCE / per-cell CE if cars fade |
| World model | spatial RSSM: per-frame encoder -> ConvGRU h (256x8x8) -> decoder; learned prior p(z\|h) and posterior q(z\|h, e_next), z = 8x8x8 per-cell Gaussian; z-dropout p = 0.5 | Memory across steps, no re-encoding of predictions, per-cell aleatoric signal (context doc section 3) |
| Action path | (acc, curv, speed) only into Pred through A; never into the prior or posterior | Matches PPUU and PlaNet; the prior reacts to the ego one step (0.2 s) later |
| Stop-gradient on prior samples | on (`--prior_sg`, default); `--no-prior_sg` for the ablation. KL balancing not used | Without it the MSE on z-dropout prior steps drives sigma_p to 0; with it the prior head learns only from the KL (adopted Oct 23) |
| Dropout | Dropout2d 0.1 in Pred (input and state) and Dec; none in the encoder or prior head | Epistemic estimate for the policy; the encoder only sees real frames; sigma_p stays aleatoric |
| Frame for prediction | ego-centric (fixed-frame prediction deferred) | Reopen if v1 loses to warp-previous or fails the geometry test |
| Prediction target | green channel only; red/blue at t+1 come from the ego pose | Model learns only how others move |
| Output | plain sigmoid, no residual on a previous frame | Latent stepping has no previous frame to add |
| Training rollout | 20 steps (4 s), autoregressive in h only; gradients through all 30 steps | Trains on its own state (drift); PPUU's forward model used 20 |
| Encoder comparison | CNN vs ViT vs ViT+MAE on single frames (3x64x64 -> 256x8x8); everything else fixed | One factor at a time (docs/experiments.md) |
| Split | Neuweiler (locationId 0) only, by whole recording: last 3 ids test, 3 before val, rest train | No leakage between splits |
| Normalisation | speed and action mean/std from train recordings only | Standard |

## Known limitations

- Road channel is the aerial photo, masked to black (0) outside the road corridor; inside it includes the levelXdata watermark in the centre island, grass texture, shadows.
- At 0.75 m/px a pedestrian is about 1 px of mass (a car ~17 px), so VRUs carry little weight in the MSE; each VRU class is inside the crop in at most ~2.5% of ego frames (Oct 23 estimate).
- Other cars come from the log and never react to the ego (log replay), as in PPUU.
- The crop sees 24 m to each side; a fast car (up to ~14 m/s) can enter the crop from outside during the 4 s horizon.
- In open-loop reconstruction, heading changes while nearly stopped (< 0.5 m/s) are lost; error stays under 1 m for 95% of 3 s windows.
