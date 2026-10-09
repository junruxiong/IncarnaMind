# Reading notes: privacy in federated learning

Working notes for chapter 2 of the thesis. Not polished — fix citations before sending to Maya.

## Ostrowski & Lam (2024), "Gradient leakage revisited"
- Reconstructs training images from shared gradients when batch size <= 8.
- Defence: add noise *and* clip; clipping alone is not enough.
- My take: their threat model assumes an honest-but-curious server. Check whether the attack still works with secure aggregation.

## Haddad et al. (2025), "Differential privacy budgets in cross-device FL"
- Epsilon of 8 cost ~3 points of accuracy on the keyboard prediction task.
- Useful table comparing per-round vs per-user accounting.
- Q: how did they pick the clipping norm? Appendix B, I think.

## Brandt (2023) survey
- Good taxonomy: inference attacks / poisoning / free-riding.
- Cite for definitions; skip the outdated benchmark section.

## Ideas / TODO
- Our hospital dataset has very unbalanced clients — none of these papers test that.
- Try a small experiment with 10 simulated clients before the group meeting.
- Ask Tom whether the 2025 workshop paper has a public code release.
