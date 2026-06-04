# Research Progress and Practical Notes on Chinese Standard Mahjong AI

## 1. SOTA Reinforcement Learning Methods for Mahjong

Top Mahjong AIs such as Microsoft’s **Suphx** improve beyond supervised learning by adding several Mahjong-specific reinforcement learning techniques.

First, Suphx uses **global reward prediction** to address Mahjong’s sparse and high-variance reward problem. Instead of relying only on the final game outcome, it trains a recurrent reward predictor to estimate the player’s eventual score from the current and previous game states. This predicted reward provides denser learning signals for policy optimization. Suphx also adds look-ahead features that encode possible winning hands and their scores, helping the RL policy reason about long-term hand conversion. [1]

Second, Suphx introduces **oracle guiding**. During early RL training, an oracle agent is allowed to observe perfect information, including opponents’ private tiles and wall tiles. This oracle is then gradually deprived of hidden information until it becomes a normal imperfect-information agent using only observable inputs. This curriculum lets the agent first learn from a much stronger perfect-information policy, then transfer that knowledge back to the legal setting. The key point is not to use hidden information at test time, but to use it as a training scaffold. [1]

Third, Suphx uses **dynamic entropy regularization**. The paper notes that Mahjong RL is highly sensitive to policy entropy: too little entropy makes self-play converge prematurely with little improvement; too much entropy makes training unstable and high-variance. Suphx therefore dynamically adjusts the entropy coefficient to keep exploration within a useful range. [1]

Fourth, Suphx proposes **parametric Monte Carlo policy adaptation**, or **pMCPA**, for run-time adaptation. Because Mahjong has an irregular game tree and standard MCTS is difficult to apply, pMCPA adapts the offline-trained policy during a specific round by simulating possible hidden states and updating the policy parameters locally. These updates are only used for the current round and are not permanently carried into future games. [1]

The practical reason RL improves over a strong supervised CNN baseline is that it is not “plain PPO self-play.” Suphx combines dense reward learning, oracle-guided curriculum, entropy control, and online adaptation. A vanilla learner against a frozen SL model often only learns to imitate or match the base; Suphx creates additional learning signals and stronger training opponents so that the RL policy has a real path beyond supervised imitation.

## 2. Botzone Chinese Standard Mahjong Bots

Public information about Botzone’s top Chinese Standard Mahjong bots is limited, but available contest pages and PKU-related materials suggest a common pattern: strong bots usually start from a **CNN or ResNet-style policy network**, train it with supervised learning on human or expert data, and then improve it using self-play reinforcement learning.

The IJCAI 2024 Botzone Mahjong AI competition page lists “响亮的名字” as rank 1 and SeaMan as rank 4; the IJCAI 2025 page lists “超强小登队 / SeaMan” as rank 1. These pages confirm the ranking lineage but do not reveal full training details. [3, 4]

The more informative public source is the PKU report, which describes a distributed RL training system using a **model pool**. The learner periodically pushes new checkpoints into the pool; multiple actor processes sample models from the pool to generate games; when the model pool exceeds its capacity, replacement strategies such as FIFO can be used. This is essentially a lightweight league/self-play population, and it avoids the weakness of always training against one frozen opponent. [2]

The same PKU source reports concrete PPO-style hyperparameters:

| Component                         | Reported value |
| --------------------------------- | -------------: |
| replay buffer size                | 50,000 samples |
| replay buffer episodes            |            400 |
| model pool size                   |             20 |
| number of actors                  |              4 |
| episodes per actor                |         10,000 |
| gamma                             |           0.99 |
| lambda / GAE                      |           0.95 |
| min samples before learner starts |            200 |
| batch size                        |           1024 |
| training epochs per batch         |              3 |
| PPO clip                          |            0.2 |
| entropy coefficient               |           0.01 |
| device example                    |            CPU |

These values are a useful starting point for your Botzone-scale system, although they should not be treated as universal optimum values. [2]

For the **8-fan minimum winning requirement** in Chinese Standard Mahjong, the most important engineering lesson is that ordinary shanten/tenpai features are not enough. A CSM bot must model **valid high-fan conversion**, not merely “can I win soon?” Useful features include current fan potential, distance to 8 fan, effective shanten under 8-fan legality, possible fan patterns, whether a low-fan hand should be abandoned, and risk-aware defense when the hand has poor conversion. Public writeups rarely disclose the exact feature set of champion bots, but the practical direction is clear: the policy must distinguish “tenpai but below 8 fan” from “legally convertible tenpai.”

## 3. Why Your PPO Self-Play Reaches Parity and How to Improve It

Your setting — PPO learner versus a frozen SL base — often converges to parity because the opponent distribution is too narrow. The learner discovers a policy that is good enough against the base, but it does not face enough strategic diversity to discover robust improvements.

A more practical self-play setup would be:

1. **Maintain a model pool**, not a single frozen base. Include the SL model, recent learner snapshots, older snapshots, and a few deliberately different policies. The PKU report uses a model pool size of 20, which is a reasonable starting point. [2]

2. **Sample opponents non-uniformly.** Do not always use the latest model. Use a mixture such as 30% SL base, 40% recent checkpoints, 20% older checkpoints, and 10% exploiters or rule-based defensive bots.

3. **Train exploiters.** Add policies whose job is to exploit your current main policy. Then train the main policy against them. This is the league-training idea used in broader imperfect-information and multi-agent RL systems.

4. **Normalize advantages and scale rewards.** Mahjong reward variance is high. Use GAE with `gamma=0.99`, `lambda=0.95`, advantage normalization per batch, reward clipping or score normalization, and a value-loss coefficient that does not dominate policy learning.

5. **Keep a KL/entropy leash to the SL policy.** Early RL should not drift too far from the SL base. A practical loss is:

```text
L = L_PPO
    + c_v * L_value
    - c_ent * H(pi)
    + beta_KL * KL(pi_RL || pi_SL)
```

Start with a relatively strong KL penalty, then decay it after the policy proves stronger in evaluation. This is especially useful when the SL policy is already competent and the environment reward is noisy.

6. **Evaluate by fixed pools, not only current self-play.** Keep a locked evaluation pool: SL base, top previous model, rule-based defensive bot, and several Botzone public bots if available. Only promote a checkpoint if it beats this pool over enough games.

For your case, the first experiment I would run is not “more PPO.” It is: keep your current PPO implementation, but replace the frozen opponent with a 20-model pool and add KL-to-SL regularization. That directly targets the parity problem.

## 4. Practical Botzone Deployment with Python 3.6 and PyTorch 1.4.0

Botzone’s old runtime creates compatibility problems when a model is trained in modern PyTorch 2.x and deployed under PyTorch 1.4.0. The safest deployment strategy is:

### Recommended method

Train however you like, but export a **minimal PyTorch 1.4-compatible inference package**.

Use:

```python
torch.save(model.state_dict(), "model_legacy.pth",
           _use_new_zipfile_serialization=False)
```

Then on the Botzone side:

```python
model = MyModel()
state = torch.load("model_legacy.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

Also make sure the model architecture code on Botzone exactly matches the training architecture.

### Avoid BatchNorm if possible

For Botzone inference, avoid BatchNorm unless you are very sure the running statistics and version behavior are compatible. Prefer:

```python
GroupNorm
LayerNorm
InstanceNorm without running-stat issues
No normalization
```

For small-batch or single-state Mahjong inference, BatchNorm is not very attractive anyway. GroupNorm or no normalization is usually cleaner.

### If you already trained with BatchNorm

Fuse BatchNorm into Conv layers before export when possible:

```text
Conv + BatchNorm -> single equivalent Conv
```

This removes BatchNorm state from inference and makes deployment more stable.

### Most reliable fallback

If PyTorch loading still crashes, reimplement only the forward pass in **NumPy**. For a small CNN/ResNet policy, this is feasible if the architecture is simple: Conv2D, ReLU, residual add, flatten, linear, softmax/action mask. It is slower but much more robust on Botzone, especially for CPU-only inference.

### My practical recommendation

For a competition bot, use this order:

1. Remove BatchNorm from the architecture.
2. Save only `state_dict`, not the full model object.
3. Save with `_use_new_zipfile_serialization=False`.
4. Load with `map_location="cpu"`.
5. Test inside a local Python 3.6 + PyTorch 1.4 Docker/conda environment before uploading to Botzone.
6. If it still crashes, export weights to `.npz` and run NumPy inference.

This is cleaner than trying to force a PyTorch 2.x object checkpoint to load directly in PyTorch 1.4.

---

# References

[1] Li, J., Koyamada, S., Ye, Q., Liu, G., Wang, C., Yang, R., Zhao, L., Qin, T., Liu, T.-Y., & Hon, H.-W. **Suphx: Mastering Mahjong with Deep Reinforcement Learning.** arXiv:2003.13590, Microsoft Research Asia, 2020.

[2] PKU AI / related Chinese Standard Mahjong reinforcement learning report. Public PDF from Peking University AI materials, including model-pool design, PPO hyperparameters, global reward prediction discussion, and distributed actor-learner implementation details.

[3] Botzone. **IJCAI 2024 Mahjong AI Competition** ranking page. Lists rank-1 team “响亮的名字” and other finalists including SeaMan.

[4] Botzone. **IJCAI 2025 Mahjong AI Competition** ranking page. Lists rank-1 team “超强小登队 / SeaMan.”

[5] Heinrich, J., Lanctot, M., & Silver, D. **Fictitious Self-Play in Extensive-Form Games.** ICML, 2015.

[6] Heinrich, J., & Silver, D. **Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.** arXiv:1603.01121, 2016. Introduces NFSP-style ideas relevant to imperfect-information self-play.

[7] OpenAI et al. **Dota 2 with Large Scale Deep Reinforcement Learning.** arXiv:1912.06680, 2019. Useful reference for league/self-play, population training, and opponent diversity.

[8] Vinyals, O. et al. **Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning.** Nature, 2019. Useful reference for league training, exploiters, and population-based multi-agent RL.
