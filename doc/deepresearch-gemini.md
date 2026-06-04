Advanced Reinforcement Learning Frameworks, Local Search Heuristics, and Legacy Platform Deployment Strategies for Competitive Chinese Standard Mahjong AIState-of-the-Art Reinforcement Learning Methodologies in Mahjong AIIn multi-agent, imperfect-information games, standard deep reinforcement learning (RL) algorithms struggle due to the vast state space, high payoff variance, and complex scoring structures. In four-player Japanese Riichi Mahjong and Chinese Standard Mahjong (CSM), the size of the information set at any decision point exceeds $10^{48}$ hidden states. Under these conditions, standard policy gradient methods often fail to converge, making specialized training frameworks necessary to surpass supervised learning (SL) baselines.Perfect-Information Phase:
┌───────────────────────────┐
│       Oracle Agent        │ ◄── Inputs: Public State (S) + Hidden Info (H)
│ (Privileged Policy Grad)  │     (Opponent Hands, Wall Tiles) 
└─────────────┬─────────────┘
              │
              │ Distillation / Gradient Transfer (KL Minimization) [1, 4]
              ▼
Imperfect-Information Phase:
┌───────────────────────────┐
│       Normal Agent        │ ◄── Inputs: Public State (S) Only 
│ (Weaned off Hidden Info)  │     (Observable Tiles, Discards, Melds)
└───────────────────────────┘
Oracle Guiding and Perfect-Information DistillationTo accelerate policy optimization under partial observability, state-of-the-art frameworks use a method known as oracle guiding or perfect-information distillation. This training pipeline splits the learning process into two distinct phases:First, an oracle agent is trained within a privileged environment. This agent has access to the full, perfect-information state of the game, including the private tiles of the other three players and the exact distribution of tiles remaining in the wall. Because the oracle does not have to reason under uncertainty, it quickly learns an optimized policy $\pi^*(a | \mathcal{S}, \mathcal{H})$ that maximizes the expected round score.Second, a normal agent $\pi_\theta(a | \mathcal{S})$ is trained using only observable public features $\mathcal{S}$. The normal agent is optimized to match the action distribution of the oracle by minimizing the Kullback-Leibler (KL) divergence between their outputs:$$\mathcal{L}_{\text{distill}}(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left \quad [4]$$During training, the privileged features $\mathcal{H}$ are gradually decayed or masked out. By slowly weaning the normal agent off this hidden information, the network parameters are forced to rely on public indicators $\mathcal{S}$ while retaining the tactical play styles discovered by the oracle.Global Reward Prediction and Dense Reward ShapingStandard Mahjong round-end scores are highly volatile, making them noisy reward signals for reinforcement learning. Because a single game consists of multiple sequential rounds, using the raw end-of-game standing as the primary reward signal fails to assign proper credit to decisions made in individual rounds. Conversely, optimizing solely for round-end scores is also sub-optimal; for example, a player with a substantial lead in a tournament might intentionally concede a small round to a low-ranking opponent to prevent a high-ranking rival from winning, securing a first-place finish overall.To resolve this credit assignment problem, SOTA systems train a global reward predictor network $\Phi(\mathcal{S}_t)$. This network takes features from the current and previous rounds to predict the agent's final game standing. This predictor provides a dense, low-variance training signal used to shape the reward at the end of each round $t$:$$\tilde{r}_t = \Phi(\mathcal{S}t) - \Phi(\mathcal{S}{t-1}) \quad $$This shaped reward reflects how much the agent's actions during the round improved its overall projected standings. Additionally, look-ahead features are engineered to calculate the mathematical feasibility of completing high-value hands, allowing the model to balance risk and point yield during play.Policy Regularization and Style Preservation ConstraintsDuring the transition from supervised learning to self-play reinforcement learning, agents often drift into overly aggressive or highly specialized policies, leaving them vulnerable to counter-exploitation. To prevent this, systems like Mixed Proximal Policy Optimization (MPPO) and Bootstrapped PPO (BPPO) implement strict divergence constraints to keep the RL policy anchored near the human supervised baseline $\pi_{\text{SL}}$.This regularization is achieved by adding a total variational divergence metric to the loss function. This metric penalizes drift in the action distribution ($D_{\text{action}}$) and shifts in the distribution of targeted winning patterns:$$D_{\text{action}}(\pi_\theta, \pi_{\text{SL}}) = \mathbb{E}_{s \sim \mathcal{D}} \left \quad [6]$$Constraining $D_{\text{action}}$ ensures the agent retains its defensive capabilities and human-like play style while refining its tactical decision-making.Run-Time Online Policy Adaptation via pMCPABecause offline policy networks are optimized against a wide distribution of generic opponents, they struggle to adapt to the highly specific hand states and opponent behaviors encountered during an actual round. To resolve this, systems like Suphx run Parametric Monte-Carlo Policy Adaptation (pMCPA) online during active match execution :Simulation: Upon receiving the initial deal of private tiles, the agent freezes its hand and randomly samples $10^5$ (100K) opponent hand configurations and wall distributions from the remaining unobserved tile pool. The offline policy is used to execute rollouts across these simulated setups.Adaptation: Gradient descent updates are performed using the basic policy gradient method over these $10^5$ simulated trajectories to fine-tune the offline policy parameters specifically to the dealt hand and seat wind.Inference: The fine-tuned policy is deployed against the actual online opponents during the active round.Adaptive StageSample VolumeMathematical OptimizationOperational ImpactOffline SL Pre-training$10^7$ state-action pairs Cross-Entropy Loss minimizationEstablishes basic defensive and offensive heuristics.Oracle Guiding (RL)Continuously generated self-playKL Divergence Minimization Accelerates policy optimization in partially observable spaces.pMCPA (Online Fine-Tuning)$10^5$ Monte Carlo rolloutsDirect Policy Gradient UpdatesCustomizes the policy to the dealt hand during active match play.Why RL Outperforms Strong Supervised BaselinesSupervised learning models excel at mimicking common human behaviors, but they are fundamentally limited by human cognitive biases and sub-optimal decision-making in highly complex scenarios. Humans often struggle to accurately calculate real-time probability distributions over hidden tiles, leading to sub-optimal risk management.Reinforcement learning overcomes these limitations by optimizing for long-term expected utility rather than imitating human actions. Through billions of self-play matches, the RL policy discovers non-linear, multi-turn trade-offs, such as intentionally delaying a win to build a higher-scoring pattern or sacrificing a hand to play defensively against an opponent who is about to win.Peking University/IJCAI Botzone Chinese Standard Mahjong Winners: Architectures and PipelinesThe PKU AILab and the IJCAI Mahjong Competitions use Chinese Standard Mahjong (CSM), also known as Mahjong Competition Rules (MCR). The tournament structure leverages duplicate matches to minimize luck and highlight strategic capability. In this duplicate format, a single match consists of 24 games where the same tile distributions are dealt across rotated seat configurations, ensuring that victory is determined by tactical skill rather than dealt hands.Input Tensor Format (34 x F Features) 
  ├── Hand Tiles (4 binary planes: representing counts 1, 2, 3, 4)
  ├── Declared Melds / Packs (Chow, Pong, Kong, Meld Kong) 
  ├── Chronological Discards of all 4 players [13]
  └── Context Metadata (Prevalent Wind, Seat Wind, Wall Count) 
Feature Representation and Deep Convolutional BackbonesTo process the game state, competitive bots on the Botzone platform use deep Residual Neural Networks (ResNets). Since there are 34 distinct tile types in Mahjong, the game state is represented as a spatial tensor of dimensions $34 \times F$, where $F$ is the number of binary feature planes.These feature planes represent the player's own hand, other players' public discards, declared melds, and context metadata. These maps are processed through residual blocks using convolutional kernels to capture spatial relationships across consecutive numbers in the three suits (Dots, Characters, Bamboos) while preserving distinct representations for Wind and Dragon tiles.Overcoming the 8-Fan Minimum (起胡) ConstraintUnder MCR rules, an agent must reach at least 8 points (fans) across 81 distinct scoring patterns to declare a winning hand (called "HU"). Declaring an invalid win carries severe point penalties on Botzone.Because standard reinforcement learning struggles to learn this hard constraint through random exploration, champion bots integrate compiled C++ helper libraries—such as PyMahjongGB or the PKU ChineseOfficialMahjongHelper—directly into their policy loop.These helper libraries analyze the agent's hand to compute the shanten (the minimum tile swaps needed to complete a winning configuration) across different hand patterns:Pythonimport PyMahjongGB

# Evaluating shanten across target patterns [14]
regular_shanten = PyMahjongGB.RegularShanten(hand=concealed_tiles)
seven_pairs = PyMahjongGB.SevenPairsShanten(hand=concealed_tiles)
thirteen_orphans = PyMahjongGB.ThirteenOrphansShanten(hand=concealed_tiles)
The bot uses these calculations to implement look-ahead action masking. For every potential action (such as declaring a Chow, Pong, or Kong), the bot simulates the resulting hand configuration. If the helper library determines that an action limits the hand's potential winning patterns to less than 8 points, that action's probability is forced to zero:$$\pi_{\text{masked}}(a | s) = \begin{cases} \pi_\theta(a | s) & \text{if } \text{MaxPotentialFan}(s') \ge 8 \\ 0 & \text{otherwise} \end{cases} \quad [17]$$This ensures the agent only transitions into states where the 8-point threshold remains reachable.Defensive Modeling and Play StylesIn duplicate formats, defense is critical to success. Under MCR scoring rules, if an agent discards a tile that allows an opponent to win, that agent suffers a direct penalty of $-(n+8)$ points, while the other two non-winning players only lose $-8$ points.To prevent this, champion bots use an explicit opponent modeling pipeline :Opponent Hand Estimation: A secondary classification network is trained to predict the tiles currently held in each opponent's hand based on their public discards and declared melds.Winning Tile Likelihood: For every potential discard in the agent's hand, the model computes the probability that each opponent needs that tile to win. If an opponent's shanten is calculated to be 1 or 0 (a ready hand), the discard safety threshold is strictly enforced. The agent will discard a totally safe tile (such as a tile discarded by that opponent earlier) even if doing so increases its own shanten, prioritizing long-term survival over aggressive play.Self-Play Frameworks for Imperfect-Information Multi-Agent EnvironmentsA common issue when training competitive game agents is the "parity trap". When an agent is trained solely against its own latest checkpoint or a frozen copy of its supervised baseline, performance gains quickly plateau.This plateau is caused by non-transitivity in multi-agent environments. In Mahjong, strategies do not follow a linear progression; instead, they exhibit cyclic dominance (similar to Rock-Paper-Scissors). If an agent trains against a static baseline, it optimizes for the specific vulnerabilities of that baseline, resulting in a narrow policy that performs poorly against other strategic styles.Population-Based Leagues and Multi-Source Opponent PoolsTo break this cycle, modern training pipelines implement a population-based league framework. Rather than training against a single opponent, the agent plays against a diverse pool of historical policies, supervised models, and specialized "exploiters".The league is structured around three primary agent archetypes :Main Agent: Optimized using Prioritized Fictitious Self-Play (PFSP) to maximize its win rate against all opponents in the league pool.Main Exploiter: Trained solely against the active Main Agent's policy. Its sole purpose is to find and exploit weaknesses in the Main Agent's strategy. Once the Main Exploiter discovers a successful counter-strategy, its checkpoint is added to the pool. This forces the Main Agent to adapt and patch that strategic vulnerability during subsequent training cycles.League Exploiter: Trained against the historical average of the entire league to identify and exploit common systematic biases across all checkpoints.To maintain strategy diversity, the opponent pool is balanced across multiple sources :$$\text{Opponent Mixture} = \{20\% \text{ Active Self-Play}, 30\% \text{ Supervised Baselines}, 50\% \text{ Historical RL Checkpoints}\} \quad [19]$$Opponents are sampled using PFSP, where the probability of selecting an opponent is proportional to its win rate against the active agent, forcing the model to focus on its most difficult matchups.Curriculum Learning and Scenario DesignIn complex multi-agent environments, curriculum learning is used to guide policy optimization. Training begins in simplified scenarios with dense, shaped rewards to teach the agent basic behaviors, such as tile connection and defensive folding. As the agent's performance meets a set threshold, the difficulty level is automatically increased, introducing more aggressive opponents and complex strategic constraints.Curriculum PhaseScenario ComplexityOpponent Pool CompositionPrimary Optimization TargetPhase 1: Heuristic Curriculum Simplified hands; dense hand-building rewardsFrozen, hand-crafted rule-based botsMaximizing tile-grouping efficiency (shanten speed).Phase 2: Challenge Curriculum Complete MCR ruleset; 8-fan constraint active Mixture of SL baselines and early RL checkpointsDeveloping baseline defensive folding patterns.Phase 3: League Self-Play Full tournament duplicates Active Main, Main Exploiter, and League Exploiters Finding approximate Nash Equilibrium strategies.Technical Deployment and Runtime Engineering on BotzoneThe Botzone platform runs a legacy environment with Python 3.6 and PyTorch 1.4.0. Model checkpoints trained with PyTorch 2.x will fail to load or execute in this environment, often crashing with exit code 120 or segmentation faults.Bypassing PyTorch 1.6+ Serialization ProtocolsBy default, PyTorch versions 1.6 and higher use a modern ZipFile-based serialization format. If a developer attempts to load a model saved in this format using PyTorch 1.4.0, the runtime will fail with a serialization error :Attempted to read a PyTorch file with version 3, but the maximum supported version for reading is 2.To bypass this issue, models trained in PyTorch 2.x must be saved using the legacy pickle serialization protocol by explicitly setting _use_new_zipfile_serialization=False :Pythonimport torch

# Saving model state dictionary using legacy serialization
torch.save(
    model.state_dict(), 
    "fused_resnet_legacy.pth", 
    _use_new_zipfile_serialization=False
)
Mathematical Folding of Batch Normalization LayersBatch Normalization layers (nn.BatchNorm2d) are a frequent source of runtime crashes on Botzone. Mismatches between old CUDNN execution libraries and modern channel tracking states in older PyTorch versions often trigger internal device errors :RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument running_mean in method wrapper__cudnn_batch_norm)This error occurs during execution even when the model has been moved to the GPU or CPU using .cuda() or .cpu().To prevent these crashes, developers fold BatchNorm layers directly into their preceding convolutional layers before serialization. This fuses the operations mathematically, eliminating the nn.BatchNorm2d layers from the model entirely while preserving identical numerical outputs.Mathematically, a 2D convolutional layer outputs:$$x_{\text{conv}} = W \cdot x + b$$The subsequent BatchNorm layer normalizes and scales this output:$$y = \gamma \left( \frac{x_{\text{conv}} - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta$$Substituting $x_{\text{conv}}$ into this equation yields:$$y = \gamma \left( \frac{W \cdot x + b - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta$$This can be rewritten as a single fused convolutional layer with modified weights $W_{\text{fused}}$ and bias $b_{\text{fused}}$ :$$W_{\text{fused}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$$$b_{\text{fused}} = (b - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$Below is a complete implementation to recursively traverse a PyTorch model and fold all sequential Convolution and BatchNorm pairs :Pythonimport torch
import torch.nn as nn
import copy

def fold_conv_bn_pair(conv, bn):
    """
    Fuses a Conv2d and BatchNorm2d layer mathematically.
    """
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    
    # Extract weights and biases
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
    
    # Extract Batch Normalization parameters
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps
    
    # Compute fused parameters 
    denom = torch.sqrt(var + eps)
    scale = gamma / denom
    
    fused_w = w * scale.view(-1, 1, 1, 1)
    fused_b = (b - mean) * scale + beta
    
    # Assign fused weights and biases
    fused_conv.weight.data.copy_(fused_w)
    fused_conv.bias.data.copy_(fused_b)
    
    return fused_conv

def fuse_bn_recursively(module):
    """
    Recursively finds and fuses BatchNorm2d layers into preceding Conv2d layers.[31]
    """
    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, nn.Conv2d) and i + 1 < len(children) and isinstance(children[i+1], nn.BatchNorm2d):
            # Fuse the sequential pair
            conv_layer = child
            bn_layer = children[i+1]
            fused = fold_conv_bn_pair(conv_layer, bn_layer)
            setattr(module, name, fused)
            # Replace the BatchNorm layer with an identity layer
            setattr(module, children[i+1], nn.Identity())
        else:
            fuse_bn_recursively(child)
    return module
Executing this recursive folding script strips the model of its BatchNorm layers, avoiding the runtime crashes associated with legacy PyTorch versions.Pure NumPy Forward Inference PipelinesOn Botzone, the safest deployment path is to avoid PyTorch inference entirely. Because Botzone executes agents using strict memory limits on CPU threads, PyTorch's large library overhead can cause execution time limits or memory allocations to exceed system caps.Since Mahjong inference is simply a forward pass through a ResNet, this process can be re-implemented in pure NumPy. Weights and biases are extracted from the PyTorch checkpoint offline and saved as a standard .npz dictionary. The bot then loads this dictionary on Botzone and executes the layers manually:Pythonimport numpy as np

class NumPyResNetBot:
    """
    An execution framework that runs ResNet forward passes in pure NumPy to bypass PyTorch legacy issues.
    """
    def __init__(self, weight_path):
        # Load the pre-extracted dictionary of weights and biases
        self.params = np.load(weight_path)
        
    def conv2d(self, x, weight, bias, padding=1, stride=1):
        """
        Executes a 2D convolution in NumPy.
        """
        batch, in_c, in_h, in_w = x.shape
        out_c, _, k_h, k_w = weight.shape
        
        # Pad the input array manually
        padded_x = np.pad(
            x, 
            ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
            mode='constant'
        )
        
        # Calculate output dimensions
        out_h = int((in_h - k_h + 2 * padding) / stride + 1)
        out_w = int((in_w - k_w + 2 * padding) / stride + 1)
        
        output = np.zeros((batch, out_c, out_h, out_w))
        
        # Perform spatial convolution
        for c_out in range(out_c):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    patch = padded_x[:, :, h_start:h_start+k_h, w_start:w_start+k_w]
                    output[:, c_out, h, w] = np.sum(patch * weight[c_out], axis=(1, 2, 3)) + bias[c_out]
                    
        return output

    def relu(self, x):
        return np.maximum(x, 0)

    def residual_block(self, x, weight1, bias1, weight2, bias2):
        """
        A residual block implementation with a skip connection.
        """
        out1 = self.relu(self.conv2d(x, weight1, bias1, padding=1))
        out2 = self.conv2d(out1, weight2, bias2, padding=1)
        return self.relu(out2 + x)
While this pure NumPy approach is slower than PyTorch's compiled backends on large models, it is highly reliable. It guarantees 100% compatibility across legacy environments and prevents the runtime crashes associated with library version conflicts.ConclusionsTo build a highly competitive AI for Chinese Standard Mahjong on Botzone, developers must integrate advanced reinforcement learning frameworks, heuristic constraints, and optimization techniques.Strategic and Tactical IntegrationA strong supervised learning baseline establishes foundational human play styles and defensive heuristics. Transitioning from this baseline to superhuman performance requires:Oracle Guiding: Training an agent with perfect-information states before distilling this policy to run on observable features, accelerating convergence under partial observability.Global Reward Prediction: Structuring rewards around long-term standing forecasts rather than raw round-end scores, stabilizing the gradient updates.Population-Based Leagues: Training the agent against a diverse, dynamic pool of historical checkpoints, exploiters, and supervised baselines, preventing policy stagnation.Action Masking Heuristics: Integrating high-performance C++ helper libraries to calculate real-time shanten, ensuring the agent only takes actions that keep the 8-point threshold reachable.Deployment and Platform OptimizationTo deploy a modern neural network in Botzone's legacy Python 3.6 and PyTorch 1.4.0 environment, developers must resolve runtime compatibility issues:Serialization Protocols: Models trained in PyTorch 2.x must be saved using legacy serialization by setting _use_new_zipfile_serialization=False to prevent loading failures.BatchNorm Folding: Fusing Batch Normalization layers directly into preceding Convolution layers eliminates unstable BatchNorm modules from the network architecture, avoiding device mismatches and memory faults.Pure NumPy Inference: For maximum reliability, developers can re-implement the model's forward pass in pure NumPy, bypassing PyTorch entirely to ensure stable execution within Botzone's strict system resource limits.