import rlcard, numpy as np
from rlcard.agents import RandomAgent
env = rlcard.make("doudizhu", config={"seed": 0})
print("num_players", env.num_players, "num_actions", env.num_actions)
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)
trajs, payoffs = env.run(is_training=True)
print("payoffs", payoffs, "traj_len_seat0", len(trajs[0]))
sa = trajs[0][0]
print("pair type", type(sa).__name__, "len", len(sa) if hasattr(sa, "__len__") else "-")
state = sa[0] if isinstance(sa, (list, tuple)) else sa
print("state keys", list(state.keys()))
print("obs shape", np.array(state["obs"]).shape, "dtype", np.array(state["obs"]).dtype)
print("n_legal", len(state["legal_actions"]), "legal_keys_sample", list(state["legal_actions"].keys())[:5])
try:
    from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
    print("RULE AGENT: available")
except Exception as e:
    print("rule agent err:", str(e)[:80])
