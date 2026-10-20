import os
import sys
import json
import time
import random
from collections import defaultdict, deque
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v79.run_v79_controlled_causal_study import (
    get_5p_fullgame, TrueNFSP_6Action_Agent, True6ActionAdapter,
    advance_chance_nodes, extract_player_spent, normalize_inf_state_v79,
)
from v81.run_v81_zero_leak_hybrid_audit import eval_7cards
from v83.run_v83_robust_range_and_cfr_audit import RobustBayesianRangeModel
from v88.run_v88_adversarial_learning_and_hero_exploitation import CallingStationBot
from v90.run_v90_adversarial_causal_audit_and_scaling import MultiwayHeroExploiter
from v1013.run_v1013_replay_and_controller_validation import (
    get_pool_standard, get_pool_aggressive, get_pool_dynamic,
    DeterministicTrueNFSP_Agent, ModelA, ModelB, ModelD_SemanticController,
    DeterministicBalancedBot, DeterministicTightPassiveBot, DeterministicUltraAggressiveBot,
    DeterministicDynamicSwitchingBot, DeterministicTightAggressiveBot,
    DeterministicBlindStealerBot, DeterministicHyperAggressiveBot,
)

# =====================================================================
# MODEL E: DEEP NEURAL REINFORCEMENT LEARNING PLAYER
# =====================================================================
class ModelE_NeuralPlayer(nn.Module):
    def __init__(self, state_dim=1417, action_dim=6, hidden_dim=256, lr=0.001):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value Network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=20000)
        self.training_steps = 0
        self.losses_policy = []
        self.losses_value = []

    def forward(self, x):
        logits = self.policy_net(x)
        value = self.value_net(x)
        return logits, value

    def act_eval(self, inf_state, legal_6, rng=None):
        norm = normalize_inf_state_v79(inf_state)
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            logits = logits.squeeze(0)
            logits[~mask] = -1e9
            probs = F.softmax(logits, dim=-1).numpy()

        if rng is not None:
            return int(rng.choice(self.action_dim, p=probs))
        return int(np.random.choice(self.action_dim, p=probs))

    def act_train(self, inf_state, legal_6, epsilon=0.10):
        norm = normalize_inf_state_v79(inf_state)
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        mask[legal_6] = True

        if random.random() < epsilon:
            return random.choice(legal_6), 0.0

        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            logits = logits.squeeze(0)
            logits[~mask] = -1e9
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            act = dist.sample()
            log_prob = dist.log_prob(act).item()

        return int(act.item()), log_prob

    def update_batch(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None, None

        batch = random.sample(self.replay_buffer, batch_size)
        s, a, r, ns, mask, done = zip(*batch)

        s_t = torch.tensor(np.array(s), dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.long)
        r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        ns_t = torch.tensor(np.array(ns), dtype=torch.float32)
        done_t = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        mask_t = torch.tensor(np.array(mask), dtype=torch.bool)

        self.train()
        logits, v_pred = self.forward(s_t)
        with torch.no_grad():
            _, next_v = self.forward(ns_t)
            target_v = r_t + (1.0 - done_t) * 0.99 * next_v

        # Value Loss
        loss_val = F.mse_loss(v_pred, target_v)

        # Policy Loss with Advantage
        adv = (target_v - v_pred).detach()
        masked_logits = logits.clone()
        masked_logits[~mask_t] = -1e9
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(a_t)
        loss_pol = -(log_probs * adv.squeeze(1)).mean()

        total_loss = loss_pol + 0.5 * loss_val

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        self.losses_policy.append(loss_pol.item())
        self.losses_value.append(loss_val.item())

        return loss_pol.item(), loss_val.item()

# =====================================================================
# SELF-PLAY LEAGUE TRAINING ORCHESTRATOR
# =====================================================================
def run_self_play_training(model_e: ModelE_NeuralPlayer, model_a: ModelA, model_b: ModelB, model_d: ModelD_SemanticController, total_hands=10000):
    print(f"\n[*] Starting Model E Self-Play League Training ({total_hands} hands)...")
    game = get_5p_fullgame()
    checkpoints = {}

    # Save E0
    torch.save(model_e.state_dict(), "v1014/models/model_e0_init.pt")
    checkpoints[0] = "v1014/models/model_e0_init.pt"
    print("  * [Checkpoint E0]: Initial untrained baseline saved.")

    t0 = time.time()
    action_counts = defaultdict(int)

    for h in range(1, total_hands + 1):
        rng = np.random.RandomState(1000000 + h)
        seat = h % 5
        st = game.new_initial_state()
        advance_chance_nodes(st, rng)

        # Sample opponents from the Self-Play League
        # 40% League Opponents (Model A, B, D), 60% Self-Play (Current E or Past E)
        opps = []
        for p in range(4):
            r_pool = random.random()
            if r_pool < 0.25: opps.append(("ModelA", model_a))
            elif r_pool < 0.50: opps.append(("ModelB", model_b))
            elif r_pool < 0.75: opps.append(("ModelD", model_d))
            else: opps.append(("Self", model_e))

        trajectory = []
        f_set = set(); dec = 0

        while not st.is_terminal():
            if st.is_chance_node(): advance_chance_nodes(st, rng); continue
            cp = st.current_player()
            l6 = True6ActionAdapter.get_legal_6actions(st, cp)

            if cp == seat:
                inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                eps = max(0.05, 0.20 - 0.15 * (h / total_hands))
                act, _ = model_e.act_train(inf, l6, epsilon=eps)
                action_counts[act] += 1
                trajectory.append((normalize_inf_state_v79(inf), act, l6))
                dec += 1
            else:
                opp_idx = cp if cp < seat else cp - 1
                opp_type, opp_agent = opps[opp_idx]
                if opp_type == "Self":
                    inf_opp = np.array(st.information_state_tensor(cp), dtype=np.float32)
                    act = opp_agent.act_eval(inf_opp, l6)
                else:
                    sp = extract_player_spent(st); tc = max(sp) - sp[cp]
                    if opp_type == "ModelD":
                        act = opp_agent.act(st, cp, l6, f_set, 1000000 + h, 0)
                        opp_agent.update_action(cp, tc, act)
                    else:
                        act = opp_agent.act(st, cp, l6, f_set, 1000000 + h, 0)

            if act == 0: f_set.add(cp)
            st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))

        # Terminal Reward
        ret = st.returns()[seat] / 100.0 # BB units
        r_scaled = ret / 100.0 # scale for neural stability

        for idx, (s_t, a_t, leg_t) in enumerate(trajectory):
            done = (idx == len(trajectory) - 1)
            ns_t = trajectory[idx+1][0] if not done else np.zeros_like(s_t)
            mask_t = [1 if i in leg_t else 0 for i in range(6)]
            model_e.replay_buffer.append((s_t, a_t, r_scaled if done else 0.0, ns_t, mask_t, done))

        # Gradient update every 5 hands
        if h % 5 == 0 and len(model_e.replay_buffer) >= 64:
            model_e.update_batch(batch_size=64)

        # Checkpoints at 2,000, 5,000, 10,000 hands
        if h in [2000, 5000, 10000]:
            name = f"E{1 if h==2000 else (2 if h==5000 else 3)}"
            path = f"v1014/models/model_{name.lower()}_{h//1000}k.pt"
            torch.save(model_e.state_dict(), path)
            checkpoints[h] = path
            tot_act = sum(action_counts.values())
            print(f"  * [Checkpoint {name} ({h:5d} hands)]: Saved -> {path} [소요: {time.time()-t0:.1f}s]")
            print(f"    - Action Dist: Fold={action_counts[0]/tot_act*100:4.1f}% | Call={action_counts[1]/tot_act*100:4.1f}% | Min={action_counts[2]/tot_act*100:4.1f}% | Half={action_counts[3]/tot_act*100:4.1f}% | Pot={action_counts[4]/tot_act*100:4.1f}% | AllIn={action_counts[5]/tot_act*100:4.1f}%")
            if model_e.losses_policy:
                print(f"    - Avg Policy Loss: {np.mean(model_e.losses_policy[-100:]):.4f} | Avg Value Loss: {np.mean(model_e.losses_value[-100:]):.4f}")

    return checkpoints

# =====================================================================
# EVALUATION & INDEPENDENT CHAMPIONSHIP
# =====================================================================
def evaluate_progression(game, checkpoint_paths, num_hands=1000):
    print("\n[*] Evaluating Milestone Progression (E0 vs E1 vs E2 vs E3 on 1,000 hands)...")
    models = {}
    for name, path in checkpoint_paths.items():
        m = ModelE_NeuralPlayer()
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        models[name] = m

    opp_pools = [("Standard", get_pool_standard), ("Aggressive", get_pool_aggressive), ("Dynamic", get_pool_dynamic)]
    results = defaultdict(list)

    base_seed = 20590001
    for h in range(num_hands):
        seed_h = base_seed + h
        seat = h % 5
        pool_name, pool_gen = opp_pools[h % 3]

        for name, m_agent in models.items():
            opps = pool_gen(seed_h)
            st = game.new_initial_state()
            advance_chance_nodes(st, np.random.RandomState(seed_h))
            f_set = set(); dec = 0
            while not st.is_terminal():
                if st.is_chance_node(): advance_chance_nodes(st, np.random.RandomState(seed_h)); continue
                cp = st.current_player()
                l6 = True6ActionAdapter.get_legal_6actions(st, cp)
                if cp == seat:
                    inf = np.array(st.information_state_tensor(seat), dtype=np.float32)
                    rng_eval = np.random.RandomState(seed_h * 101 + dec * 17)
                    act = m_agent.act_eval(inf, l6, rng=rng_eval)
                    dec += 1
                else:
                    act = opps[cp if cp < seat else cp - 1].select_action(st, cp, l6)
                if act == 0: f_set.add(cp)
                st.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st, cp, act))
            results[name].append(st.returns()[seat] / 100.0)

    summary = {}
    for name in models:
        rets = results[name]
        m_bb = np.mean(rets) * 100
        se_bb = stats.sem(rets) * 100
        summary[name] = {"mean": float(m_bb), "se": float(se_bb)}
        print(f"  * Milestone {name:<10}: {m_bb:+7.2f} ± {se_bb:5.2f} bb/100")

    return summary

def run_independent_championship(game, model_a, model_b, model_d, model_e, seeds, pools):
    print("\n[*] Running 5-Seed Large-Sample Championship (Model A vs Model D vs Model E3)...")
    results = defaultdict(lambda: {"A": [], "D": [], "E": []})
    seed_details = defaultdict(dict)

    for s_idx, s_base in enumerate(seeds, 1):
        print(f"\n--- [Evaluation Seed {s_idx}/5: Base {s_base}] ---")
        t_s = time.time()
        for pool_title, pool_gen in pools:
            rets_a = []; rets_d = []; rets_e = []
            md_eval = ModelD_SemanticController(model_a, model_b, window_size=50)

            for h in range(1000):
                seed_h = s_base + h
                seat = h % 5

                # Model A
                opps_a = pool_gen(seed_h)
                st_a = game.new_initial_state()
                advance_chance_nodes(st_a, np.random.RandomState(seed_h))
                f_a = set(); dec_a = 0
                while not st_a.is_terminal():
                    if st_a.is_chance_node(): advance_chance_nodes(st_a, np.random.RandomState(seed_h)); continue
                    cp = st_a.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_a, cp)
                    a = model_a.act(st_a, cp, l6, f_a, seed_h, dec_a) if cp == seat else opps_a[cp if cp<seat else cp-1].select_action(st_a, cp, l6)
                    if cp == seat: dec_a += 1
                    if a == 0: f_a.add(cp)
                    st_a.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_a, cp, a))
                rets_a.append(st_a.returns()[seat] / 100.0)

                # Model D
                opps_d = pool_gen(seed_h)
                st_d = game.new_initial_state()
                advance_chance_nodes(st_d, np.random.RandomState(seed_h))
                f_d = set(); dec_d = 0
                while not st_d.is_terminal():
                    if st_d.is_chance_node(): advance_chance_nodes(st_d, np.random.RandomState(seed_h)); continue
                    cp = st_d.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_d, cp)
                    sp = extract_player_spent(st_d); tc = max(sp) - sp[cp]
                    if cp == seat:
                        a = md_eval.act(st_d, cp, l6, f_d, seed_h, dec_d)
                        dec_d += 1
                    else:
                        a = opps_d[cp if cp<seat else cp-1].select_action(st_d, cp, l6)
                        md_eval.update_action(cp, tc, a)
                    if a == 0: f_d.add(cp)
                    st_d.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_d, cp, a))
                rets_d.append(st_d.returns()[seat] / 100.0)

                # Model E (E3)
                opps_e = pool_gen(seed_h)
                st_e = game.new_initial_state()
                advance_chance_nodes(st_e, np.random.RandomState(seed_h))
                f_e = set(); dec_e = 0
                while not st_e.is_terminal():
                    if st_e.is_chance_node(): advance_chance_nodes(st_e, np.random.RandomState(seed_h)); continue
                    cp = st_e.current_player()
                    l6 = True6ActionAdapter.get_legal_6actions(st_e, cp)
                    if cp == seat:
                        inf = np.array(st_e.information_state_tensor(seat), dtype=np.float32)
                        rng_e = np.random.RandomState(seed_h * 101 + dec_e * 17)
                        a = model_e.act_eval(inf, l6, rng=rng_e)
                        dec_e += 1
                    else:
                        a = opps_e[cp if cp<seat else cp-1].select_action(st_e, cp, l6)
                    if a == 0: f_e.add(cp)
                    st_e.apply_action(True6ActionAdapter.map_6action_to_engine_amount(st_e, cp, a))
                rets_e.append(st_e.returns()[seat] / 100.0)

            results[pool_title]["A"].extend(rets_a)
            results[pool_title]["D"].extend(rets_d)
            results[pool_title]["E"].extend(rets_e)

            ma_s = np.mean(rets_a) * 100; md_s = np.mean(rets_d) * 100; me_s = np.mean(rets_e) * 100
            seed_details[f"Seed_{s_idx}"][pool_title] = {"A": ma_s, "D": md_s, "E": me_s}
            print(f"  * {pool_title:<38}: Model A={ma_s:+7.2f} | Model D={md_s:+7.2f} | Model E3={me_s:+7.2f} bb/100")
        print(f"  [Seed {s_idx} 완료 소요 시간: {time.time()-t_s:.1f}s]")

    # Statistical Synthesis
    pool_summary = {}
    tot_a = []; tot_d = []; tot_e = []
    for pool_title in pools:
        p_name = pool_title[0]
        ra = results[p_name]["A"]; rd = results[p_name]["D"]; re = results[p_name]["E"]
        tot_a.extend(ra); tot_d.extend(rd); tot_e.extend(re)

        ma = np.mean(ra)*100; se_a = stats.sem(ra)*100
        md = np.mean(rd)*100; se_d = stats.sem(rd)*100
        me = np.mean(re)*100; se_e = stats.sem(re)*100

        diff_ea = [(e - a)*100 for a, e in zip(ra, re)]
        m_ea = np.mean(diff_ea); se_ea = stats.sem(diff_ea)
        p_ea = stats.t.sf(abs(m_ea/se_ea), df=len(diff_ea)-1)*2 if se_ea>0 else 1.0

        pool_summary[p_name] = {
            "model_a": {"mean": float(ma), "se": float(se_a)},
            "model_d": {"mean": float(md), "se": float(se_d)},
            "model_e": {"mean": float(me), "se": float(se_e)},
            "diff_ea": {"mean": float(m_ea), "se": float(se_ea), "p": float(p_ea)}
        }

    tot_ma = np.mean(tot_a)*100; tot_md = np.mean(tot_d)*100; tot_me = np.mean(tot_e)*100
    diff_tot_ea = [(e - a)*100 for a, e in zip(tot_a, tot_e)]
    m_tot_ea = np.mean(diff_tot_ea); se_tot_ea = stats.sem(diff_tot_ea)
    p_tot_ea = stats.t.sf(abs(m_tot_ea/se_tot_ea), df=len(diff_tot_ea)-1)*2 if se_tot_ea>0 else 1.0

    print("\n" + "=" * 80)
    print("V10.14 5-SEED INDEPENDENT CHAMPIONSHIP FINAL SYNTHESIS (15,000 Hands per Model)")
    print(f"  * Model A (Frozen Superstar) : {tot_ma:+7.2f} ± {stats.sem(tot_a)*100:5.2f} bb/100")
    print(f"  * Model D (Semantic Champ)  : {tot_md:+7.2f} ± {stats.sem(tot_d)*100:5.2f} bb/100")
    print(f"  * Model E3 (Self-Play Neural): {tot_me:+7.2f} ± {stats.sem(tot_e)*100:5.2f} bb/100 (E3 - A: {m_tot_ea:+6.2f} bb/100, p={p_tot_ea:.4f})")
    print("=" * 80)

    return {
        "pools": pool_summary,
        "total": {
            "model_a": float(tot_ma), "model_d": float(tot_md), "model_e": float(tot_me),
            "diff_ea": float(m_tot_ea), "se_ea": float(se_tot_ea), "p_ea": float(p_tot_ea)
        },
        "seed_details": seed_details
    }

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 80)
    print("POKER AI V10.14: SELF-PLAY LEAGUE BOOTSTRAP & NEW NEURAL PLAYER (MODEL E)")
    print("=" * 80)

    # 1. Load Frozen Models A, B, D
    game = get_5p_fullgame()
    bp = DeterministicTrueNFSP_Agent()
    bp.sl_net.load_state_dict(torch.load('v79/models/sys_a_pure_seed_42.pt', map_location='cpu'))

    q = MultiwayHeroExploiter()
    d = torch.load('v106/data/q_exploiter_15k_cache.pt', weights_only=False)
    for k, v in d.items(): q.q_table[k] = v
    print(f"[+] Loaded Q-table: {len(q.q_table)} states")

    ma = ModelA(bp, sims=300)
    mb = ModelB(bp, q, sims=300)
    md = ModelD_SemanticController(ma, mb, window_size=50)

    # 2. Phase 2 & 3 & 4: Model E Self-Play League Training
    model_e = ModelE_NeuralPlayer(state_dim=1417, action_dim=6, hidden_dim=256, lr=0.001)
    checkpoints = run_self_play_training(model_e, ma, mb, md, total_hands=10000)

    # 3. Phase 4: Progression Evaluation (E0, E1, E2, E3)
    prog_paths = {
        "E0 (0k)": "v1014/models/model_e0_init.pt",
        "E1 (2k)": "v1014/models/model_e1_2k.pt",
        "E2 (5k)": "v1014/models/model_e2_5k.pt",
        "E3 (10k)": "v1014/models/model_e3_10k.pt"
    }
    progression_results = evaluate_progression(game, prog_paths, num_hands=1000)

    # 4. Phase 5: 5-Seed Large-Sample Championship (Model A vs Model D vs Model E3)
    seeds = [20610001, 20620001, 20630001, 20640001, 20650001]
    pools = [
        ("Pool 1: Standard (CS 포함)", get_pool_standard),
        ("Pool 2: Aggressive Red-Team (CS 배제)", get_pool_aggressive),
        ("Pool 3: Dynamic Switching (적응형 상대)", get_pool_dynamic),
    ]
    champ_results = run_independent_championship(game, ma, mb, md, model_e, seeds, pools)

    # 5. Output JSON
    final_output = {
        "v1013_correction": {
            "clarification": "V10.13 reported Pool 1 D-A as overall D-A. Correct 3-pool integrated D-A is +18.92 bb/100 (t=1.84, p=0.1367).",
            "seed_integrated_differences": {
                "Seed_1": 3.79, "Seed_2": -4.37, "Seed_3": 45.06, "Seed_4": 41.56, "Seed_5": 8.56
            }
        },
        "progression": progression_results,
        "championship": champ_results
    }

    out_file = "v1014/data/v1014_training_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] V10.14 전체 결과 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
