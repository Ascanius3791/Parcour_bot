"""PPO for the parkour humanoid: networks, normalisers, the update and policy export.

Architecture (actor and critic are separate networks of the same shape):
  observation -> running mean/std normalisation, clipped to +-10
  -> first layer with one linear block per observation part (body sensors, rays,
     next obstacles, route) so the 1089 ray inputs do not dominate the parameter count
  -> ELU -> 256 -> ELU -> 128 -> ELU -> output
The actor outputs the mean of a Gaussian over target joint angles in [-1, 1]; its standard
deviation is a learned parameter per action.
"""

import math

import numpy as np
import torch
from torch import nn

BLOCK_WIDTHS = (256, 64, 48, 16)  # first-layer width per observation part
HIDDEN = (256, 128)
OBS_CLIP = 10.0


class RunningNorm:
    """Running mean / variance (parallel Welford), float64."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0)
        delta = mean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        return np.clip((x - self.mean) / self.std(), -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class RewardScaler:
    """Divides rewards by the running std of the discounted return (per env)."""

    def __init__(self, num_envs, gamma):
        self.gamma = gamma
        self.ret = np.zeros(num_envs, np.float64)
        self.norm = RunningNorm(())

    def __call__(self, reward, done):
        self.ret = self.ret * self.gamma + reward
        self.norm.update(self.ret)
        self.ret[done] = 0.0
        return (reward / self.norm.std()).astype(np.float32)

    def resize(self, num_envs):
        self.ret = np.zeros(num_envs, np.float64)


class Net(nn.Module):
    def __init__(self, segments, out, out_gain):
        super().__init__()
        self.segments = list(segments)
        self.blocks = nn.ModuleList(nn.Linear(n, w) for n, w in zip(segments, BLOCK_WIDTHS))
        layers, width = [], sum(BLOCK_WIDTHS)
        for h in HIDDEN:
            layers += [nn.Linear(width, h), nn.ELU()]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, out)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.orthogonal_(mod.weight, math.sqrt(2))
                nn.init.zeros_(mod.bias)
        nn.init.orthogonal_(self.head.weight, out_gain)

    def forward(self, x):
        parts = torch.split(x, self.segments, dim=-1)
        h = torch.cat([b(p) for b, p in zip(self.blocks, parts)], -1)
        return self.head(self.trunk(nn.functional.elu(h)))

    def dense_layers(self):
        """The network as plain dense layers (the block layer as one block-diagonal matrix)."""
        n_in, n_out = sum(self.segments), sum(BLOCK_WIDTHS)
        w0 = torch.zeros(n_out, n_in)
        b0 = torch.cat([b.bias for b in self.blocks])
        r = c = 0
        for b, n, w in zip(self.blocks, self.segments, BLOCK_WIDTHS):
            w0[r:r + w, c:c + n] = b.weight
            r, c = r + w, c + n
        layers = [(w0, b0)]
        layers += [(m.weight, m.bias) for m in self.trunk if isinstance(m, nn.Linear)]
        layers.append((self.head.weight, self.head.bias))
        return [(w.detach().float(), b.detach().float()) for w, b in layers]


class ActorCritic(nn.Module):
    def __init__(self, segments, act_size, init_std):
        super().__init__()
        self.actor = Net(segments, act_size, 0.01)
        self.critic = Net(segments, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_size,), math.log(init_std)))

    def dist(self, obs):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class PPO:
    def __init__(self, model, lr=3e-4, epochs=5, minibatches=4, clip=0.2, gamma=0.99,
                 lam=0.95, value_coef=1.0, entropy_coef=0.001, max_grad=1.0, desired_kl=0.01):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.lr = lr
        self.epochs, self.minibatches, self.clip = epochs, minibatches, clip
        self.gamma, self.lam = gamma, lam
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.max_grad, self.desired_kl = max_grad, desired_kl

    def advantages(self, rewards, values, dones, last_value):
        """GAE. rewards, values, dones: [T, N]; last_value: [N]."""
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + values

    def update(self, batch):
        """batch: dict of flat tensors obs, act, logp, value, ret, adv, mu, std."""
        n = batch["obs"].shape[0]
        mb = n // self.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for k in range(self.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                obs, act = batch["obs"][idx], batch["act"][idx]
                dist = self.model.dist(obs)
                logp = dist.log_prob(act).sum(-1)
                with torch.no_grad():  # KL(old || new) of diagonal Gaussians, adapts the lr
                    mu_o, std_o = batch["mu"][idx], batch["std"][idx]
                    kl = (torch.log(dist.stddev / std_o)
                          + (std_o ** 2 + (mu_o - dist.mean) ** 2) / (2 * dist.stddev ** 2) - 0.5).sum(-1).mean()
                    if kl > 2 * self.desired_kl:
                        self.lr = max(1e-6, self.lr / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                adv = batch["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = torch.exp(logp - batch["logp"][idx])
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                v = self.model.value(obs)
                v_old, ret = batch["value"][idx], batch["ret"][idx]
                v_clip = v_old + (v - v_old).clamp(-self.clip, self.clip)
                value_loss = torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
                self.opt.step()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl.item()
                updates += 1
        return {k: v / updates for k, v in stats.items()}


def export_policy(path, model, obs_norm, meta):
    """Writes the actor in the format read by src/core/policy.cpp."""
    layers = model.actor.dense_layers()
    sizes = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    header = ["parcour_policy 1", "layers " + " ".join(map(str, sizes)), "activation elu",
              f"obs_clip {OBS_CLIP}"]
    header += [f"{k} {v}" for k, v in meta.items()]
    header.append("end")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(obs_norm.mean.astype("<f4").tobytes())
        f.write(obs_norm.std().astype("<f4").tobytes())
        for w, b in layers:
            f.write(w.numpy().astype("<f4").tobytes())
            f.write(b.numpy().astype("<f4").tobytes())
