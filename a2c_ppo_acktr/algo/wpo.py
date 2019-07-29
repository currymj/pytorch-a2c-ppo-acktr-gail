import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class WPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 beta,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.beta = beta

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def _actions_pdist(self, actions):
        return torch.cdist(actions, actions)

    def _sinkhorn_loss(self, old_actions, new_actions, eps=1e-2, niter=10):
        raise NotImplementedError

    def _log_space_sinkhorn(self, actions, old_log_probs, new_log_probs, eps=1e-2, niter=10):
        raise NotImplementedError # this is no longer what we want, want to look at dists not logprobs
        # see "computational optimal transport" peyre and cuturi, pg. 76, eqns 4.43 and 4.44
        # initialize g to all ones
        # need min_row(self, mat, eps) and min_col(self, mat, eps)
        # for those, need softmin(self, vec, eps)
        # matrix C is euclidean costs, S is C_ij - f_i - g_j
        C = self._actions_pdist(actions)
        g = torch.ones_like(old_log_probs)
        a = old_log_probs.view(-1)
        b = new_log_probs.view(-1)
        f = torch.log(torch.ones_like(a) / len(a))
        g = torch.log(torch.ones_like(b) / len(b))

        def scaled_smat(f, g):
            smat = C - f.unsqueeze(1) - g.unsqueeze(0)
            scaled_smat = torch.exp(-smat / eps)
            return scaled_smat
        for n in range(niter):
            minrow_s = -eps*torch.log( torch.sum(scaled_smat(f, g), 1))
            f = minrow_s + f + eps*a
            mincol_s = -eps*torch.log( torch.sum(scaled_smat(f, g), 0))
            g = mincol_s + g + eps*b

        loss = (f @ torch.exp(a)) + (g @ torch.exp(b))

        # this might be correct but the given logprobs can't be used as
        # sinkhorn marginals, at least not in an obvious way
        return loss


    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        # save current policy here
        behavior_policy = copy.deepcopy(self.actor_critic)

        # does it need its own gradients or not? (probably not)
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample


                # get old action_feats here
                _, old_action_dist, _, _, _ = behavior_policy.evaluate_actions_with_feats(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, keep_grad=False)
                print(old_action_dist)
                # get new action_feats also


                # Reshape to do in a single forward pass for all steps
                values, action_dist, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions_with_feats(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                print(action_dist)


                # loss function here
                # this is the likelihood ratio computation
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ # this would be the "standard" loss
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ # this is the clipped loss
                # below is the min of the two
                action_loss = -torch.min(surr1, surr2).mean() # summing over each point in trajectory
                # loss function here

                # what we want to do for our loss function is:
                # compute the standard loss (surr1 above)
                # also compute the sinkhorn based penalty. what are the inputs to this?
                # -- for small # of discrete actions, just assume we have the matrix
                # -- left marginal is old policy (do we have this somewhere)? right marginal is new policy (how do we get this?)
                # -- what about case for continuous actions -- with gaussian policies maybe we can compute directly?
                # -- could we also simply: look at log probs for all actual actions taken, and compute pdist? is this defensible?

                # actions taken are taken under old policy, so it's potentially a biased sample
                # one supposes we would rather sample unif. at random to get completely unbiased



                print('actions_batch', actions_batch.shape)
                print('old_log_probs', old_action_log_probs_batch.shape)
                print('action_log_probs', action_log_probs.shape)



                sinkhorn_loss = self._sinkhorn_loss(old_action_dist, action_dist)
                print('sinkhorn_loss', sinkhorn_loss.item())

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss + sinkhorn_loss*self.beta).backward()
                 # if we're already doing sinkhorn do we need a separate entropy regularizer?

                 # do we want to keep clipping gradient norm?
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
