import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class WPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def _actions_pdist(self, actions):
        return torch.cdist(actions, actions)

    def _log_space_sinkhorn(self, actions, old_log_probs, new_log_probs):
        raise NotImplementedError

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

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

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

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

                # consider that last case: we'd have old_action_log_probs_batch, action_log_probs (the two marginals), and actions_batch (actions taken)
                # compute pdist matrix on actions_batch
                # use log space sinkhorn on each log_probs to compute sinkhorn_penalty
                # add onto action_loss

                # considering last case: the whole point of this is that the state distributions are similar enough
                # that we don't have to do any IS reweighting to deal with them. so sampling under one policy should be good
                # enough, I think? (not sure if possible to reweight the actions)

                print('actions_batch', actions_batch.shape)
                print('old_log_probs', old_action_log_probs_batch.shape)
                print('action_log_probs', action_log_probs.shape)


                actions_pairwise = self._actions_pdist(actions_batch)
                print('actions_pairwise', actions_pairwise.shape)

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
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
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
