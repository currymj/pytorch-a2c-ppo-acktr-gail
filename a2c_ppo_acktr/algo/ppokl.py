import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class PPOKL():

    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 beta,
                 prox_target,
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
        self.beta = beta
        self.prox_target = prox_target
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def _kl_loss(self, old_actions, new_actions):
        mu_diff = new_actions.loc - old_actions.loc
        middle_term = torch.sum((mu_diff * (1.0 / new_actions.scale)) * (mu_diff), dim=1)
        det_ratio = torch.log(torch.prod(old_actions.scale, dim=1) / torch.prod(new_actions.scale, dim=1))
        k = old_actions.loc.shape[1]
        cov_trace = torch.sum(old_actions.scale / new_actions.scale, dim=1)
        return torch.mean(0.5 * (cov_trace + middle_term - k + det_ratio), dim=0)


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
                # get new action_feats also


                # Reshape to do in a single forward pass for all steps
                values, action_dist, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions_with_feats(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # loss function here
                # this is the likelihood ratio computation
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ # this would be the "standard" loss
                action_loss = -surr1.mean() # summing over each point in trajectory

                kl_loss = self._kl_loss(old_action_dist, action_dist)

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

                kl_item = kl_loss.item()
                self.optimizer.zero_grad()
                #if sinkhorn_item < self.prox_target/1.5:
                #self.beta /= 2.0
                #if sinkhorn_item > self.prox_target*1.5:
                #self.beta *= 2.0

                (value_loss * self.value_loss_coef + action_loss + kl_loss*self.beta - dist_entropy * self.entropy_coef).backward()
                # if we're already doing sinkhorn do we need a separate entropy regularizer?

                # do we want to keep clipping gradient norm?
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        if kl_item < self.prox_target/1.5:
            self.beta /= 2.0
        if kl_item > self.prox_target*1.5:
            self.beta *= 2.0

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
