import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

class WPO():
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
                 no_wasserstein=False,
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


        self.no_wasserstein=no_wasserstein
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def _actions_pdist(self, actions):
        return torch.cdist(actions, actions)

    def _sinkhorn_loss(self, old_actions, new_actions,
            eps=1e-2,
            niter=10,
            num_projections=64):
        ## NOTE!!!! this is sliced wasserstein not sinkhorn!!!
        ## change this name soon TODO
        # create matrix of unit vectors to project on
        # each column vector is one unit vector

        # each row of actions.loc is a single mu
        # so actions.loc @ slice_matrix should give batch x projection result

        # actions.scale is a matrix of size row x diag
        # result should be a matrix of size batch x num_projections
        # should just be actions.scale @ (projmat ** 2) taking advantage
        # of diag
        action_size = old_actions.loc.shape[1]
        # random vectors and normalize
        proj_vectors = torch.rand(action_size, num_projections).to(old_actions.loc.device)
        proj_vectors /= torch.norm(proj_vectors, dim=0).unsqueeze(0)

        sampling_rate = 100
        samples = torch.arange(0.1,1.0,1.0/sampling_rate).to(old_actions.loc.device)
        sampled_erfinv_coef = np.sqrt(2.0) * torch.erfinv(1.0 - 2.0*samples).view(1,1,-1)

        def projections(dist_object):
            locs = dist_object.loc @ proj_vectors
            scales = dist_object.scale @ (proj_vectors ** 2)
            return locs, scales

        
        old_projected_locs, old_projected_scales = projections(old_actions)
        new_projected_locs, new_projected_scales = projections(new_actions)

        def integration(old_projected_locs, old_projected_scales, new_projected_locs, new_projected_scales):
            total = old_projected_locs - new_projected_locs
            total = total.unsqueeze(2)
            total = total - old_projected_scales.unsqueeze(2)*sampled_erfinv_coef
            total += new_projected_scales.unsqueeze(2)*sampled_erfinv_coef
            total = torch.abs(total)
            return torch.mean(total, dim=2)
        # given a scalar mu/sigma, need closed form wasserstein-1 solution
        integral_results = integration(old_projected_locs, old_projected_scales, new_projected_locs, new_projected_scales)
        per_batch_sliced_wasserstein = torch.mean(integral_results, dim=1)
        total_loss = torch.mean(per_batch_sliced_wasserstein)

        return total_loss


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

                sinkhorn_loss = self._sinkhorn_loss(old_action_dist, action_dist)

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

                sinkhorn_item = sinkhorn_loss.item()
                self.optimizer.zero_grad()
                if sinkhorn_item < self.prox_target/1.5:
                    self.beta /= 2.0
                if sinkhorn_item > self.prox_target*1.5:
                    self.beta *= 2.0

                if self.no_wasserstein:
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                else:
                    (value_loss * self.value_loss_coef + action_loss + sinkhorn_loss*self.beta - dist_entropy * self.entropy_coef).backward()
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
