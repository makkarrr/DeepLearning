import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        # ==================================================== #

    # Calculate variance schedule
        beta_t = torch.linspace(beta_1, beta_T, T)

        # Calculate other terms based on variance schedule
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

        return {
            'beta_t': beta_t[t_s-1],
            'sqrt_beta_t': sqrt_beta_t[t_s-1],
            'alpha_t': alpha_t[t_s-1],
            'sqrt_alpha_bar': sqrt_alpha_bar[t_s-1],
            'oneover_sqrt_alpha': oneover_sqrt_alpha[t_s-1],
            'alpha_t_bar': alpha_t_bar[t_s-1],
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar[t_s-1]
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        mask_p = self.dmconfig.mask_p
        noise_loss = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  
        t_s = torch.randint(1, T+1, (images.shape[0],))

        sched = [self.scheduler(t) for t in t_s]

        sqrt_alpha_bar = torch.stack([sched[t]['sqrt_alpha_bar'] for t in range(images.shape[0]) ])
        sqrt_alpha_bar = sqrt_alpha_bar.to(device)

        sqrt_oneminus_alpha_bar = torch.stack([ sched[t]['sqrt_oneminus_alpha_bar'] for t in range(images.shape[0])])
        sqrt_oneminus_alpha_bar = sqrt_oneminus_alpha_bar.to(device)

        noise = torch.randn_like(images)
        noise = noise.to(device)

        x_t = images * sqrt_alpha_bar.view(-1,1,1,1) + noise * sqrt_oneminus_alpha_bar.view(-1,1,1,1)

        t_view = t_s/T
        t_view = t_view.view(-1,1,1,1)
        t_view = t_view.to(device)

        one_hot_conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float()
        # torch.eye(self.dmconfig.num_classes, device = device)[conditions]
        
        mask = torch.bernoulli(torch.full((images.shape[0],), 1-mask_p)).to(torch.int)
        mask = mask.unsqueeze(1).expand(-1, 10)
        mask = mask.to(device)
        one_hot_conditions_masked = one_hot_conditions * mask
        mask = mask - 1
        one_hot_conditions_masked = one_hot_conditions_masked + mask
        noise_loss = self.loss_fn(noise, self.network(x_t, t_view, one_hot_conditions_masked))
        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = conditions.shape[0]
        shape = (batch_size,self.dmconfig.num_channels, self.dmconfig.input_dim[0],self.dmconfig.input_dim[1])
        x = torch.randn(shape)
        x = x.to(device)
        
        with torch.no_grad():
          for t in range(T,0,-1):
            if t>1:
              z = torch.randn(shape)
            else:
              z = torch.zeros(shape)
            
            z = z.to(device)
            
            ts = torch.ones((batch_size,)) * float(t)/T
            ts = ts.view(batch_size,1,1,1)
            ts = ts.to(device)

            # if conditions.shape != (self.dmconfig.batch_size, self.dmconfig.num_classes):
            #   one_hot_conditions = torch.eye(self.dmconfig.num_classes, device = device)[conditions]
            if conditions.shape != (batch_size, self.dmconfig.num_classes):
              conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float()

            no_conditions = torch.ones((batch_size, self.dmconfig.num_classes)) * -1
            no_conditions = no_conditions.to(device)
            one_hot_conditions = conditions

            e = (1+omega)*self.network(x,ts,one_hot_conditions)
            e = e - omega * self.network(x,ts,no_conditions)
            e = e.to(device)
            sched = self.scheduler(t)

            oneover_sqrt_alpha = sched['oneover_sqrt_alpha']
            sqrt_oneminus_alpha_bar = float(sched['sqrt_oneminus_alpha_bar'])
            one_over_sqrt_oneminus_alpha_bar = 1/sqrt_oneminus_alpha_bar
            alpha_t = sched['alpha_t']
            sigma_t = sched['sqrt_beta_t']

            x = oneover_sqrt_alpha * (x - e * (1-alpha_t) * one_over_sqrt_oneminus_alpha_bar) + sigma_t * z
            pass
          pass

        generated_images = (x * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images




       # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_image
        # batch_size = conditions.size(0)

        # X_t = torch.randn(
        #     batch_size, 
        #     self.dmconfig.num_channels,
        #     self.dmconfig.input_dim[0],
        #     self.dmconfig.input_dim[1], 
        #     device=device
        # ).to(device)

        # with torch.no_grad():
        #     for t in reversed(range(1, T+1)):
        #         # compute nomralized time step
        #         nt = float(t / T)

        #         # get noise_schedule_dict 
        #         noise_schedule_dict = self.scheduler(t)

        #         # sample z from N(0, I)
        #         z = torch.randn_like(X_t) if t > 1 else torch.zeros_like(X_t)
        #         z = z.to(device)

        #         # turn conditions into one-hot encoding
        #         # NOTE: Somtimes conditions are already one-hot encoded
        #         if conditions.shape != (batch_size, self.dmconfig.num_classes):
        #             conditions = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float()

        #         masked_conditions = self.dmconfig.condition_mask_value * torch.ones_like(conditions)

        #         # import pdb; pdb.set_trace()
        #         # get conditional noise prediction
        #         normalized_time_steps = nt * torch.ones(batch_size, 1, dtype=torch.float, device=device)
        #         normalized_time_steps = normalized_time_steps.view(-1, 1, 1, 1)
        #         cond_noise_pred = self.network(X_t, normalized_time_steps, conditions)

        #         # get unconditional noise prediction
        #         noise_pred = self.network(X_t, normalized_time_steps, masked_conditions)

        #         # get weighted noise
        #         weighted_noise = (1 + omega) * cond_noise_pred - omega * noise_pred
        #         weighted_noise = weighted_noise.to(device)
        #         # update X_t
        #         weighted_noise_coeff = (
        #             (1 - noise_schedule_dict["alpha_t"]) 
        #             / noise_schedule_dict["sqrt_oneminus_alpha_bar"]
        #         )

        #         weighted_noise_coeff = weighted_noise_coeff

        #         std_t = noise_schedule_dict["sqrt_beta_t"]

        #         X_pre = X_t
        #         oneover_sqrt_alpha = noise_schedule_dict["oneover_sqrt_alpha"].view(-1, 1, 1, 1)
        #         oneover_sqrt_alpha = oneover_sqrt_alpha.to(device)

        #         # print(oneover_sqrt_alpha.device)
        #         # print(X_t.device)
        #         # print(weighted_noise_coeff.device)
        #         # print(weighted_noise.device)

        #         # import pdb; pdb.set_trace()
        #         X_t = (oneover_sqrt_alpha * (X_t 
        #             - ((1 - noise_schedule_dict["alpha_t"]) / noise_schedule_dict["sqrt_oneminus_alpha_bar"] * weighted_noise))
        #             + std_t * z)
                    
                

        # # import pdb; pdb.set_trace()
        # # ==================================================== #
        # generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        # return generated_images

        