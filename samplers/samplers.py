import torch
import numpy 
from typing import Callable, Optional
from tqdm import tqdm







def restart(
    mixture: torch.Tensor,
    noises: torch.Tensor,
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    cond_index: int = 0,
    s_churn: float = 40.0,  # > 0 to add randomness
    num_resamples: int = 2,
    use_tqdm: bool = False,
    gaussian: bool = False,
):
    
    # Set initial noise
    x = sigmas[0] * noises  # [batch_size, num-sources, sample-length]

    for i in tqdm(range(len(sigmas) - 1), disable=not use_tqdm):

        if cond is not None:
                noisey_cond = cond + torch.randn_like(cond) * sigma
                x[:, :cond_index] = noisey_cond

        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        x[:1] = mixture - x[1:].sum(dim=0, keepdim=True) #from the derivation of the dirac posterior [x1,x2,...,xn = y - sum(x_rest)]

        #One step Heun 
        score = sigma * denoise_fn(x, sigma = sigma) #compute p(x) -> prior on sources
        ds = score[1:] - score[:1] # compute posterior
        x[1:] -= ds * (sigma_next - sigma) #take the Euler step
         

        #Heun second-order correction
        if sigma_next != None or sigma_next != 0:
            score_next = sigma_next * denoise_fn(x, sigma = sigma_next) #compute p(x) -> prior on sources
            ds_next = score[1:] - score[:1] # compute posterior
            x[1:] -= ds * (sigma_next - sigma)*(0.5*ds_next + 0.5*ds) #take the Euler step
        
        


        

        
                


        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        score = (x - denoise_fn(x, sigma=sigma)) / sigma #solving dx/dt 
        ds = score[1:] - score[:1]
        x[1:] += ds * (sigma_next - sigma_hat) #take the euler step
