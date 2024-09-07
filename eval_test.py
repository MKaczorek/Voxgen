import argparse
import hydra
import os
import torch
import pathlib
from tqdm import tqdm
from typing import Callable, Optional
from math import sqrt
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import soundfile as sf
from samplers.samplers import euler, heun, sde, restart_sampler

from medley_vox import MedleyVox

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]




if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("config", type=str, help="Path to config file")
  parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
  parser.add_argument("medleyvox", type=str, help="Path to MedleyVox dataset")
  parser.add_argument("-T", default=100, type=int, help="Number of diffusion steps")
  parser.add_argument("-S", default=40.0, type=float, help="S churn")
  parser.add_argument("--out", type=str, help="Output directory")
  parser.add_argument("--cond", action="store_true", help="Use conditioning")
  parser.add_argument(
      "--self-cond", action="store_true", help="Use self conditioning"
  )
  parser.add_argument("--hop-length", type=int, help="Hop length")
  parser.add_argument("--window", type=int, help="Window size")
  parser.add_argument("--full-duet", action="store_true", help="Drop duet songs")
  parser.add_argument("--retry", type=int, default=0, help="Retry")
  parser.add_argument("--outer-retry", type=int, default=0, help="Outer retry")
  parser.add_argument("--sampler", default='euler', type=str, help="sampler name")

  args = parser.parse_args()

  config_path, config_name = os.path.split(args.config)

  with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name=config_name)


  #restart info
  restart_info = '{"0": [3, 1, 1.209375, 2.560625], "1": [6, 1, 0.068125, 0.12], "2": [6, 5, 0.036875, 0.068125], "3": [6, 5, 0.01875, 0.036875], "4": [6, 20, 0.00375, 0.01875]}'
  
  #import lenght and sampling rate from the config file
  sr = cfg.sampling_rate
  length = cfg.length

  #esentially importing model trained on separated sources 
  model = hydra.utils.instantiate(cfg.model)
  vctk_checkpoint = torch.load(
        args.ckpt,
        map_location="cpu",
  )
  model.load_state_dict(vctk_checkpoint["state_dict"])
  diffusion_schedule = hydra.utils.instantiate(
        cfg.callbacks.audio_samples_logger.diffusion_schedule
  )

  model = model.cuda()
  model.eval()
  diffusion_schedule = diffusion_schedule.cuda()


  inner_denoise_fn = model.model.diffusion.diffusion.denoise_fn

  def denoise_fn(x, sigma):
        x = x.unsqueeze(1)
        return inner_denoise_fn(x, sigma=sigma).squeeze(1)

  sigmas = diffusion_schedule(args.T, "cuda")
  print(sigmas)

  dataset = MedleyVox(
        args.medleyvox,
        sample_rate=sr,
        drop_duet=not args.full_duet,
  )

  hop_length = length // 2 if args.hop_length is None else args.hop_length
  window_size = length if args.window is None else args.window
  loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
  s_churn = args.S

  if args.cond:
        print(window_size / sr, hop_length / sr)

  accumulate_metrics_mean = {}


  with tqdm(dataset) as pbar:
      for mix_num, (x_cpu, y_cpu, ids) in enumerate(pbar):
          x = x_cpu.cuda()
          y = y_cpu.cuda()
          n = y.shape[0]

          outer_trials = []
          for i in range(args.outer_retry + 1):
              if args.cond:
                  cond = torch.zeros(n, window_size).cuda()
                  sub_m = torch.zeros(window_size).cuda()
                  result = []
                  for sub_x, sub_y in zip(
                      torch.split(x, hop_length), torch.split(y, hop_length, 1)
                  ):
                      noise = torch.randn(n, window_size).cuda()
                      overlap_size = window_size - sub_x.numel()
                      if overlap_size > 0:
                          sub_m = torch.cat([sub_m[-overlap_size:], sub_x])
                          cond = cond[:, -overlap_size:]
                      else:
                          sub_m = sub_x
                          cond = None

                      trials = []
                      for i in range(args.retry + 1):
                          pred = sde(
                              sub_m,
                              noise,
                              denoise_fn,
                              sigmas,
                              s_churn=s_churn,
                              cond=cond,
                              cond_index=overlap_size,
                              use_tqdm=False,
                              gaussian=False,
                          )
                          sub_pred = pred[:, -sub_x.numel() :]

                          if args.retry > 0:
                              loss, align_pred = loss_func(
                                  sub_pred.unsqueeze(0),
                                  sub_y.unsqueeze(0),
                                  return_est=True,
                              )
                              trials.append((loss, align_pred.squeeze()))
                          else:
                              trials.append((0, sub_pred))

                      _, sub_pred = min(trials, key=lambda x: x[0])
                      cond = torch.cat(
                          ([] if cond is None else [cond])
                          + [sub_pred if args.self_cond else sub_y],
                          dim=1,
                      )
                      result.append(sub_pred)

                  result = torch.cat(result, dim=1)
              else:
                  original_length = x.numel()
                  padding = length - (original_length % length)
                  if padding < length:
                      x = torch.cat([x, x.new_zeros(padding)], dim=0)
                  if args.sampler == 'sde':
                    print('Sampler: SDE')
                    result = sde(
                        x,
                        torch.randn(n, x.numel()).cuda(),
                        denoise_fn,
                        sigmas,
                        use_tqdm=False,
                    )[:, :original_length]

                  if args.sampler == 'euler':
                    print('Sampler: Euler')
                    result = euler(
                        x,
                        torch.randn(n, x.numel()).cuda(),
                        denoise_fn,
                        sigmas,
                        use_tqdm=False,
                    )[:, :original_length]

                  if args.sampler == 'heun':
                    print('Sampler: Heun')
                    result = heun(
                        x,
                        torch.randn(n, x.numel()).cuda(),
                        denoise_fn,
                        sigmas,
                        use_tqdm=False,
                    )[:, :original_length]
                    
                  if args.sampler == 'restart':
                    print('Sampler: Restart')
                    result = restart_sampler(
                            denoise_fn = denoise_fn,
                            restart_info= restart_info,
                            noises = torch.randn(n, x.numel()).cuda(),
                            mixture = x,          
                        )[:, :original_length]
              loss, reordered_sources = loss_func(
                  result.unsqueeze(0), y.unsqueeze(0), return_est=True
              )
              outer_trials.append((loss, reordered_sources))

          _, reordered_sources = min(outer_trials, key=lambda x: x[0])
          est = reordered_sources.squeeze().cpu().numpy()

          utt_metrics = get_metrics(
              x_cpu.numpy(),
              y_cpu.numpy(),
              est,
              sample_rate=sr,
              metrics_list=COMPUTE_METRICS,
          )

          # calculate improvement
          for metric in COMPUTE_METRICS:
              v = utt_metrics.pop("input_" + metric)
              utt_metrics[metric + "i"] = utt_metrics[metric] - v

          for k, v in utt_metrics.items():
              if k not in accumulate_metrics_mean:
                  accumulate_metrics_mean[k] = 0

              accumulate_metrics_mean[k] += (v - accumulate_metrics_mean[k]) / (
                  mix_num + 1
              )

          pbar.set_postfix(accumulate_metrics_mean)

          if args.out is not None:
              out_dir = pathlib.Path(args.out) / f"medleyvox_{mix_num}"
              out_dir.mkdir(parents=True, exist_ok=True)

              sf.write(
                  out_dir / "mixture.wav",
                  x_cpu.numpy(),
                  sr,
                  "PCM_16",
              )

              for i, s in enumerate(est):
                  out_path = out_dir / f"{ids[i]}.wav"
                  sf.write(out_path, s, sr, "PCM_16")

  print(accumulate_metrics_mean)


