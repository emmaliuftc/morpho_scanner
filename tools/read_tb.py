import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def extract_tb(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found in", log_dir)
        return

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    
    print("\nTraining Progress (Error / PSNR Improvements):")
    print("-" * 50)
    
    if 'eval-metrics/PSNR' in tags:
        print("EVAL PSNR:")
        events = ea.Scalars('eval-metrics/PSNR')
        for e in events:
            print(f"  Step {e.step:5d}: {e.value:.2f} dB")
            
    if 'train-metrics/PSNR' in tags:
        print("\nTRAIN PSNR (Sampled):")
        events = ea.Scalars('train-metrics/PSNR')
        for i, e in enumerate(events):
            if i % (len(events)//5 + 1) == 0 or i == len(events)-1:
                print(f"  Step {e.step:5d}: {e.value:.2f} dB")
                
    if 'train-loss/loss' in tags:
        print("\nTRAIN TOTAL LOSS (Sampled):")
        events = ea.Scalars('train-loss/loss')
        for i, e in enumerate(events):
            if i % (len(events)//5 + 1) == 0 or i == len(events)-1:
                print(f"  Step {e.step:5d}: {e.value:.6f}")

if __name__ == "__main__":
    tb_dir = "/home/coding/github/morpho_scanner/0813_0726_cerf_rerun_10k/captures_0726_nerf_dataset/nerfacto/10k_run/tensorboard"
    extract_tb(tb_dir)
