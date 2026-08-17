import os
import argparse
from tensorboard.backend.event_processing import event_accumulator

def read_tb_logs(logdir):
    # Find all event files
    event_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(logdir) for f in filenames if 'tfevents' in f]
    if not event_files:
        print(f"No tfevents files found in {logdir}")
        return

    # Load the latest event file
    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading from: {latest_event_file}\n")
    
    # Load event accumulator
    ea = event_accumulator.EventAccumulator(latest_event_file)
    ea.Reload()

    # Get available scalar tags (e.g. Train/Loss, Eval/PSNR)
    tags = ea.Tags()['scalars']
    
    print(f"{'Metric':<30} | {'Latest Step':<15} | {'Value'}")
    print("-" * 65)
    
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        if events:
            latest_event = events[-1]
            print(f"{tag:<30} | {latest_event.step:<15} | {latest_event.value:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="Path to the directory containing tfevents files")
    args = parser.parse_args()
    read_tb_logs(args.logdir)
