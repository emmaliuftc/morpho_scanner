import glob
import imageio
import os
import sys

def fix_gif_loops(directory):
    gifs = glob.glob(os.path.join(directory, "*.gif"))
    for gif_path in gifs:
        print(f"Fixing loop on {gif_path}...")
        try:
            frames = imageio.mimread(gif_path, memtest=False)
            # Re-save with loop=0
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
        except Exception as e:
            print(f"Error processing {gif_path}: {e}")

if __name__ == "__main__":
    for d in sys.argv[1:]:
        if os.path.exists(d):
            fix_gif_loops(d)
        else:
            print(f"Directory {d} not found.")
