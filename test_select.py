import sys
import select

# check stdin stream for readiness, timeout after 5 seconds
ready, _, _ = select.select([sys.stdin], [], [], 5.0)

if ready:
    user_input = sys.stdin.readline().strip()
    print(f"typed: {user_input}")
else:
    print("didn't type anything.")