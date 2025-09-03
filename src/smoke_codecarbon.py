from codecarbon import EmissionsTracker
import time

# Simple test: track a short busy-loop
with EmissionsTracker(measure_power_secs=1) as tracker:
    # small CPU-bound task
    s = 0
    for i in range(10_000_000):
        s += i
    time.sleep(0.5)
print("Done (check CodeCarbon console output or codecarbon.cache)")
