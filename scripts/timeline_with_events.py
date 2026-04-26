import pandas as pd
import matplotlib.pyplot as plt

pred = pd.read_csv("batch_predictions.csv")
events = pd.read_csv("marine_events_merged.csv")

plt.figure(figsize=(14,4))
plt.plot(pred.time_sec, pred.marine_prob, color="#00e5ff", lw=1.6)

plt.axhline(0.6, color="red", linestyle="--", alpha=0.6)

for _, r in events.iterrows():
    plt.axvspan(r.start_time_sec, r.end_time_sec, color="#00e5ff", alpha=0.18)

plt.ylim(0,1.05)
plt.xlabel("Time (seconds)")
plt.ylabel("Probability")
plt.title("Marine Biological Activity Timeline")
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("static/outputs/marine_timeline_with_events.png", dpi=150)
plt.close()

print("✅ Timeline saved")
