import pandas as pd

df = pd.read_csv("marine_events.csv").sort_values("start_time_sec")

merged = []
s, e = None, None

for _, r in df.iterrows():
    if s is None:
        s, e = r.start_time_sec, r.end_time_sec
    elif r.start_time_sec <= e + 0.25:
        e = max(e, r.end_time_sec)
    else:
        merged.append([s, e, e-s])
        s, e = r.start_time_sec, r.end_time_sec

merged.append([s, e, e-s])

out = pd.DataFrame(merged, columns=["start_time_sec","end_time_sec","duration_sec"])
out.to_csv("marine_events_merged.csv", index=False)

print("Merged events saved to: marine_events_merged.csv")
print(out)
