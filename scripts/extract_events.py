import pandas as pd

THRESH = 0.6

df = pd.read_csv("batch_predictions.csv")

events = []
active = False
start = 0

for _, r in df.iterrows():
    if r.marine_prob >= THRESH and not active:
        start = r.time_sec
        active = True
    elif r.marine_prob < THRESH and active:
        end = r.time_sec
        events.append([start, end, end-start])
        active = False

if active:
    end = df.iloc[-1].time_sec
    events.append([start, end, end-start])

out = pd.DataFrame(events, columns=["start_time_sec","end_time_sec","duration_sec"])
out.to_csv("marine_events.csv", index=False)

print("Marine activity events saved to: marine_events.csv")
print(out)
