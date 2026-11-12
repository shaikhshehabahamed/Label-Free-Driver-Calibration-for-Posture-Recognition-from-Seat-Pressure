"""Order-confound floor: how well does ELAPSED TIME ALONE predict the posture label for an unseen driver?

Every driver performed P1..P20 in the same order with similar durations, so elapsed time since the first posture is a proxy for the label.
Any pressure-based accuracy has to be read against this floor.  Leave-one-driver-out, same rows as the main experiments.
"""
import guard  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from common import RES, tag
from ladder import FPS, Stream

st = Stream(f=16)                          # pressure data not used; Stream only supplies the frame table / row selection
fr = st.fr
rows = st.rows(5.0)
first = fr[fr.label > 0].groupby("u").frame_id.min()
t = (fr.frame_id.values - fr.u.map(first).values) / FPS              # seconds since the driver's first labelled frame
y, u = fr.label.values[rows], fr.u.values[rows]
X = t[rows][:, None]
out = []
for s in np.unique(u):
    te = u == s
    m = RandomForestClassifier(100, min_samples_leaf=20, n_jobs=2, random_state=0).fit(X[~te], y[~te])
    out.append(dict(u=s, acc=accuracy_score(y[te], m.predict(X[te]))))
d = pd.DataFrame(out)
d.to_csv(RES / f"time_only_baseline{tag()}.csv", index=False)
print(f"time-only LOSO accuracy: {d.acc.mean():.3f} (sd {d.acc.std():.3f}) over {len(d)} drivers; chance 0.05")
