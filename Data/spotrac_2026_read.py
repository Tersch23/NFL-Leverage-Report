import pandas as pd

cb = pd.read_csv("spotrac_2026_CB.csv")
lb = pd.read_csv("spotrac_2026_LB.csv")

print("CB rows:", len(cb))
print("LB rows:", len(lb))

print(cb.head(3))
print(lb.head(3))
