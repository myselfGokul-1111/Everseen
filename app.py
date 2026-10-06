"""Step 4: streamlit run app.py"""
import json, pandas as pd, streamlit as st, plotly.express as px
st.set_page_config(layout="wide"); st.title("Dwell-Time & Bottleneck Mapper")
m = pd.read_csv("out/zone_metrics.csv"); j = pd.read_csv("out/journeys.csv"); flags = json.load(open("out/flags.json"))
c = st.columns(4)
c[0].metric("Tracked customers", j.track_id.nunique()); c[1].metric("Zones", len(m))
c[2].metric("Avg journey (s)", round(j.duration_s.mean(), 1)); c[3].metric("Issues flagged", len(flags))
l, r = st.columns(2)
l.subheader("Foot-traffic heatmap"); l.image("out/heatmap.png")
r.subheader("Flagged issues")
for f in flags: (r.error if f["type"] == "BOTTLENECK" else r.warning)(f"**{f['type']}** - {f['reason']}")
if not flags: r.success("No issues flagged")
st.subheader("Zone metrics"); st.dataframe(m, use_container_width=True)
a, b = st.columns(2)
a.plotly_chart(px.bar(m, x="zone", y="avg_dwell_s", title="Average dwell time (s)"), use_container_width=True)
top = j.path.value_counts().head(8).reset_index(); top.columns = ["path", "customers"]
b.plotly_chart(px.bar(top, x="customers", y="path", orientation="h", title="Top customer journeys"), use_container_width=True)
