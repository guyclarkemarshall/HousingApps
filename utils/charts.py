import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def leaderboard_bar(d: pd.DataFrame):
    top = d.sort_values("proportion", ascending=False).head(15)
    fig = px.bar(
        top, x="proportion", y="landlord_name", orientation="h",
        hover_data=["landlord_code","landlord_type","region","group_size","sample_achieved","relevant_population"],
        labels={"proportion":"Proportion (%)","landlord_name":"Landlord"}
    )
    fig.update_layout(yaxis={"categoryorder":"total ascending"}, margin={"l":10,"r":10,"t":30,"b":30})
    return fig

def scatter_with_trend(d: pd.DataFrame, x: str):
    scat = d.dropna(subset=[x,"proportion"]).copy()
    fig = px.scatter(
        scat, x=x, y="proportion", hover_name="landlord_name",
        hover_data=["landlord_code","region","landlord_type","sample_achieved","group_size","relevant_population"],
        labels={x:x.replace("_"," ").title(),"proportion":"Proportion (%)"}
    )
    if len(scat) >= 2:
        xv = scat[x].astype(float).values
        yv = scat["proportion"].astype(float).values
        m, b = np.polyfit(xv, yv, 1)
        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
        ys = m*xs + b
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="trend"))
    return fig

def distribution_box(d: pd.DataFrame, split_by: str):
    bx = d.dropna(subset=["proportion"]).copy()
    if split_by not in bx.columns:
        split_by = "region"
    counts = bx[split_by].value_counts()
    keep = counts[counts >= 5].index
    bx = bx[bx[split_by].isin(keep)]
    if len(bx) > 0:
        med = bx.groupby(split_by)["proportion"].median().sort_values(ascending=False)
        bx[split_by] = pd.Categorical(bx[split_by], categories=med.index, ordered=True)
    fig = px.box(bx, x=split_by, y="proportion", points="outliers",
                 labels={"proportion":"Proportion (%)", split_by:split_by.replace("_"," ").title()})
    return fig
