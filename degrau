
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
import io, re

st.set_page_config(page_title="Ajuste em Lote: Degraus (1×/2×) + Tendência", page_icon="📊", layout="wide")

def logistic(z): return 1.0/(1.0+np.exp(-z))

def single_step_trend(t, m1, c1, m2, c2, t1, w):
    t = np.asarray(t, float); s = logistic((t - t1)/max(w, 1e-12))
    return (m1*t + c1)*(1.0 - s) + (m2*t + c2)*s

def double_step_trend(t, m1, c1, m2, c2, m3, c3, t1, t2, w1, w2):
    t = np.asarray(t, float)
    s1 = logistic((t - t1)/max(w1, 1e-12))
    s2 = logistic((t - t2)/max(w2, 1e-12))
    return (m1*t + c1)*(1.0 - s1) + (m2*t + c2)*s1*(1.0 - s2) + (m3*t + c3)*s2

def stats(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    resid = y - yhat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y))**2)) + 1e-15
    r2  = 1.0 - rss / tss
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    return dict(rss=rss, r2=r2, rmse=rmse, mae=mae, resid=resid)

def aicc(n, rss, k):
    if n - k - 1 <= 0: return np.inf
    return n*np.log(rss/max(n,1)) + 2*k + (2*k*(k+1))/(n - k - 1)

def thirds_linfits(x, y):
    n = len(x)
    if n < 6:
        m = (y[-1]-y[0]) / max(x[-1]-x[0], 1e-12)
        c = float(np.median(y) - m*np.median(x))
        return (m,c), (m,c), (m,c)
    s1, s2 = n//3, 2*n//3
    def linfit(xx, yy):
        A = np.vstack([xx, np.ones_like(xx)]).T
        m, c = np.linalg.lstsq(A, yy, rcond=None)[0]
        return float(m), float(c)
    return linfit(x[:s1], y[:s1]), linfit(x[s1:s2], y[s1:s2]), linfit(x[s2:], y[s2:])

def deriv_peaks(x, y, k=2):
    dv = np.gradient(y, x)
    idx = np.argsort(np.abs(dv))[-k:]
    idx = np.unique(np.clip(idx, 0, len(x)-1)); idx.sort()
    return idx

def fit_single(x, y, w_min, w_max):
    (m1,c1), (m2,c2), _ = thirds_linfits(x, y)
    idx = deriv_peaks(x, y, 1)
    t1g = float(x[idx[0]]) if len(idx) else float(np.quantile(x, 0.5))
    p0 = np.array([m1, c1, m2, c2, t1g, max(w_min, 0.01*(x[-1]-x[0]))], float)
    def obj(p):
        m1, c1, m2, c2, t1, w = p
        pen = 0.0
        if not (x[0] <= t1 <= x[-1]): pen += 1e6
        if not (w_min <= w <= w_max): pen += 1e6
        yhat = single_step_trend(x, m1, c1, m2, c2, t1, max(w, w_min))
        return np.sum((y - yhat)**2) + pen
    res = minimize(obj, p0, method="Nelder-Mead",
                   options=dict(maxiter=10000, xatol=1e-9, fatol=1e-9))
    p = res.x
    p[4] = float(np.clip(p[4], x[0], x[-1]))
    p[5] = float(np.clip(p[5], w_min, w_max))
    yhat = single_step_trend(x, *p)
    S = stats(y, yhat); S.update(dict(params=p.tolist(), yhat=yhat, k=6, model="Degrau Simples + Tendência"))
    return S

def fit_double(x, y, w_min, w_max):
    (m1,c1), (m2,c2), (m3,c3) = thirds_linfits(x, y)
    idx = deriv_peaks(x, y, 2)
    if len(idx) < 2:
        t1g, t2g = float(np.quantile(x, 0.4)), float(np.quantile(x, 0.6))
    else:
        t1g, t2g = float(x[idx[0]]), float(x[idx[1]])
        if t2g - t1g < 0.02*(x[-1]-x[0]): t1g, t2g = float(np.quantile(x, 0.35)), float(np.quantile(x, 0.65))
    w1g = w2g = max(w_min, 0.01*(x[-1]-x[0]))
    p0 = np.array([m1, c1, m2, c2, m3, c3, t1g, t2g, w1g, w2g], float)
    def obj(p):
        m1, c1, m2, c2, m3, c3, t1, t2, w1, w2 = p
        pen = 0.0
        if not (x[0] <= t1 <= x[-1]): pen += 1e6
        if not (x[0] <= t2 <= x[-1]): pen += 1e6
        if t2 <= t1: pen += 1e6*(1.0 + (t1 - t2))
        if not (w_min <= w1 <= w_max): pen += 1e6
        if not (w_min <= w2 <= w_max): pen += 1e6
        yhat = double_step_trend(x, m1, c1, m2, c2, m3, c3, t1, t2, max(w1, w_min), max(w2, w_min))
        return np.sum((y - yhat)**2) + pen
    res = minimize(obj, p0, method="Nelder-Mead",
                   options=dict(maxiter=15000, xatol=1e-9, fatol=1e-9))
    p = res.x
    p[6] = float(np.clip(p[6], x[0], x[-1]))
    p[7] = float(np.clip(p[7], x[0], x[-1]))
    if p[7] <= p[6]: p[7] = min(x[-1], p[6] + 0.02*(x[-1]-x[0]))
    p[8] = float(np.clip(p[8], w_min, w_max))
    p[9] = float(np.clip(p[9], w_min, w_max))
    yhat = double_step_trend(x, *p)
    S = stats(y, yhat); S.update(dict(params=p.tolist(), yhat=yhat, k=10, model="Degrau Duplo + Tendência"))
    return S

st.title("Ajuste em Lote • Degrau Simples vs Duplo + Tendência (Auto)")

with st.sidebar:
    st.header("Entrada")
    up = st.file_uploader("Arquivo Excel (.xlsx)", type=["xlsx"])
    sheet = st.text_input("Nome da planilha (opcional)", "")
    mode = st.radio("Formato dos dados", ["Pares adjacentes (x,y)", "Tempo comum + múltiplas colunas de Y"], index=0)
    st.caption("• Pares adjacentes: [col0,col1], [col2,col3], ...  • Tempo comum: 1 coluna X e várias Y.")
    st.header("Parâmetros do ajuste")
    width_min_ms = st.number_input("Largura mínima (×10⁻³ s)", min_value=0.0, value=0.1, step=0.05, format="%.2f")
    width_max_frac = st.slider("Largura máx. (fração do span do tempo)", 0.01, 0.50, 0.20, step=0.01)
    aicc_margin = st.slider("Margem AICc para preferir 2 degraus", 0.0, 5.0, 2.0, step=0.5)
    show_top = st.number_input("Máx. gráficos mostrados", min_value=1, max_value=30, value=8, step=1)

def clean_series(s):
    if s.dtype.kind in "OUS":
        s = s.astype(str).str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def pick_columns_ui(df, mode):
    cols = list(df.columns)
    if mode == "Pares adjacentes (x,y)":
        pairs = [(i, i+1) for i in range(0, len(cols)-1, 2)]
        if not pairs: return [], None
        labels = [f"cols[{i},{i+1}]  —  {cols[i]} | {cols[i+1]}" for i,_ in pairs]
        sel = st.multiselect("Escolha quais pares ajustar", options=list(range(len(pairs))), default=list(range(len(pairs))), format_func=lambda k: labels[k])
        return [pairs[k] for k in sel], None
    else:
        xcol = st.selectbox("Coluna de tempo (X)", cols, index=0)
        ycols = st.multiselect("Colunas de valor (Y)", [c for c in cols if c != xcol], default=[c for c in cols if c != xcol])
        return None, (xcol, ycols)

def tag(ds):  # compact suffix for wide columns
    s = re.sub(r"[^\w]+","_", str(ds)).strip("_")
    return s[:18]

if up is not None:
    try:
        xl = pd.ExcelFile(up)
        sh = sheet if sheet and sheet in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sh)
    except Exception as e:
        st.error(f"Erro ao ler Excel: {e}"); st.stop()

    df = df.copy()
    for c in df.columns: df[c] = clean_series(df[c])

    w_min_global = max(1e-9, (width_min_ms*1e-3))

    datasets = []
    if mode == "Pares adjacentes (x,y)":
        pairs, _ = pick_columns_ui(df, mode)
        if not pairs: st.warning("Nenhum par selecionado."); st.stop()
        for i,j in pairs:
            xi = df.iloc[:, i].astype(float).to_numpy()
            yi = df.iloc[:, j].astype(float).to_numpy()
            ok = np.isfinite(xi) & np.isfinite(yi)
            xi = xi[ok]; yi = yi[ok]
            if len(xi) >= 10:
                datasets.append((f"cols[{i},{j}]  —  {df.columns[i]} | {df.columns[j]}", xi, yi))
    else:
        _, (xcol, ycols) = pick_columns_ui(df, mode)
        xbase = df[xcol].astype(float).to_numpy()
        okx = np.isfinite(xbase); xbase = xbase[okx]
        for yname in ycols:
            yv = df[yname].astype(float).to_numpy()
            yv = yv[okx]
            ok = np.isfinite(xbase) & np.isfinite(yv)
            xi = xbase[ok]; yi = yv[ok]
            if len(xi) >= 10:
                datasets.append((f"{yname} (X={xcol})", xi, yi))

    if not datasets: st.warning("Sem datasets válidos."); st.stop()

    results, preds_best, preds_both = [], [], []
    wide_best_blocks, wide_both_blocks = [], []

    for name, x, y in datasets:
        idx = np.argsort(x); x = x[idx]; y = y[idx]
        T = float(x[-1]-x[0]) if len(x) > 1 else 1.0
        w_min = w_min_global; w_max = max(w_min*1.1, width_max_frac*T)

        S1 = fit_single(x, y, w_min, w_max)
        S2 = fit_double(x, y, w_min, w_max)
        a1 = aicc(len(x), S1["rss"], S1["k"]); a2 = aicc(len(x), S2["rss"], S2["k"])
        t_sep_ok = abs(S2["params"][7] - S2["params"][6]) > 0.02*T
        w_ok = (S2["params"][8] <= w_max) and (S2["params"][9] <= w_max)
        use_two = (a2 + aicc_margin < a1) and t_sep_ok and w_ok
        best = S2 if use_two else S1
        best = dict(best); best.update(dict(dataset=name, aicc_single=a1, aicc_double=a2, use_two=bool(use_two)))
        results.append(best)

        preds_best.append(pd.DataFrame({"dataset": name, "x": x, "y": y, "yfit": best["yhat"]}))
        preds_both.append(pd.DataFrame({"dataset": name, "x": x, "y": y, "yfit_single": S1["yhat"], "yfit_double": S2["yhat"]}))

        # wide blocks
        t = tag(name)
        wide_best_blocks.append(pd.DataFrame({"x_"+t: x, "y_"+t: y, "yfit_"+t: best["yhat"]}))
        wide_both_blocks.append(pd.DataFrame({"x_"+t: x, "y_"+t: y, "yfit1_"+t: S1["yhat"], "yfit2_"+t: S2["yhat"]}))

    results_df = pd.DataFrame(results)

    def expand_params(row):
        if row["use_two"]: lbls = ["m1","c1","m2","c2","m3","c3","t1","t2","width1","width2"]
        else: lbls = ["m1","c1","m2","c2","t1","width"]
        vals = row["params"]; d = {k: (vals[i] if i < len(vals) else np.nan) for i,k in enumerate(lbls)}
        return pd.Series(d)

    par_df = results_df.apply(expand_params, axis=1)
    out_df = pd.concat([results_df.drop(columns=["params","yhat","resid"]), par_df], axis=1)

    st.success(f"Concluído: {len(results)} datasets ajustados.")
    st.dataframe(out_df[["dataset","model","r2","rmse","mae","use_two"]].sort_values("dataset"), use_container_width=True, hide_index=True)

    # Plots
    show_df = out_df.sort_values("dataset").reset_index(drop=True)
    to_show = min(len(show_df), int(show_top))
    if to_show > 0:
        tabs = st.tabs([f"{i+1}. {show_df.loc[i,'dataset']}" for i in range(to_show)])
        for i, tab in enumerate(tabs):
            with tab:
                name = show_df.loc[i, "dataset"]
                row = next(r for r in results if r["dataset"] == name)
                x = preds_best[i]["x"].values
                y = preds_best[i]["y"].values
                yfit = row["yhat"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Dados", marker=dict(size=5, opacity=0.6)))
                fig.add_trace(go.Scatter(x=x, y=yfit, mode="lines", name="Curva Ajustada", line=dict(width=3)))
                fig.update_layout(height=460, margin=dict(l=20,r=20,t=60,b=40),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                                  xaxis_title="Tempo (s)", yaxis_title="Valor",
                                  title=f"{row['model']} • {name} (R² = {row['r2']:.4f})")
                st.plotly_chart(fig, use_container_width=True)

    # Exports
    with st.expander("Exportar resultados"):
        best_long = pd.concat(preds_best, ignore_index=True)
        both_long = pd.concat(preds_both, ignore_index=True)
        wide_best = pd.concat([b.reset_index(drop=True) for b in wide_best_blocks], axis=1)
        wide_both = pd.concat([b.reset_index(drop=True) for b in wide_both_blocks], axis=1)

        st.markdown("**CSV (longos):**")
        st.download_button("Parâmetros (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                           file_name="parametros_batch.csv", mime="text/csv")
        st.download_button("Predições • Melhor modelo (CSV)", best_long.to_csv(index=False).encode("utf-8"),
                           file_name="predicoes_best_batch.csv", mime="text/csv")
        st.download_button("Predições • Ambos os modelos (CSV)", both_long.to_csv(index=False).encode("utf-8"),
                           file_name="predicoes_both_batch.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("**Excel (inclui formatos LONGO e LARGO):**")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            out_df.to_excel(writer, sheet_name="parametros", index=False)
            best_long.to_excel(writer, sheet_name="predicoes_best", index=False)
            both_long.to_excel(writer, sheet_name="predicoes_both", index=False)
            wide_best.to_excel(writer, sheet_name="curvas_best_wide", index=False)
            wide_both.to_excel(writer, sheet_name="curvas_both_wide", index=False)
        st.download_button("Baixar tudo (Excel .xlsx)",
                           buf.getvalue(),
                           file_name="ajustes_batch_v3.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Envie um arquivo Excel (.xlsx). No modo 'Pares adjacentes', os dados são lidos como [x,y],[x,y], ...")
