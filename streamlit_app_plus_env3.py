import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="VOC & 환경 데이터 시각화", layout="wide")
st.title("🌿 식물 VOC & 환경 데이터 시각화")

# =========================================================
# 환경 데이터 모드 (데모 집중): 데이터셋 선택 + 반복(Rep) 지원
# =========================================================
st.sidebar.header("📁 환경 데이터 업로드")
tomato_file = st.sidebar.file_uploader("토마토 환경 (xlsx/xls/csv)", type=["xlsx","xls","csv"], key="env_tomato")
larva_file  = st.sidebar.file_uploader("애벌레 환경 (xlsx/xls/csv)", type=["xlsx","xls","csv"], key="env_larva")

@st.cache_data(show_spinner=False)
def read_env(file):
    if file is None:
        return None
    name = file.name.lower()
    b = file.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(b))
    else:
        xlf = pd.ExcelFile(io.BytesIO(b))
        df = xlf.parse(xlf.sheet_names[0])
    return df

df_t = read_env(tomato_file)
df_l = read_env(larva_file)

st.caption("업로드 후 자동으로 Date/Time/Timestamp, Temperature (℃), Relative humidity (%), PAR Light(광도), Repetition을 감지/표준화합니다.")

ENV_CANON = {
    "Date": ["date","날짜"],
    "Time": ["time","시간"],
    "Timestamp": ["timestamp","datatime","datetime","일시","측정시각"],
    "Temperature (℃)": ["temperature (℃)","temperature","temp","온도"],
    "Relative humidity (%)": ["relative humidity (%)","humidity","humid","rh","습도"],
    "PAR Light (μmol·m2·s-1)": ["par light (μmol·m2·s-1)","par","ppfd","광도","light","light (μmol m-2 s-1)","light (μmol·m2·s-1)","light (μmol m−2 s−1)"],
    "Repetition": ["repetition","replicate","rep","반복","반복수"],
}
def normalize(s):
    return str(s).strip().lower().replace("_"," ").replace("-"," ").replace("−","-")

def env_standardize_columns(df):
    if df is None:
        return None
    col_map = {}
    for c in df.columns:
        lc = normalize(c)
        mapped = None
        for canon, aliases in ENV_CANON.items():
            if lc == normalize(canon) or lc in [normalize(a) for a in aliases]:
                mapped = canon
                break
        if mapped:
            col_map[c] = mapped
    if col_map:
        df = df.rename(columns=col_map)
    return df

def coerce_ts(df):
    if df is None:
        return None
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    elif "Date" in df.columns and "Time" in df.columns:
        ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    elif "Date" in df.columns:
        ts = pd.to_datetime(df["Date"], errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    df = df.assign(__ts__=ts)
    df = df.dropna(subset=["__ts__"]).sort_values("__ts__")
    for col in ["Temperature (℃)", "Relative humidity (%)", "PAR Light (μmol·m2·s-1)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_t = env_standardize_columns(df_t)
df_l = env_standardize_columns(df_l)
df_t = coerce_ts(df_t)
df_l = coerce_ts(df_l)

# ----- 데이터셋 선택 -----
available_datasets = []
if df_t is not None and not df_t.empty:
    available_datasets.append("Tomato")
if df_l is not None and not df_l.empty:
    available_datasets.append("Larva")

if not available_datasets:
    st.info("좌측에서 **토마토** 또는 **애벌레** 환경 파일을 업로드하세요.")
    st.stop()

dataset = st.radio("분석할 데이터셋 선택", options=available_datasets, horizontal=True)
df_env = df_t if dataset=="Tomato" else df_l

# ----- 반복 선택 + 겹치는 기간(반복 간) 옵션 -----
has_rep = "Repetition" in (df_env.columns if df_env is not None else [])
rep_values = []
if has_rep:
    rep_values = sorted(df_env["Repetition"].dropna().astype(str).unique().tolist())
rep_choice = st.selectbox("🔁 Repetition 선택", ["전체(통합)"] + rep_values if has_rep else ["(반복 없음)"], index=0, disabled=not has_rep)
use_overlap_reps = st.checkbox("반복 간 '겹치는 기간'만 사용(통합일 때)", value=True, disabled=(not has_rep or rep_choice!="전체(통합)"),
                               help="반복마다 기록 기간이 다르면 교집합 시간만으로 평균/표준편차를 계산합니다.")

# ----- 1시간 리샘플링 기반 파생지표 -----
RES_RULE = "60min"
exclude_zero_for_mean = st.checkbox("광 평균/표준편차 계산 시 0(불꺼짐) 제외", value=True)

def hourly_metrics(df):
    idx = df.set_index("__ts__").sort_index()
    base = idx.resample(RES_RULE).asfreq().index
    out = {}
    # 온/습
    if "Temperature (℃)" in idx.columns:
        out["temp_mean"] = idx["Temperature (℃)"].resample(RES_RULE).mean()
        out["temp_sd"]   = idx["Temperature (℃)"].resample(RES_RULE).std()
    if "Relative humidity (%)" in idx.columns:
        out["hum_mean"] = idx["Relative humidity (%)"].resample(RES_RULE).mean()
        out["hum_sd"]   = idx["Relative humidity (%)"].resample(RES_RULE).std()
    # 광
    if "PAR Light (μmol·m2·s-1)" in idx.columns:
        par = idx["PAR Light (μmol·m2·s-1)"]
        par_hour_all = par.resample(RES_RULE).mean()  # 0 포함 평균
        # 듀티(광>0 비율)
        duty = par.resample(RES_RULE).apply(lambda g: float((g>0).mean()) if len(g)>0 else np.nan)
        # on-mean/sd
        def mean_on(g):
            g2 = g[g>0] if exclude_zero_for_mean else g
            return float(g2.mean()) if len(g2)>0 else np.nan
        def sd_on(g):
            g2 = g[g>0] if exclude_zero_for_mean else g
            return float(g2.std()) if len(g2)>1 else (0.0 if len(g2)==1 else np.nan)
        par_mean_on = par.resample(RES_RULE).apply(mean_on)
        par_sd_on   = par.resample(RES_RULE).apply(sd_on)
        # 시간당 적산: 평균(전체) * 3600 / 1e6
        hli = (par_hour_all * 3600.0) / 1_000_000.0
        out["par_mean_on"] = par_mean_on
        out["par_sd_on"]   = par_sd_on
        out["par_duty"]    = duty
        out["hli"]         = hli
    dfh = pd.DataFrame(index=base)
    for k,v in out.items():
        dfh[k] = v.reindex(base)
    return dfh.reset_index().rename(columns={"index":"Time"})

def hourly_by_rep(df_env):
    if not has_rep:
        return {"__ALL__": hourly_metrics(df_env)}
    rep_map = {}
    for r in sorted(df_env["Repetition"].dropna().unique()):
        sub = df_env[df_env["Repetition"]==r]
        if sub.empty: 
            continue
        rep_map[str(r)] = hourly_metrics(sub)
    return rep_map

rep_series = hourly_by_rep(df_env)

# ----- 통합(전체) 시계열 생성(반복 간 평균±SD) -----
def merge_times(rep_map, key):
    # 교집합 또는 합집합 시간축 선택
    if not rep_map:
        return pd.DataFrame(columns=["Time","mean","sd","n"])
    time_sets = [set(df["Time"]) for df in rep_map.values() if key in df.columns]
    if not time_sets:
        return pd.DataFrame(columns=["Time","mean","sd","n"])
    if use_overlap_reps:
        base_times = sorted(set.intersection(*time_sets)) if time_sets else []
    else:
        base_times = sorted(set.union(*time_sets)) if time_sets else []
    rows = []
    for ts in base_times:
        vals = []
        for df in rep_map.values():
            if key in df.columns:
                v = df.loc[df["Time"]==ts, key]
                if len(v)==1 and pd.notna(v.values[0]):
                    vals.append(float(v.values[0]))
        if len(vals)>0:
            rows.append({"Time": ts, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0, "n": len(vals)})
    return pd.DataFrame(rows)

# ----- 메트릭 가용성 파악 (KeyError 방지) -----
def available_metrics_for(df_or_map):
    keys = set()
    if isinstance(df_or_map, dict):
        for _, d in df_or_map.items():
            keys.update([c for c in d.columns if c not in ["Time"]])
    else:
        keys.update([c for c in df_or_map.columns if c not in ["Time"]])
    # 보기 좋게 순서
    order = ["temp_mean","hum_mean","par_mean_on","par_duty","hli"]
    return [k for k in order if k in keys]

pretty = {
    "temp_mean": "Temperature (℃)",
    "hum_mean": "Relative humidity (%)",
    "par_mean_on": "PAR (Light-on mean, μmol·m⁻²·s⁻¹)",
    "par_duty": "Photoperiod duty (0~1)",
    "hli": "Hourly light integral (mol·m⁻²·h⁻¹)",
}

st.markdown("### 📐 지표 선택")
if rep_choice == "전체(통합)":
    avail = available_metrics_for(rep_series)
else:
    df_sel = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
    avail = available_metrics_for(df_sel) if df_sel is not None else []

if not avail:
    st.info("사용 가능한 지표가 없습니다. 데이터의 컬럼 구성을 확인하세요.")
    st.stop()

metrics = st.multiselect("표시할 지표", options=[pretty.get(k,k) for k in avail], default=[pretty.get(k,k) for k in avail])
inv = {pretty.get(k,k): k for k in avail}
metrics_keys = [inv[m] for m in metrics if m in inv]

st.markdown("### 📊 표시 유형")
view_type = st.radio("그래프 유형", ["시계열", "막대그래프"], index=0, horizontal=True)

# 오차모드 (통합 시)
err_mode = st.radio("오차 막대", ["SD","SEM"], index=0) if rep_choice=="전체(통합)" else "SD"

def sem_from_sd(sd, n):
    try:
        return sd/np.sqrt(n) if (sd is not None and n and n>0) else np.nan
    except Exception:
        return np.nan

# ----- 시계열 -----
if view_type == "시계열":
    if rep_choice == "전체(통합)":
        for key in metrics_keys:
            agg = merge_times(rep_series, key)
            if agg.empty:
                st.info(f"{pretty.get(key,key)}: 데이터 없음")
                continue
            agg = agg.sort_values("Time")
            agg["err"] = agg["sd"] if err_mode=="SD" else agg.apply(lambda r: sem_from_sd(r["sd"], r["n"]), axis=1)
            fig = px.line(agg, x="Time", y="mean", error_y="err", markers=True, labels={"mean":pretty.get(key,key)},
                          title=f"{pretty.get(key,key)} — 반복 통합(시간당 평균±{err_mode})")
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        dfh = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
        if dfh is None or dfh.empty:
            st.info("선택한 반복에서 데이터가 없습니다.")
        else:
            for key in metrics_keys:
                if key not in dfh.columns:
                    st.info(f"{pretty.get(key,key)}: 데이터 없음")
                    continue
                fig = px.line(dfh, x="Time", y=key, markers=True, labels={key:pretty.get(key,key)},
                              title=f"{pretty.get(key,key)} — Repetition {rep_choice}")
                fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(fig, use_container_width=True)

# ----- 막대그래프 -----
else:
    # 집계 기준: 시간당 값의 평균±SD vs. 일단위 합/평균
    agg_basis = st.selectbox("집계 기준", ["시간당 값의 평균±SD", "일단위 합계/평균±SD"])
    def summarize_hours(dfh, key):
        d = dfh.dropna(subset=[key])
        if d.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
        return {"mean":float(d[key].mean()), "sd":float(d[key].std()), "n":int(d[key].count())}

    def summarize_daily(dfh, key):
        if dfh.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
        d = dfh.copy()
        d["date"] = pd.to_datetime(d["Time"]).dt.date
        if key == "hli":
            grp = d.groupby("date")["hli"].sum().rename("val")
        else:
            grp = d.groupby("date")[key].mean().rename("val")
        if grp.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
        return {"mean":float(grp.mean()), "sd":float(grp.std()), "n":int(grp.count())}

    if rep_choice == "전체(통합)":
        # 반복별 막대 vs 반복 통합 막대
        bar_mode = st.radio("막대 유형", ["반복별 막대", "반복 통합(하나의 막대)"], index=0, horizontal=True)
        for key in metrics_keys:
            rows = []
            for r, dfh in rep_series.items():
                if r == "__ALL__": continue
                if key not in dfh.columns: continue
                stat = summarize_hours(dfh, key) if agg_basis=="시간당 값의 평균±SD" else summarize_daily(dfh, key)
                rows.append({"Repetition": str(r), "mean":stat["mean"], "sd":stat["sd"], "n":stat["n"]})
            if not rows:
                st.info(f"{pretty.get(key,key)}: 요약할 데이터 없음")
                continue
            bar_df = pd.DataFrame(rows)
            if bar_mode == "반복 통합(하나의 막대)":
                pooled = {"Repetition":"ALL",
                          "mean":bar_df["mean"].mean(),
                          "sd":bar_df["mean"].std(ddof=1) if len(bar_df)>1 else 0.0,
                          "n":len(bar_df)}
                bar_df = pd.DataFrame([pooled])
            err_mode2 = st.radio(f"{pretty.get(key,key)} — 오차", ["SD","SEM"], index=0, key=f"err_{key}")
            if err_mode2=="SEM":
                bar_df["err"] = bar_df.apply(lambda r: (r["sd"]/np.sqrt(r["n"])) if r["n"]>0 else np.nan, axis=1)
            else:
                bar_df["err"] = bar_df["sd"]
            fig = px.bar(bar_df, x="Repetition", y="mean", error_y="err",
                         labels={"mean":pretty.get(key,key)}, title=f"{pretty.get(key,key)} — {agg_basis}")
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        dfh = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
        for key in metrics_keys:
            if dfh is None or key not in dfh.columns:
                st.info(f"{pretty.get(key,key)}: 데이터 없음")
                continue
            stat = summarize_hours(dfh, key) if agg_basis=="시간당 값의 평균±SD" else summarize_daily(dfh, key)
            bar_df = pd.DataFrame([{"Repetition": str(rep_choice), "mean":stat["mean"], "sd":stat["sd"], "n":stat["n"]}])
            err_mode2 = st.radio(f"{pretty.get(key,key)} — 오차", ["SD","SEM"], index=0, key=f"err_single_{key}")
            if err_mode2=="SEM":
                bar_df["err"] = bar_df.apply(lambda r: (r["sd"]/np.sqrt(r["n"])) if r["n"]>0 else np.nan, axis=1)
            else:
                bar_df["err"] = bar_df["sd"]
            fig = px.bar(bar_df, x="Repetition", y="mean", error_y="err",
                         labels={"mean":pretty.get(key,key)}, title=f"{pretty.get(key,key)} — {agg_basis}")
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------- 도움말 ----------
with st.expander("ℹ️ 용어/계산 방식"):
    st.markdown("""
- **겹치는 기간(반복 간)**: 반복마다 수집 시작/끝 시간이 다르면, 같은 시간대만 골라 공정하게 비교합니다.
- **Photoperiod duty**: 1시간 창에서 **광도>0**인 샘플 비율(0~1).
- **PAR (Light-on mean)**: 1시간 창에서 **광도>0** 값만 평균(옵션으로 0 포함 가능).
- **HLI (Hourly Light Integral)**: 평균 PAR × 3600 / 1e6 → mol·m⁻²·h⁻¹.
- **DLI (Daily Light Integral)**: 하루 동안의 HLI 합계(막대그래프에서 “일단위 합계/평균±SD”로 확인).
""")
