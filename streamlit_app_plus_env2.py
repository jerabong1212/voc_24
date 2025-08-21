import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Lazy stats import (only when needed) ----------
def _lazy_import_stats():
    import importlib
    sm = importlib.import_module("statsmodels.api")
    smf = importlib.import_module("statsmodels.formula.api")
    try:
        mc_mod = importlib.import_module("statsmodels.stats.multicomp")
        MultiComparison = getattr(mc_mod, "MultiComparison")
    except Exception:
        MultiComparison = None
    try:
        sp = importlib.import_module("scikit_posthocs")
        HAS_SCPH = True
    except Exception:
        sp = None
        HAS_SCPH = False
    try:
        from scipy import stats as scipy_stats
    except Exception:
        scipy_stats = None
    return sm, smf, MultiComparison, sp, HAS_SCPH, scipy_stats

st.set_page_config(page_title="VOC & 환경 데이터 시각화", layout="wide")
st.title("🌿 식물 VOC & 환경 데이터 시각화")

# =========================================================
# 1) VOC 데이터 업로드(기존 기능 유지, 요약)
# =========================================================
st.sidebar.header("📁 VOC 데이터 불러오기")

@st.cache_data(show_spinner=False)
def _read_any(file_bytes, name):
    ext = str(name).lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes)), None
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    return None, xlf.sheet_names

@st.cache_data(show_spinner=False)
def _read_excel_sheet(file_bytes, sheet_name):
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    return xlf.parse(sheet_name)

@st.cache_data(show_spinner=False)
def _read_excel_all(file_bytes, sheet_names):
    frames = []
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    for s in sheet_names:
        try:
            df = xlf.parse(s)
            df["__Sheet__"] = s
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _template_bytes():
    template_cols = [
        "Name","Treatment","Start Date","End Date","Chamber","Line",
        "Progress","Interval (h)","Temp (℃)","Humid (%)",
        "Repetition","Sub-repetition",
        "linalool","DMNT","beta-caryophyllene"
    ]
    buf = io.BytesIO()
    pd.DataFrame(columns=template_cols).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()

st.sidebar.download_button(
    "⬇️ VOC 템플릿 엑셀",
    data=_template_bytes(),
    file_name="VOC_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded = st.sidebar.file_uploader("VOC 데이터 업로드 (xlsx/xls/csv)", type=["xlsx","xls","csv"])
use_demo = st.sidebar.button("🧪 VOC 데모 데이터")

df, file_name = None, None
sheet_names = None
file_bytes = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    tmp, sheet_names = _read_any(file_bytes, uploaded.name)
    if tmp is not None:  # CSV
        df = tmp
    file_name = uploaded.name

if use_demo and df is None and uploaded is None:
    demo = {
        "Name": ["A"]*18,
        "Treatment": ["control"]*6 + ["herbivory"]*6 + ["threat"]*6,
        "Start Date": pd.to_datetime(["2025-08-01"]*18),
        "End Date": pd.to_datetime(["2025-08-02"]*18),
        "Chamber": ["C1"]*9 + ["C2"]*9,
        "Line": ["L1"]*18,
        "Progress": (["before"]*3 + ["after"]*3)*3,
        "Interval (h)": [-1,0,1, -1,0,1]*3,
        "Temp (℃)": np.random.normal(24, 0.3, 18),
        "Humid (%)": np.random.normal(55, 1.2, 18),
        "Repetition": [1]*18,
        "Sub-repetition": [1,2,3]*6,
        "linalool": np.r_[np.random.normal(5,0.3,6), np.random.normal(7,0.3,6), np.random.normal(9,0.3,6)],
    }
    df = pd.DataFrame(demo)
    file_name = "DEMO"

if df is None and sheet_names is not None and file_bytes is not None:
    st.sidebar.markdown("**엑셀 시트 구성 감지됨**")
    combine_all = st.sidebar.checkbox("📑 모든 시트 합쳐서 분석", value=False)
    if combine_all:
        df = _read_excel_all(file_bytes, sheet_names)
        st.sidebar.caption("모든 시트를 세로 병합했습니다.")
    else:
        sel_sheet = st.sidebar.selectbox("📑 시트 선택", sheet_names, index=0)
        df = _read_excel_sheet(file_bytes, sel_sheet)

# ---------- VOC 표준 컬럼 자동 매핑 ----------
CANON = {
    "Name": ["name","sample","시료","샘플","이름"],
    "Treatment": ["treatment","처리","처리구","group","그룹"],
    "Start Date": ["start date","start","시작","시작일"],
    "End Date": ["end date","end","종료","종료일"],
    "Chamber": ["chamber","룸","방","챔버"],
    "Line": ["line","라인","계통","품종"],
    "Progress": ["progress","상태","단계","before/after","stage"],
    "Interval (h)": ["interval (h)","interval","time (h)","time","시간","시간(h)","interval(h)","시각","측정간격"],
    "Temp (℃)": ["temp (℃)","temp","temperature","온도"],
    "Humid (%)": ["humid (%)","humidity","습도"],
    "Repetition": ["repetition","replicate","rep","반복","반복수"],
    "Sub-repetition": ["sub-repetition","technical replicate","subrep","sub_rep","소반복","소반복수"],
}
def _normalize(s):
    return str(s).strip().lower().replace("_"," ").replace("-"," ")

def standardize_columns(df):
    if df is None:
        return df
    col_map = {}
    for c in df.columns:
        lc = _normalize(c)
        mapped = None
        for canon, aliases in CANON.items():
            if lc == _normalize(canon) or lc in [_normalize(a) for a in aliases]:
                mapped = canon
                break
        if mapped:
            col_map[c] = mapped
    if col_map:
        df = df.rename(columns=col_map)
    return df

if df is not None:
    df = standardize_columns(df)

# =========================================================
# 2) 사이드바 공통 모드 선택
# =========================================================
mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝", "환경 데이터 분석"])

# =========================================================
# 3) 환경 데이터 분석 (강화)
# =========================================================
if mode == "환경 데이터 분석":
    st.subheader("🌱 토마토/애벌레 생장상 환경 데이터 분석")

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

    st.caption("업로드 후 자동으로 Date/Time/Timestamp, Temperature (℃), Relative humidity (%), PAR Light(광도) 컬럼을 감지/표준화합니다.")

    # 컬럼 자동 매핑 (환경)
    ENV_CANON = {
        "Date": ["date","날짜"],
        "Time": ["time","시간"],
        "Timestamp": ["timestamp","datatime","datetime","일시","측정시각"],
        "Temperature (℃)": ["temperature (℃)","temperature","temp","온도"],
        "Relative humidity (%)": ["relative humidity (%)","humidity","humid","rh","습도"],
        "PAR Light (μmol·m2·s-1)": ["par light (μmol·m2·s-1)","par","ppfd","광도","light","light (μmol m-2 s-1)","light (μmol·m2·s-1)","light (μmol m−2 s−1)"],
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

    if df_t is None or df_l is None or df_t.empty or df_l.empty:
        st.info("좌측에서 토마토/애벌레 환경 파일을 모두 업로드하세요.")
        st.stop()

    # ---------- 겹치는 기간만 사용 옵션 ----------
    min_ts = max(df_t["__ts__"].min(), df_l["__ts__"].min())
    max_ts = min(df_t["__ts__"].max(), df_l["__ts__"].max())
    st.caption(f"⏱️ 두 데이터가 공통으로 존재하는 시간 범위(겹치는 기간): **{min_ts} ~ {max_ts}**")
    use_overlap = st.checkbox("두 데이터의 '겹치는 기간'만 사용", value=True, help="두 데이터의 시간 구간이 다를 때 공평 비교를 위해 교집합 시간만 사용합니다.")
    if use_overlap:
        df_t = df_t[(df_t["__ts__"]>=min_ts) & (df_t["__ts__"]<=max_ts)]
        df_l = df_l[(df_l["__ts__"]>=min_ts) & (df_l["__ts__"]<=max_ts)]

    # ---------- 시간 리샘플링(1시간 고정) & 광 분석 파이프라인 ----------
    st.markdown("### 🔆 광 분석 설정")
    exclude_zero_for_mean = st.checkbox("평균/표준편차 계산 시 0(불꺼짐) 제외", value=True)
    # 1시간 리샘플링 고정
    RES_RULE = "60min"

    def hourly_env(df):
        idx = df.set_index("__ts__").sort_index()
        out = {}

        # 기본 시간축
        base = idx.resample(RES_RULE).asfreq().index

        # 온도/습도 시간 평균
        if "Temperature (℃)" in idx.columns:
            out["temp_mean"] = idx["Temperature (℃)"].resample(RES_RULE).mean()
            out["temp_sd"]   = idx["Temperature (℃)"].resample(RES_RULE).std()
        if "Relative humidity (%)" in idx.columns:
            out["hum_mean"] = idx["Relative humidity (%)"].resample(RES_RULE).mean()
            out["hum_sd"]   = idx["Relative humidity (%)"].resample(RES_RULE).std()

        # PAR: 시계열, on/off, 평균(on만), 적산광도(시간당)
        if "PAR Light (μmol·m2·s-1)" in idx.columns:
            par = idx["PAR Light (μmol·m2·s-1)"]
            par_hour_all = par.resample(RES_RULE).mean()  # 시간 평균(0 포함)
            # on 비율(듀티사이클): 양수 비율
            duty = par.resample(RES_RULE).apply(lambda g: float((g>0).mean()) if len(g)>0 else np.nan)
            # 평균/표준편차 (불켜진 값만)
            def mean_on(g):
                g2 = g[g>0]
                return float(g2.mean()) if len(g2)>0 else np.nan
            def sd_on(g):
                g2 = g[g>0]
                return float(g2.std()) if len(g2)>1 else (0.0 if len(g2)==1 else np.nan)
            par_mean_on = par.resample(RES_RULE).apply(mean_on)
            par_sd_on   = par.resample(RES_RULE).apply(sd_on)
            # 적산광도(시간당): mean(all) * 3600 / 1e6  → mol m^-2 h^-1
            hli = (par_hour_all * 3600.0) / 1_000_000.0

            out["par_mean_on"] = par_mean_on
            out["par_sd_on"]   = par_sd_on
            out["par_duty"]    = duty
            out["hli"]         = hli

        dfh = pd.DataFrame(index=base)
        for k,v in out.items():
            dfh[k] = v.reindex(base)

        dfh = dfh.reset_index().rename(columns={"index":"Time"})
        return dfh

    h_t = hourly_env(df_t)
    h_l = hourly_env(df_l)

    # --------- 시각화 유형 선택: 시계열 vs 막대그래프 ---------
    st.markdown("### 📊 표시 유형")
    view_type = st.radio("그래프 유형", ["시계열", "막대그래프"], index=0, horizontal=True)

    # --------- 메트릭 선택 ---------
    st.markdown("### 📐 지표 선택")
    metric_groups = {
        "온도": ["temp_mean"],
        "습도": ["hum_mean"],
        "광(불켜진 시간 평균)": ["par_mean_on"],
        "광주기(듀티사이클)": ["par_duty"],
        "적산광도(시간당, mol m⁻² h⁻¹)": ["hli"],
    }
    pretty = {
        "temp_mean": "Temperature (℃)",
        "hum_mean": "Relative humidity (%)",
        "par_mean_on": "PAR (Light-on mean, μmol·m⁻²·s⁻¹)",
        "par_duty": "Photoperiod duty (0~1)",
        "hli": "Hourly light integral (mol·m⁻²·h⁻¹)",
    }
    # exclude_zero_for_mean 반영: 표시 설명 업데이트
    if exclude_zero_for_mean and "par_mean_on" in pretty:
        pretty["par_mean_on"] += " — 0 제외"

    choices = []
    for k, v in metric_groups.items():
        if st.checkbox(k, value=True):
            choices += v

    # --------- 시계열 표시 ---------
    if view_type == "시계열":
        for key in choices:
            label = pretty.get(key, key)
            a = h_t[["Time", key]].copy()
            b = h_l[["Time", key]].copy()
            a["Dataset"] = "Tomato"
            b["Dataset"] = "Larva"
            dfp = pd.concat([a,b], ignore_index=True)
            # 광 평균 처리: 0 제외 옵션이 이미 par_mean_on에 반영됨
            if key == "par_mean_on" and not exclude_zero_for_mean:
                # 원한다면 0 포함 평균으로 대체할 수도 있지만, 여기서는 on 전용만 제공
                pass
            fig = px.line(dfp, x="Time", y=key, color="Dataset", markers=True, labels={key: label})
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), title=label)
            st.plotly_chart(fig, use_container_width=True)

    # --------- 막대그래프(요약 통계) ---------
    else:
        # 집계 기준 선택: 시간당 값의 평균±SD vs. 일단위(적산) 합계±SD
        agg_basis = st.selectbox("집계 기준", ["시간당 값의 평균±SD", "일단위 합계/평균±SD"])
        def summarize(dfh, key):
            d = dfh.dropna(subset=[key]).copy()
            if d.empty:
                return {"mean": np.nan, "sd": np.nan, "n": 0}
            return {"mean": float(d[key].mean()), "sd": float(d[key].std()), "n": int(d[key].count())}

        for key in choices:
            label = pretty.get(key, key)
            if agg_basis == "시간당 값의 평균±SD":
                sum_t = summarize(h_t, key)
                sum_l = summarize(h_l, key)
            else:
                # 일단위로 집계: hli는 합계(하루 적산), 나머지는 일 평균
                t = h_t.copy()
                l = h_l.copy()
                t["date"] = pd.to_datetime(t["Time"]).dt.date
                l["date"] = pd.to_datetime(l["Time"]).dt.date
                if key == "hli":
                    t_day = t.groupby("date")["hli"].sum().rename("val")
                    l_day = l.groupby("date")["hli"].sum().rename("val")
                else:
                    t_day = t.groupby("date")[key].mean().rename("val")
                    l_day = l.groupby("date")[key].mean().rename("val")
                sum_t = {"mean": float(t_day.mean()), "sd": float(t_day.std()), "n": int(t_day.count())}
                sum_l = {"mean": float(l_day.mean()), "sd": float(l_day.std()), "n": int(l_day.count())}

            bar_df = pd.DataFrame([
                {"Dataset":"Tomato", "mean":sum_t["mean"], "sd":sum_t["sd"], "n":sum_t["n"]},
                {"Dataset":"Larva",  "mean":sum_l["mean"], "sd":sum_l["sd"], "n":sum_l["n"]},
            ])
            # SEM 옵션
            err_mode = st.radio(f"{label} — 오차 막대", ["SD","SEM"], index=0, key=f"err_{key}")
            if err_mode == "SEM":
                bar_df["err"] = bar_df.apply(lambda r: (r["sd"]/np.sqrt(r["n"])) if r["n"]>0 else np.nan, axis=1)
            else:
                bar_df["err"] = bar_df["sd"]

            fig = px.bar(bar_df, x="Dataset", y="mean", error_y="err", labels={"mean":label}, title=f"{label} — {agg_basis}")
            fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # --------- 광주기/적산 관련 도움말 ---------
    with st.expander("ℹ️ 광주기/광도/적산광도 계산 방식"):
        st.markdown("""
- **광주기(듀티사이클)**: 1시간 창에서 **광도>0**인 샘플이 차지하는 비율(0~1).  
- **광도(불켜진 시간 평균)**: 1시간 창에서 **광도>0**인 값들의 평균(0은 제외).  
- **적산광도(시간당, HLI)**: 1시간 동안의 평균 PAR × 3600 / 1e6 → mol·m⁻²·h⁻¹.  
  하루 적산(=DLI)은 시간당 적산을 일 단위로 합해 계산.
""")

# =========================================================
# (기존 VOC 모드들은 간단 메시지로 가려둠 — 원본 앱에선 유지됨)
# =========================================================
if mode in ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"]:
    st.info("이 데모 파일에서는 환경 모드에 집중했어요. 기존 VOC 모드는 이전 버전에서 동작하던 대로 유지 가능해요. 원본 통합본에 그대로 병합해 줄게요.")
