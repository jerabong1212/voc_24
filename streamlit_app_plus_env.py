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
# 1) 업로드 + 시트 선택/병합 + 템플릿 다운로드 + 컬럼 자동 매핑
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

# 템플릿 다운로드
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

# 엑셀 다중 시트 지원
if df is None and sheet_names is not None and file_bytes is not None:
    st.sidebar.markdown("**엑셀 시트 구성 감지됨**")
    combine_all = st.sidebar.checkbox("📑 모든 시트 합쳐서 분석", value=False)
    if combine_all:
        df = _read_excel_all(file_bytes, sheet_names)
        st.sidebar.caption("모든 시트를 세로 병합했습니다.")
    else:
        sel_sheet = st.sidebar.selectbox("📑 시트 선택", sheet_names, index=0)
        df = _read_excel_sheet(file_bytes, sel_sheet)

# ---------- 표준 컬럼 이름으로 자동 매핑 (VOC) ----------
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
# 2) VOC 애널리틱스 준비
# =========================================================
NAME_COL      = "Name"
TREAT_COL     = "Treatment"
START_COL     = "Start Date"
END_COL       = "End Date"
CHAMBER_COL   = "Chamber"
LINE_COL      = "Line"
PROGRESS_COL  = "Progress"
INTERVAL_COL  = "Interval (h)"
TEMP_COL      = "Temp (℃)"
HUMID_COL     = "Humid (%)"

REP_CANDIDATES    = ["Repetition", "rep", "Rep", "repetition", "반복", "반복수"]
SUBREP_CANDIDATES = ["Sub-repetition", "subrep", "Subrep", "Sub-rep", "sub-repetition", "소반복", "소반복수"]
REP_COL    = next((c for c in REP_CANDIDATES if (df is not None and c in df.columns)), None)
SUBREP_COL = next((c for c in SUBREP_CANDIDATES if (df is not None and c in df.columns)), None)

VOC_24_CANDIDATES = [
    "(+/-)-trans-nerolidol",
    "(E)-2-hexenal;(Z)-3-hexenal",
    "(S)-citronellol",
    "(Z)-3-hexen-1-ol",
    "(Z)-3-hexenyl acetate",
    "2-phenylethanol",
    "alpha-farnesene",
    "alpha-pinene",
    "benzaldehyde",
    "beta-caryophyllene",
    "beta-pinene",
    "DEN",
    "eucalyptol",
    "indole",
    "lemonol",
    "linalool",
    "methyl jasmonate (20180404ATFtest)",
    "methyl salicylate",
    "nicotine",
    "nitric oxide",
    "ocimene;Limonene;myrcene",
    "Pinenes",
    "toluene",
    "xylenes + ethylbenzene",
]
DISPLAY_MAP = {
    "DEN": "DMNT",
    "DMNT": "DMNT",
    "methyl jasmonate (20180404ATFtest)": "Methyl jasmonate",
    "methyl jasmonate (temporary)": "Methyl jasmonate",
}
def display_name(col):
    return DISPLAY_MAP.get(col, col)

def resolve_voc_columns(df, candidates):
    if df is None:
        return []
    resolved = []
    for col in candidates:
        if col in df.columns:
            resolved.append(col)
        elif col == "DEN" and "DMNT" in df.columns:
            resolved.append("DMNT")
    meta_cols = set([NAME_COL,TREAT_COL,START_COL,END_COL,CHAMBER_COL,LINE_COL,PROGRESS_COL,INTERVAL_COL,TEMP_COL,HUMID_COL,REP_COL,SUBREP_COL])
    numeric_candidates = [c for c in (df.columns if df is not None else []) if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not resolved and numeric_candidates:
        resolved = numeric_candidates
    return resolved

voc_columns = resolve_voc_columns(df, VOC_24_CANDIDATES) if df is not None else []

if df is not None and INTERVAL_COL in df.columns:
    df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")

# =========================================================
# 3) 사이드바 옵션 (공통)
# =========================================================
st.sidebar.header("🔧 분석 옵션")
mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝", "환경 데이터 분석"])

# 공통 필터 (VOC 쪽에서만 사용)
chambers = ["전체"] + sorted(df[CHAMBER_COL].dropna().astype(str).unique().tolist()) if (df is not None and CHAMBER_COL in df.columns) else ["전체"]
lines    = ["전체"] + sorted(df[LINE_COL].dropna().astype(str).unique().tolist()) if (df is not None and LINE_COL in df.columns) else ["전체"]
chamber_sel = st.sidebar.selectbox("🏠 Chamber", chambers, index=0, disabled=(mode=="환경 데이터 분석"))
line_sel    = st.sidebar.selectbox("🧵 Line", lines, index=0, disabled=(mode=="환경 데이터 분석"))

treatments = sorted(df[TREAT_COL].dropna().astype(str).unique().tolist()) if (df is not None and TREAT_COL in df.columns) else []
intervals_all = sorted(df[INTERVAL_COL].dropna().unique().tolist()) if (df is not None and INTERVAL_COL in df.columns) else []
reps_all = ["전체"] + sorted(df[REP_COL].dropna().astype(str).unique().tolist()) if (df is not None and REP_COL) else ["전체"]
progress_vals_all = sorted(df[PROGRESS_COL].dropna().astype(str).unique().tolist()) if (df is not None and PROGRESS_COL in df.columns) else []

rep_sel = st.sidebar.selectbox("🔁 Repetition", reps_all, index=0, disabled=(mode=="환경 데이터 분석")) if (df is not None and REP_COL) else "전체"
progress_sel = st.sidebar.multiselect("🧭 Progress(복수 선택 가능)", progress_vals_all, default=progress_vals_all, disabled=(mode=="환경 데이터 분석"))

facet_by_chamber = st.sidebar.checkbox("Chamber로 분할 보기", value=False, disabled=(mode=="환경 데이터 분석"))
facet_by_line    = st.sidebar.checkbox("Line으로 분할 보기", value=False, disabled=(mode=="환경 데이터 분석"))
err_mode = st.sidebar.radio("오차 기준", ["SD", "SEM"], index=0, disabled=(mode=="환경 데이터 분석"))

# =========================================================
# 4) VOC 모드들
# =========================================================
def apply_filters(df):
    out = df.copy()
    if CHAMBER_COL in out.columns and chamber_sel != "전체":
        out = out[out[CHAMBER_COL].astype(str) == str(chamber_sel)]
    if LINE_COL in out.columns and line_sel != "전체":
        out = out[out[LINE_COL].astype(str) == str(line_sel)]
    if PROGRESS_COL in out.columns and progress_sel:
        out = out[out[PROGRESS_COL].astype(str).isin(progress_sel)]
    if REP_COL and rep_sel != "전체":
        out = out[out[REP_COL].astype(str) == str(rep_sel)]
    return out

def add_facets(kwargs, data_frame):
    if facet_by_chamber and CHAMBER_COL in data_frame.columns:
        kwargs["facet_col"] = CHAMBER_COL
    if facet_by_line and LINE_COL in data_frame.columns:
        if "facet_col" in kwargs:
            kwargs["facet_row"] = LINE_COL
        else:
            kwargs["facet_col"] = LINE_COL
    return kwargs

def sem_from_sd(sd, n):
    try:
        return sd / np.sqrt(n) if (sd is not None and n and n > 0) else np.nan
    except Exception:
        return np.nan

def attach_error_col(df_stats, err_mode):
    df_stats = df_stats.copy()
    if "sd" not in df_stats.columns:
        df_stats["sd"] = np.nan
    if "n" not in df_stats.columns:
        df_stats["n"] = np.nan
    if err_mode == "SEM":
        df_stats["err"] = df_stats.apply(lambda r: sem_from_sd(r.get("sd", np.nan), r.get("n", np.nan)), axis=1)
    else:
        df_stats["err"] = df_stats.get("sd", np.nan)
    return df_stats

if mode in ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"]:
    if df is None or not len(voc_columns):
        st.info("VOC 데이터 업로드 후 이용해주세요.")
    else:
        # 공통: VOC 선택
        if mode != "전체 VOC 스크리닝":
            selected_voc = st.sidebar.selectbox("📌 VOC 물질 선택", [display_name(c) for c in voc_columns])
            inv_map = {display_name(c): c for c in voc_columns}
            selected_voc_internal = inv_map[selected_voc]
        else:
            selected_voc, selected_voc_internal = None, None

        filtered_df = apply_filters(df)

        if mode == "처리별 VOC 비교":
            chart_type = st.sidebar.radio("차트 유형", ["막대그래프", "박스플롯"], index=0)
            selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all)

            if selected_interval == "전체":
                data_use = filtered_df.copy()
                title_suffix = "모든 시간"
            else:
                data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
                title_suffix = f"Interval: {selected_interval}h"

            y_label = f"{selected_voc} 농도 (ppb)"
            color_kw = {"color": PROGRESS_COL} if (PROGRESS_COL in data_use.columns and data_use[PROGRESS_COL].notna().any()) else {}

            if chart_type == "막대그래프":
                group_keys = [TREAT_COL]
                if PROGRESS_COL in data_use.columns:
                    group_keys.append(PROGRESS_COL)
                if CHAMBER_COL in data_use.columns and facet_by_chamber:
                    group_keys.append(CHAMBER_COL)
                if LINE_COL in data_use.columns and facet_by_line:
                    group_keys.append(LINE_COL)

                if SUBREP_COL and SUBREP_COL in data_use.columns:
                    per_subrep = (
                        data_use.groupby(group_keys + ([REP_COL] if REP_COL else []) + [SUBREP_COL])[selected_voc_internal]
                        .mean()
                        .reset_index()
                    )
                else:
                    per_subrep = data_use.copy()

                if REP_COL and REP_COL in data_use.columns:
                    per_rep = per_subrep.groupby(group_keys + [REP_COL])[selected_voc_internal].mean().reset_index()
                    grouped = per_rep.groupby(group_keys)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index()
                else:
                    grouped = per_subrep.groupby(group_keys)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index()

                grouped = attach_error_col(grouped, err_mode)

                fig_kwargs = dict(
                    x=TREAT_COL, y="mean",
                    labels={"mean": y_label, TREAT_COL: "처리"},
                    title=f"{selected_voc} - 처리별 평균 비교 ({title_suffix})",
                    **color_kw
                )
                fig_kwargs = add_facets(fig_kwargs, grouped)
                fig = px.bar(grouped, **fig_kwargs, error_y="err", barmode="group")
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

            else:  # 박스플롯
                use_rep_agg_box = st.sidebar.checkbox("박스플롯도 반복 평균 기반으로 요약", value=False) if REP_COL else False
                if use_rep_agg_box and REP_COL:
                    group_keys_box = [TREAT_COL]
                    if PROGRESS_COL in data_use.columns:
                        group_keys_box.append(PROGRESS_COL)
                    if CHAMBER_COL in data_use.columns and facet_by_chamber:
                        group_keys_box.append(CHAMBER_COL)
                    if LINE_COL in data_use.columns and facet_by_line:
                        group_keys_box.append(LINE_COL)
                    if SUBREP_COL and SUBREP_COL in data_use.columns:
                        per_subrep_box = data_use.groupby(group_keys_box + [REP_COL, SUBREP_COL])[selected_voc_internal].mean().reset_index()
                    else:
                        per_subrep_box = data_use.groupby(group_keys_box + [REP_COL])[selected_voc_internal].mean().reset_index()
                    per_rep_box = per_subrep_box.groupby(group_keys_box + [REP_COL])[selected_voc_internal].mean().reset_index()
                    data_for_box = per_rep_box
                    y_for_box = selected_voc_internal
                else:
                    data_for_box = data_use
                    y_for_box = selected_voc_internal
                fig_kwargs = dict(
                    x=TREAT_COL, y=y_for_box,
                    labels={y_for_box: y_label, TREAT_COL: "처리"},
                    title=f"{selected_voc} - 처리별 분포 (박스플롯) ({title_suffix})",
                    points="outliers",
                    **color_kw,
                )
                fig_kwargs = add_facets(fig_kwargs, data_for_box)
                fig = px.box(data_for_box, **fig_kwargs)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

        elif mode == "시간별 VOC 변화":
            selected_treatment = st.sidebar.selectbox("🧪 처리구 선택", ["전체"] + treatments)

            if selected_treatment == "전체":
                data_use = filtered_df.copy()
                title_prefix = "모든 처리"
            else:
                data_use = filtered_df[filtered_df[TREAT_COL].astype(str) == str(selected_treatment)].copy()
                title_prefix = f"{selected_treatment} 처리"

            tick_vals = sorted(df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in df.columns else []

            group_keys_display = [INTERVAL_COL]
            if selected_treatment == "전체":
                group_keys_display.append(TREAT_COL)
            if PROGRESS_COL in data_use.columns:
                group_keys_display.append(PROGRESS_COL)
            if CHAMBER_COL in data_use.columns and facet_by_chamber:
                group_keys_display.append(CHAMBER_COL)
            if LINE_COL in data_use.columns and facet_by_line:
                group_keys_display.append(LINE_COL)

            n_rep = data_use[REP_COL].nunique() if REP_COL and REP_COL in data_use.columns else 0

            if SUBREP_COL and SUBREP_COL in data_use.columns:
                per_subrep_ts = (
                    data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[selected_voc_internal]
                    .mean()
                    .reset_index()
                )
            else:
                per_subrep_ts = data_use.copy()

            if n_rep and n_rep >= 2:
                per_rep_ts = per_subrep_ts.groupby(group_keys_display + [REP_COL])[selected_voc_internal].mean().reset_index()
                final = per_rep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis = "반복 SD/SEM"
            else:
                final = per_subrep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis = "소반복 SD/SEM"

            final = attach_error_col(final, err_mode)

            fig_kwargs = dict(
                x=INTERVAL_COL,
                y="mean",
                error_y="err",
                markers=True,
                labels={INTERVAL_COL: "Interval (h)", "mean": f"{selected_voc} 평균농도 (ppb)"},
                title=f"{title_prefix} - {selected_voc} 변화 추이 (평균±{err_mode}, 기준: {err_basis})",
            )
            if selected_treatment == "전체":
                fig_kwargs["color"] = TREAT_COL
            elif PROGRESS_COL in data_use.columns:
                fig_kwargs["color"] = PROGRESS_COL
            fig_kwargs = add_facets(fig_kwargs, final)
            fig_voc = px.line(final, **fig_kwargs)
            if tick_vals:
                fig_voc.update_xaxes(tickmode='array', tickvals=tick_vals)
            fig_voc.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_voc, use_container_width=True)

            # 환경변수 부가 시리즈
            for env_col in [TEMP_COL, HUMID_COL]:
                if env_col not in data_use.columns:
                    continue
                if SUBREP_COL and SUBREP_COL in data_use.columns:
                    per_subrep_env = (
                        data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[env_col]
                        .mean()
                        .reset_index()
                    )
                else:
                    per_subrep_env = data_use.copy()

                if n_rep and n_rep >= 2:
                    per_rep_env = per_subrep_env.groupby(group_keys_display + ([REP_COL] if REP_COL else []))[env_col].mean().reset_index()
                    ts_env = per_rep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                    err_basis_env = "반복 SD/SEM"
                else:
                    ts_env = per_subrep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                    err_basis_env = "소반복 SD/SEM"

                ts_env = attach_error_col(ts_env, err_mode)

                ylab = "온도 (°C)" if env_col == TEMP_COL else "상대습도 (%)" if env_col == HUMID_COL else env_col
                fig_kwargs_env = dict(
                    x=INTERVAL_COL,
                    y="mean",
                    error_y="err",
                    markers=True,
                    labels={INTERVAL_COL: "Interval (h)", "mean": ylab},
                    title=f"{title_prefix} - {env_col} 변화 추이 (평균±{err_mode}, 기준: {err_basis_env})",
                )
                if PROGRESS_COL in data_use.columns:
                    fig_kwargs_env["color"] = PROGRESS_COL
                fig_kwargs_env = add_facets(fig_kwargs_env, ts_env)
                fig_env = px.line(ts_env, **fig_kwargs_env)
                if tick_vals:
                    fig_env.update_xaxes(tickmode='array', tickvals=tick_vals)
                fig_env.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_env, use_container_width=True)

        else:  # 전체 VOC 스크리닝
            selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all, key="scr_interval")
            alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f", key="scr_alpha")
            include_rep_block_scr = st.sidebar.checkbox("반복을 블록요인으로 포함(스크리닝)", value=bool(REP_COL)) if REP_COL else False

            if selected_interval == "전체":
                data_use = filtered_df.copy()
                title_suffix = "모든 시간"
            else:
                data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
                title_suffix = f"Interval: {selected_interval}h"

            sm, smf, MultiComparison, sp, HAS_SCPH, scipy_stats = _lazy_import_stats()

            results = []
            for voc in voc_columns:
                base_cols = [TREAT_COL, voc]
                if REP_COL: base_cols.append(REP_COL)
                if SUBREP_COL: base_cols.append(SUBREP_COL)
                sub = data_use[base_cols].dropna().copy()
                if sub.empty or sub[TREAT_COL].nunique() < 2:
                    continue
                if not all(sub.groupby(TREAT_COL)[voc].count() >= 2):
                    continue
                a_df = sub.rename(columns={voc: "y", TREAT_COL: "treat"})
                try:
                    if include_rep_block_scr and REP_COL and REP_COL in a_df.columns:
                        a_df["rep"] = a_df[REP_COL].astype(str)
                        model = smf.ols("y ~ C(treat) + C(rep)", data=a_df).fit()
                    else:
                        model = smf.ols("y ~ C(treat)", data=a_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    pval = float(anova_table.loc["C(treat)", "PR(>F)"])
                except Exception:
                    pval = np.nan
                results.append({"VOC": display_name(voc), "p_value": pval})

            if not results:
                st.info("조건에 맞는 데이터가 없어 스크리닝 결과가 비어있습니다.")
            else:
                res_df = pd.DataFrame(results).sort_values("p_value", na_position="last")
                st.markdown(f"**Interval: {title_suffix}**, α={alpha}")
                st.dataframe(res_df, use_container_width=True)
                st.download_button(
                    "⬇️ 스크리닝 결과 CSV",
                    data=res_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="voc_screening_results.csv",
                    mime="text/csv",
                )

# =========================================================
# 5) 환경 데이터 분석 (NEW)
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
            # 1st sheet by default
            df = xlf.parse(xlf.sheet_names[0])
        return df

    df_t = read_env(tomato_file)
    df_l = read_env(larva_file)

    st.caption("업로드 전/후 컬럼 자동 매핑을 적용해, 다양한 헤더 표기를 흡수합니다.")

    # 컬럼 자동 매핑 (환경 버전)
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

    df_t = env_standardize_columns(df_t)
    df_l = env_standardize_columns(df_l)

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
        # Numeric coercion
        for col in ["Temperature (℃)", "Relative humidity (%)", "PAR Light (μmol·m2·s-1)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    df_t = coerce_ts(df_t)
    df_l = coerce_ts(df_l)

    if df_t is None or df_l is None or df_t.empty or df_l.empty:
        st.info("좌측에서 토마토/애벌레 환경 파일을 모두 업로드하세요.")
    else:
        # 기간 교집합으로 자르기 옵션
        min_ts = max(df_t["__ts__"].min(), df_l["__ts__"].min())
        max_ts = min(df_t["__ts__"].max(), df_l["__ts__"].max())
        st.caption(f"시간 교집합: **{min_ts} ~ {max_ts}**")
        use_overlap = st.checkbox("두 데이터의 **겹치는 기간만** 사용", value=True)
        if use_overlap:
            df_t = df_t[(df_t["__ts__"]>=min_ts) & (df_t["__ts__"]<=max_ts)]
            df_l = df_l[(df_l["__ts__"]>=min_ts) & (df_l["__ts__"]<=max_ts)]

        # 리샘플링 간격 선택
        res_map = {"원자료": None, "5분": "5min", "10분": "10min", "30분": "30min", "60분": "60min"}
        res_key = st.selectbox("표시 간격(리샘플링)", list(res_map.keys()), index=2)
        rule = res_map[res_key]

        def resample_df(df, rule):
            if rule is None:
                return df.copy()
            return (
                df.set_index("__ts__")
                  .resample(rule)
                  .mean(numeric_only=True)
                  .reset_index()
                  .rename(columns={"__ts__":"__ts__"})
            )

        df_t_r = resample_df(df_t, rule)
        df_l_r = resample_df(df_l, rule)

        # 개요 테이블
        def _interval_report(df):
            dts = df["__ts__"].sort_values()
            if len(dts) < 2:
                return {"records": len(dts), "median_min": np.nan, "p90_min": np.nan, "irregular(>%2x med)": np.nan}
            deltas = (dts.diff().dropna().dt.total_seconds() / 60.0).values
            med = float(np.median(deltas))
            p90 = float(np.percentile(deltas, 90))
            irregular = float((deltas > (2*med)).mean()) if med>0 else np.nan
            return {"records": int(len(dts)), "median_min": med, "p90_min": p90, "irregular(>%2x med)": irregular}

        rep_t = _interval_report(df_t)
        rep_l = _interval_report(df_l)

        overview = pd.DataFrame([
            {"Dataset":"Tomato", "Start": df_t["__ts__"].min(), "End": df_t["__ts__"].max(), **rep_t},
            {"Dataset":"Larva",  "Start": df_l["__ts__"].min(), "End": df_l["__ts__"].max(), **rep_l},
        ])

        st.markdown("#### ⏱️ 기록 개요")
        st.dataframe(overview, use_container_width=True)

        metrics = ["Temperature (℃)", "Relative humidity (%)", "PAR Light (μmol·m2·s-1)"]
        metrics = [m for m in metrics if (m in df_t_r.columns and m in df_l_r.columns)]
        sel_metrics = st.multiselect("분석할 지표 선택", metrics, default=metrics)

        # 시계열 겹쳐보기
        for m in sel_metrics:
            st.markdown(f"#### 📈 {m} 추이 (겹쳐보기)")
            a = df_t_r[["__ts__", m]].assign(Dataset="Tomato").rename(columns={"__ts__":"Time","{}".format(m):"Value"})
            b = df_l_r[["__ts__", m]].assign(Dataset="Larva").rename(columns={"__ts__":"Time","{}".format(m):"Value"})
            dd = pd.concat([a,b], ignore_index=True).dropna(subset=["Value"])
            fig = px.line(dd, x="Time", y="Value", color="Dataset")
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # 시간대별(시각) 프로파일
            st.caption("시간대별 평균 프로파일")
            dd["hour"] = pd.to_datetime(dd["Time"]).dt.hour
            prof = dd.groupby(["Dataset","hour"])["Value"].agg(["mean","std","count"]).reset_index()
            fig2 = px.line(prof, x="hour", y="mean", color="Dataset", error_y="std", markers=True,
                           labels={"hour":"Hour of day","mean":f"{m} (mean±SD)"})
            fig2.update_layout(margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig2, use_container_width=True)

            # 분포 비교 (박스)
            st.caption("분포 비교 (박스플롯)")
            fig3 = px.box(dd, x="Dataset", y="Value", points="outliers")
            fig3.update_layout(margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig3, use_container_width=True)

            # 간단 통계 + 검정
            sm, smf, MultiComparison, sp, HAS_SCPH, scipy_stats = _lazy_import_stats()
            g = dd.groupby("Dataset")["Value"]
            stat_tbl = g.agg(mean="mean", sd="std", n="count", min="min", max="max").reset_index()
            stat_tbl["cv(%)"] = (stat_tbl["sd"] / stat_tbl["mean"] * 100.0).replace([np.inf,-np.inf], np.nan)
            st.dataframe(stat_tbl, use_container_width=True)

            if scipy_stats is not None and stat_tbl["n"].min() >= 3:
                # 등분산성 (Levene)
                t_vals = dd.loc[dd["Dataset"]=="Tomato","Value"].dropna().values
                l_vals = dd.loc[dd["Dataset"]=="Larva","Value"].dropna().values
                try:
                    lev_stat, lev_p = scipy_stats.levene(t_vals, l_vals, center="median")
                except Exception:
                    lev_stat, lev_p = np.nan, np.nan
                # 평균 차이 (독립 t, Welch)
                try:
                    t_stat, t_p_eq = scipy_stats.ttest_ind(t_vals, l_vals, equal_var=True)
                    w_stat, w_p = scipy_stats.ttest_ind(t_vals, l_vals, equal_var=False)
                except Exception:
                    t_stat=t_p_eq=w_stat=w_p=np.nan
                # 분포 차이 (Mann–Whitney)
                try:
                    u_stat, u_p = scipy_stats.mannwhitneyu(t_vals, l_vals, alternative="two-sided")
                except Exception:
                    u_stat=u_p=np.nan

                res = pd.DataFrame([
                    {"Test":"Levene (등분산)", "stat":lev_stat, "p":lev_p},
                    {"Test":"t-test (equal var)", "stat":t_stat, "p":t_p_eq},
                    {"Test":"Welch t-test", "stat":w_stat, "p":w_p},
                    {"Test":"Mann–Whitney U", "stat":u_stat, "p":u_p},
                ])
                st.markdown("**간이 가설검정 결과**")
                st.dataframe(res, use_container_width=True)
            else:
                st.caption("표본이 부족하거나 SciPy 미설치로 가설검정을 생략했습니다.")

# ---------- 원본 VOC 데이터 확인 ----------
with st.expander("🔍 (선택) VOC 원본 데이터 보기"):
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("VOC 데이터가 업로드되지 않았습니다.")
