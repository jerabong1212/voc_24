import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# Common helpers
# =========================================================
st.set_page_config(page_title="VOC & 환경 데이터 시각화", layout="wide")
st.title("🌿 식물 VOC & 환경 데이터 시각화")

def sem_from_sd(sd, n):
    try:
        return sd / np.sqrt(n) if (sd is not None and n and n > 0) else np.nan
    except Exception:
        return np.nan

def safe_line(df, x, y, **kwargs):
    """Robust px.line wrapper: coerce, dropna, guard empty/invalid dtypes."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info(f"그래프 대상 데이터가 없습니다: {y}")
        return
    d = df[[x, y]].copy()
    # coerce
    d[x] = pd.to_datetime(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if d.empty:
        st.info(f"그래프 대상 데이터가 없습니다(결측/비수치): {y}")
        return
    fig = px.line(d, x=x, y=y, markers=True, **kwargs)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

def safe_bar(df, x, y, **kwargs):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("막대그래프 대상 데이터가 없습니다.")
        return
    dd = df.copy()
    dd[y] = pd.to_numeric(dd[y], errors="coerce")
    dd = dd.dropna(subset=[y])
    if dd.empty:
        st.info("막대그래프 대상 데이터가 없습니다(결측/비수치).")
        return
    fig = px.bar(dd, x=x, y=y, **kwargs)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 1) VOC 데이터: 업로드/자동매핑/모드 (기존 기능 복원)
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

uploaded_voc = st.sidebar.file_uploader("VOC 데이터 업로드 (xlsx/xls/csv)", type=["xlsx","xls","csv"], key="voc_file")
use_demo_voc = st.sidebar.button("🧪 VOC 데모 데이터")

voc_df, voc_file_name = None, None
voc_sheet_names = None
voc_file_bytes = None

if uploaded_voc is not None:
    voc_file_bytes = uploaded_voc.getvalue()
    tmp, voc_sheet_names = _read_any(voc_file_bytes, uploaded_voc.name)
    if tmp is not None:  # CSV
        voc_df = tmp
    voc_file_name = uploaded_voc.name

if use_demo_voc and voc_df is None and uploaded_voc is None:
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
    voc_df = pd.DataFrame(demo)
    voc_file_name = "DEMO"

# 엑셀 다중 시트 지원
if voc_df is None and voc_sheet_names is not None and voc_file_bytes is not None:
    st.sidebar.markdown("**엑셀 시트 구성 감지됨 (VOC)**")
    combine_all = st.sidebar.checkbox("📑 모든 시트 합쳐서 분석", value=False, key="voc_merge_all")
    if combine_all:
        voc_df = _read_excel_all(voc_file_bytes, voc_sheet_names)
        st.sidebar.caption("모든 시트를 세로 병합했습니다.")
    else:
        sel_sheet = st.sidebar.selectbox("📑 시트 선택", voc_sheet_names, index=0, key="voc_sheet_sel")
        voc_df = _read_excel_sheet(voc_file_bytes, sel_sheet)

# VOC 컬럼 매핑
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

if voc_df is not None:
    voc_df = standardize_columns(voc_df)

# VOC constants
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
REP_COL    = next((c for c in REP_CANDIDATES if (voc_df is not None and c in voc_df.columns)), None)
SUBREP_COL = next((c for c in SUBREP_CANDIDATES if (voc_df is not None and c in voc_df.columns)), None)

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
DISPLAY_MAP = {"DEN":"DMNT","DMNT":"DMNT","methyl jasmonate (20180404ATFtest)":"Methyl jasmonate","methyl jasmonate (temporary)":"Methyl jasmonate"}
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

voc_columns = resolve_voc_columns(voc_df, VOC_24_CANDIDATES) if voc_df is not None else []

if voc_df is not None and INTERVAL_COL in voc_df.columns:
    voc_df[INTERVAL_COL] = pd.to_numeric(voc_df[INTERVAL_COL], errors="coerce")

# =========================================================
# 2) 분석 모드 선택 (VOC 3종 + 환경 데이터 분석)
# =========================================================
mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝", "환경 데이터 분석"])

# =========================================================
# 3) VOC 모드들
# =========================================================
def apply_filters(df, chamber_sel, line_sel, progress_sel, rep_sel):
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

def add_facets(kwargs, data_frame, facet_by_chamber, facet_by_line):
    if facet_by_chamber and CHAMBER_COL in data_frame.columns:
        kwargs["facet_col"] = CHAMBER_COL
    if facet_by_line and LINE_COL in data_frame.columns:
        if "facet_col" in kwargs:
            kwargs["facet_row"] = LINE_COL
        else:
            kwargs["facet_col"] = LINE_COL
    return kwargs

if mode in ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"]:
    if voc_df is None or not len(voc_columns):
        st.info("VOC 데이터 업로드 후 이용해주세요.")
    else:
        # 공통 필터
        chambers = ["전체"] + sorted(voc_df[CHAMBER_COL].dropna().astype(str).unique().tolist()) if CHAMBER_COL in voc_df.columns else ["전체"]
        lines    = ["전체"] + sorted(voc_df[LINE_COL].dropna().astype(str).unique().tolist()) if LINE_COL in voc_df.columns else ["전체"]
        chamber_sel = st.sidebar.selectbox("🏠 Chamber", chambers, index=0)
        line_sel    = st.sidebar.selectbox("🧵 Line", lines, index=0)
        intervals_all = sorted(voc_df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in voc_df.columns else []
        reps_all = ["전체"] + sorted(voc_df[REP_COL].dropna().astype(str).unique().tolist()) if REP_COL else ["전체"]
        progress_vals_all = sorted(voc_df[PROGRESS_COL].dropna().astype(str).unique().tolist()) if PROGRESS_COL in voc_df.columns else []
        rep_sel = st.sidebar.selectbox("🔁 Repetition", reps_all, index=0) if REP_COL else "전체"
        progress_sel = st.sidebar.multiselect("🧭 Progress(복수 선택 가능)", progress_vals_all, default=progress_vals_all)
        facet_by_chamber = st.sidebar.checkbox("Chamber로 분할 보기", value=False)
        facet_by_line    = st.sidebar.checkbox("Line으로 분할 보기", value=False)
        err_mode = st.sidebar.radio("오차 기준", ["SD", "SEM"], index=0)

        if mode != "전체 VOC 스크리닝":
            selected_voc = st.sidebar.selectbox("📌 VOC 물질 선택", [display_name(c) for c in voc_columns])
            inv_map = {display_name(c): c for c in voc_columns}
            selected_voc_internal = inv_map[selected_voc]
        else:
            selected_voc, selected_voc_internal = None, None

        filtered_df = apply_filters(voc_df, chamber_sel, line_sel, progress_sel, rep_sel)

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
                    per_subrep = data_use.groupby(group_keys + ([REP_COL] if REP_COL else []) + [SUBREP_COL])[selected_voc_internal].mean().reset_index()
                else:
                    per_subrep = data_use.copy()

                if REP_COL and REP_COL in data_use.columns:
                    per_rep = per_subrep.groupby(group_keys + [REP_COL])[selected_voc_internal].mean().reset_index()
                    grouped = per_rep.groupby(group_keys)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index()
                else:
                    grouped = per_subrep.groupby(group_keys)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index()

                grouped["err"] = grouped.apply(lambda r: sem_from_sd(r["sd"], r["n"]) if err_mode=="SEM" else r["sd"], axis=1)

                fig_kwargs = dict(x=TREAT_COL, y="mean", labels={"mean": y_label, TREAT_COL: "처리"}, title=f"{selected_voc} - 처리별 평균 비교 ({title_suffix})", **color_kw)
                fig_kwargs = add_facets(fig_kwargs, grouped, facet_by_chamber, facet_by_line)
                fig = px.bar(grouped, **fig_kwargs, error_y="err", barmode="group")
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

            else:  # 박스플롯
                use_rep_agg_box = st.sidebar.checkbox("박스플롯도 반복 평균 기반으로 요약", value=False) if REP_COL else False
                if use_rep_agg_box and REP_COL:
                    group_keys_box = [TREAT_COL]
                    if PROGRESS_COL in data_use.columns: group_keys_box.append(PROGRESS_COL)
                    if CHAMBER_COL in data_use.columns and facet_by_chamber: group_keys_box.append(CHAMBER_COL)
                    if LINE_COL in data_use.columns and facet_by_line: group_keys_box.append(LINE_COL)
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
                fig_kwargs = dict(x=TREAT_COL, y=y_for_box, labels={y_for_box: y_label, TREAT_COL: "처리"}, title=f"{selected_voc} - 처리별 분포 (박스플롯) ({title_suffix})", points="outliers", **color_kw)
                fig_kwargs = add_facets(fig_kwargs, data_for_box, facet_by_chamber, facet_by_line)
                fig = px.box(data_for_box, **fig_kwargs)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

        elif mode == "시간별 VOC 변화":
            treatments = sorted(voc_df[TREAT_COL].dropna().astype(str).unique().tolist()) if TREAT_COL in voc_df.columns else []
            selected_treatment = st.sidebar.selectbox("🧪 처리구 선택", ["전체"] + treatments)

            if selected_treatment == "전체":
                data_use = filtered_df.copy()
                title_prefix = "모든 처리"
            else:
                data_use = filtered_df[filtered_df[TREAT_COL].astype(str) == str(selected_treatment)].copy()
                title_prefix = f"{selected_treatment} 처리"

            tick_vals = sorted(voc_df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in voc_df.columns else []
            group_keys_display = [INTERVAL_COL]
            if selected_treatment == "전체": group_keys_display.append(TREAT_COL)
            if PROGRESS_COL in data_use.columns: group_keys_display.append(PROGRESS_COL)
            if CHAMBER_COL in data_use.columns and facet_by_chamber: group_keys_display.append(CHAMBER_COL)
            if LINE_COL in data_use.columns and facet_by_line: group_keys_display.append(LINE_COL)

            n_rep = data_use[REP_COL].nunique() if REP_COL and REP_COL in data_use.columns else 0
            if SUBREP_COL and SUBREP_COL in data_use.columns:
                per_subrep_ts = data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[selected_voc_internal].mean().reset_index()
            else:
                per_subrep_ts = data_use.copy()
            if n_rep and n_rep >= 2:
                per_rep_ts = per_subrep_ts.groupby(group_keys_display + [REP_COL])[selected_voc_internal].mean().reset_index()
                final = per_rep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis = "반복 SD/SEM"
            else:
                final = per_subrep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis = "소반복 SD/SEM"
            final["err"] = final.apply(lambda r: sem_from_sd(r["sd"], r["n"]) if err_mode=="SEM" else r["sd"], axis=1)

            fig_kwargs = dict(x=INTERVAL_COL, y="mean", error_y="err", markers=True, labels={INTERVAL_COL: "Interval (h)", "mean": f"{selected_voc} 평균농도 (ppb)"}, title=f"{title_prefix} - {selected_voc} 변화 추이 (평균±{err_mode}, 기준: {err_basis})")
            if selected_treatment == "전체": fig_kwargs["color"] = TREAT_COL
            elif PROGRESS_COL in data_use.columns: fig_kwargs["color"] = PROGRESS_COL
            fig_kwargs = add_facets(fig_kwargs, final, facet_by_chamber, facet_by_line)
            fig_voc = px.line(final, **fig_kwargs)
            if tick_vals: fig_voc.update_xaxes(tickmode='array', tickvals=tick_vals)
            fig_voc.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_voc, use_container_width=True)

            # 환경 보조 시리즈
            for env_col in [TEMP_COL, HUMID_COL]:
                if env_col not in data_use.columns: continue
                if SUBREP_COL and SUBREP_COL in data_use.columns:
                    per_subrep_env = data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[env_col].mean().reset_index()
                else:
                    per_subrep_env = data_use.copy()
                if n_rep and n_rep >= 2:
                    per_rep_env = per_subrep_env.groupby(group_keys_display + ([REP_COL] if REP_COL else []))[env_col].mean().reset_index()
                    ts_env = per_rep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                    err_basis_env = "반복 SD/SEM"
                else:
                    ts_env = per_subrep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                    err_basis_env = "소반복 SD/SEM"
                ts_env["err"] = ts_env.apply(lambda r: sem_from_sd(r["sd"], r["n"]) if err_mode=="SEM" else r["sd"], axis=1)
                ylab = "온도 (°C)" if env_col == TEMP_COL else "상대습도 (%)" if env_col == HUMID_COL else env_col
                fig_kwargs_env = dict(x=INTERVAL_COL, y="mean", error_y="err", markers=True, labels={INTERVAL_COL: "Interval (h)", "mean": ylab}, title=f"{title_prefix} - {env_col} 변화 추이 (평균±{err_mode}, 기준: {err_basis_env})")
                if selected_treatment == "전체": fig_kwargs_env["color"] = TREAT_COL
                elif PROGRESS_COL in data_use.columns: fig_kwargs_env["color"] = PROGRESS_COL
                fig_kwargs_env = add_facets(fig_kwargs_env, ts_env, facet_by_chamber, facet_by_line)
                fig_env = px.line(ts_env, **fig_kwargs_env)
                if tick_vals: fig_env.update_xaxes(tickmode='array', tickvals=tick_vals)
                fig_env.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_env, use_container_width=True)

        else:  # 전체 VOC 스크리닝 (간단 ANOVA p 값만 요약)
            selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all if 'intervals_all' in locals() else ["전체"], key="scr_interval")
            alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f", key="scr_alpha")
            include_rep_block_scr = st.sidebar.checkbox("반복을 블록요인으로 포함(스크리닝)", value=bool(REP_COL)) if REP_COL else False

            if selected_interval == "전체":
                data_use = filtered_df.copy()
                title_suffix = "모든 시간"
            else:
                data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
                title_suffix = f"Interval: {selected_interval}h"

            import importlib
            sm = importlib.import_module("statsmodels.api")
            smf = importlib.import_module("statsmodels.formula.api")

            results = []
            for voc in voc_columns:
                base_cols = [TREAT_COL, voc]
                if REP_COL: base_cols.append(REP_COL)
                if SUBREP_COL: base_cols.append(SUBREP_COL)
                sub = data_use[base_cols].dropna().copy()
                if sub.empty or sub[TREAT_COL].nunique() < 2: continue
                if not all(sub.groupby(TREAT_COL)[voc].count() >= 2): continue
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
# 4) 환경 데이터 분석 (NEW: 데이터셋 선택 + 반복 지원)
# =========================================================
if mode == "환경 데이터 분석":
    st.subheader("🌱 환경 데이터 분석 (토마토/애벌레 개별)")

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

    # 자동 매핑
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
        if df is None: return None
        col_map = {}
        for c in df.columns:
            lc = normalize(c)
            mapped = None
            for canon, aliases in ENV_CANON.items():
                if lc == normalize(canon) or lc in [normalize(a) for a in aliases]:
                    mapped = canon; break
            if mapped: col_map[c] = mapped
        if col_map: df = df.rename(columns=col_map)
        return df
    def coerce_ts(df):
        if df is None: return None
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

    df_t = coerce_ts(env_standardize_columns(df_t))
    df_l = coerce_ts(env_standardize_columns(df_l))

    # 데이터셋 선택 (개별 보기)
    available_datasets = []
    if df_t is not None and not df_t.empty: available_datasets.append("Tomato")
    if df_l is not None and not df_l.empty: available_datasets.append("Larva")
    if not available_datasets:
        st.info("좌측에서 **토마토** 또는 **애벌레** 환경 파일을 업로드하세요.")
        st.stop()
    dataset = st.radio("분석할 데이터셋 선택", options=available_datasets, horizontal=True)
    df_env = df_t if dataset=="Tomato" else df_l

    # 반복 지원
    has_rep = "Repetition" in (df_env.columns if df_env is not None else [])
    rep_values = sorted(df_env["Repetition"].dropna().astype(str).unique().tolist()) if has_rep else []
    rep_choice = st.selectbox("🔁 Repetition 선택", ["전체(통합)"] + rep_values if has_rep else ["(반복 없음)"], index=0, disabled=not has_rep)
    use_overlap_reps = st.checkbox("반복 간 '겹치는 기간'만 사용(통합일 때)", value=True, disabled=(not has_rep or rep_choice!="전체(통합)"), help="반복마다 기록 기간이 다르면 교집합 시간만으로 평균/표준편차 계산")

    # 1시간 리샘플링 + 광 처리
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
            duty = par.resample(RES_RULE).apply(lambda g: float((g>0).mean()) if len(g)>0 else np.nan)
            def mean_on(g):
                g2 = g[g>0] if exclude_zero_for_mean else g
                return float(g2.mean()) if len(g2)>0 else np.nan
            def sd_on(g):
                g2 = g[g>0] if exclude_zero_for_mean else g
                return float(g2.std()) if len(g2)>1 else (0.0 if len(g2)==1 else np.nan)
            par_mean_on = par.resample(RES_RULE).apply(mean_on)
            par_sd_on   = par.resample(RES_RULE).apply(sd_on)
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
            if sub.empty: continue
            rep_map[str(r)] = hourly_metrics(sub)
        return rep_map

    rep_series = hourly_by_rep(df_env)

    # 통합 시계열
    def merge_times(rep_map, key):
        if not rep_map: return pd.DataFrame(columns=["Time","mean","sd","n"])
        time_sets = [set(df["Time"]) for df in rep_map.values() if key in df.columns]
        if not time_sets: return pd.DataFrame(columns=["Time","mean","sd","n"])
        base_times = sorted(set.intersection(*time_sets)) if use_overlap_reps else sorted(set.union(*time_sets))
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

    def available_metrics_for(df_or_map):
        keys = set()
        if isinstance(df_or_map, dict):
            for _, d in df_or_map.items():
                keys.update([c for c in d.columns if c != "Time"])
        else:
            keys.update([c for c in df_or_map.columns if c != "Time"])
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
        dfh_sel = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
        avail = available_metrics_for(dfh_sel) if dfh_sel is not None else []
    if not avail:
        st.info("사용 가능한 지표가 없습니다. 데이터 컬럼을 확인하세요.")
        st.stop()

    options_pretty = [pretty.get(k,k) for k in avail]
    metrics = st.multiselect("표시할 지표", options=options_pretty, default=options_pretty)
    inv = {pretty.get(k,k): k for k in avail}
    metrics_keys = [inv[m] for m in metrics if m in inv]

    st.markdown("### 📊 표시 유형")
    view_type = st.radio("그래프 유형", ["시계열", "막대그래프"], index=0, horizontal=True)

    # ---- 시계열 ----
    if view_type == "시계열":
        if rep_choice == "전체(통합)":
            err_mode_env = st.radio("오차 막대", ["SD","SEM"], index=0, key="env_err_all")
            for key in metrics_keys:
                agg = merge_times(rep_series, key).sort_values("Time")
                if agg.empty:
                    st.info(f"{pretty.get(key,key)}: 데이터 없음"); continue
                agg["err"] = agg.apply(lambda r: sem_from_sd(r["sd"], r["n"]) if err_mode_env=="SEM" else r["sd"], axis=1)
                safe_line(agg, x="Time", y="mean", labels={"mean":pretty.get(key,key)}, title=f"{pretty.get(key,key)} — 반복 통합(시간당 평균±{err_mode_env})", error_y="err")
        else:
            dfh = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
            if dfh is None or dfh.empty:
                st.info("선택한 반복에서 데이터가 없습니다.")
            else:
                for key in metrics_keys:
                    if key not in dfh.columns: st.info(f"{pretty.get(key,key)}: 데이터 없음"); continue
                    # Safe plot
                    safe_line(dfh, x="Time", y=key, labels={key:pretty.get(key,key)}, title=f"{pretty.get(key,key)} — Repetition {rep_choice}")

    # ---- 막대그래프 ----
    else:
        agg_basis = st.selectbox("집계 기준", ["시간당 값의 평균±SD", "일단위 합계/평균±SD"], key="env_bar_basis")
        def summarize_hours(dfh, key):
            d = dfh.dropna(subset=[key])
            if d.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
            return {"mean":float(pd.to_numeric(d[key], errors='coerce').mean()), "sd":float(pd.to_numeric(d[key], errors='coerce').std()), "n":int(d[key].count())}
        def summarize_daily(dfh, key):
            if dfh.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
            d = dfh.copy()
            d["date"] = pd.to_datetime(d["Time"], errors="coerce").dt.date
            d = d.dropna(subset=["date"])
            if key == "hli":
                grp = d.groupby("date")["hli"].sum().rename("val")
            else:
                grp = d.groupby("date")[key].mean().rename("val")
            if grp.empty: return {"mean":np.nan,"sd":np.nan,"n":0}
            return {"mean":float(grp.mean()), "sd":float(grp.std()), "n":int(grp.count())}

        if rep_choice == "전체(통합)":
            bar_mode = st.radio("막대 유형", ["반복별 막대", "반복 통합(하나의 막대)"], index=0, horizontal=True, key="env_bar_mode")
            for key in metrics_keys:
                rows = []
                for r, dfh in rep_series.items():
                    if r == "__ALL__": continue
                    if dfh is None or key not in dfh.columns: continue
                    stat = summarize_hours(dfh, key) if agg_basis=="시간당 값의 평균±SD" else summarize_daily(dfh, key)
                    rows.append({"Repetition": str(r), "mean":stat["mean"], "sd":stat["sd"], "n":stat["n"]})
                if not rows:
                    st.info(f"{pretty.get(key,key)}: 요약할 데이터 없음"); continue
                bar_df = pd.DataFrame(rows)
                if bar_mode == "반복 통합(하나의 막대)":
                    pooled = {"Repetition":"ALL", "mean":bar_df["mean"].mean(), "sd":bar_df["mean"].std(ddof=1) if len(bar_df)>1 else 0.0, "n":len(bar_df)}
                    bar_df = pd.DataFrame([pooled])
                err_mode2 = st.radio(f"{pretty.get(key,key)} — 오차", ["SD","SEM"], index=0, key=f"env_bar_err_{key}")
                bar_df["err"] = bar_df.apply(lambda r: (r["sd"]/np.sqrt(r["n"])) if (err_mode2=="SEM" and r["n"]>0) else r["sd"], axis=1)
                safe_bar(bar_df, x="Repetition", y="mean", labels={"mean":pretty.get(key,key)}, title=f"{pretty.get(key,key)} — {agg_basis}", error_y="err")
        else:
            dfh = rep_series.get(rep_choice) if has_rep else rep_series.get("__ALL__")
            for key in metrics_keys:
                if dfh is None or key not in dfh.columns:
                    st.info(f"{pretty.get(key,key)}: 데이터 없음"); continue
                stat = summarize_hours(dfh, key) if agg_basis=="시간당 값의 평균±SD" else summarize_daily(dfh, key)
                bar_df = pd.DataFrame([{"Repetition": str(rep_choice), "mean":stat["mean"], "sd":stat["sd"], "n":stat["n"]}])
                err_mode2 = st.radio(f"{pretty.get(key,key)} — 오차", ["SD","SEM"], index=0, key=f"env_bar_err_single_{key}")
                bar_df["err"] = bar_df.apply(lambda r: (r["sd"]/np.sqrt(r["n"])) if (err_mode2=="SEM" and r["n"]>0) else r["sd"], axis=1)
                safe_bar(bar_df, x="Repetition", y="mean", labels={"mean":pretty.get(key,key)}, title=f"{pretty.get(key,key)} — {agg_basis}", error_y="err")

# ---------- 원본 데이터 보기 ----------
with st.expander("🔍 VOC 원본 데이터 보기"):
    if voc_df is not None:
        st.dataframe(voc_df, use_container_width=True)
    else:
        st.caption("VOC 데이터가 업로드되지 않았습니다.")
with st.expander("🔍 환경 원본 데이터 컬럼 도움말"):
    st.markdown("""
필수/선택 컬럼 (자동 매핑 지원)
- **Timestamp** _또는_ (**Date** + **Time**)
- **Temperature (℃)**, **Relative humidity (%)**
- **PAR Light (μmol·m2·s-1)** (선택, 광 분석 시 필요)
- **Repetition** (선택, 반복 분석 시)
""")
