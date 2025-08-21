import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Lazy imports for heavy stats to avoid boot failures if libs missing
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
    return sm, smf, MultiComparison, sp, HAS_SCPH

st.set_page_config(page_title="VOC 실험 시각화", layout="wide")
st.title("🌿 식물 VOC 실험 결과 시각화")

# -------------------------
# File input (robust for Streamlit Cloud)
# -------------------------
st.sidebar.header("📁 데이터 불러오기")
demo_btn = st.sidebar.button("🧪 데모 데이터 불러오기")

uploaded = st.sidebar.file_uploader("VOC_data.xlsx 업로드 (또는 CSV)", type=["xlsx","xls","csv"])

df = None
load_error = None

try:
    if demo_btn:
        # Minimal demo dataset
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
    elif uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
except Exception as e:
    load_error = str(e)

if df is None and not load_error:
    st.info("왼쪽 사이드바에서 **VOC_data.xlsx** 또는 CSV를 업로드하거나, **🧪 데모 데이터** 버튼을 눌러 시작하세요.")
if load_error:
    st.error(f"데이터 로딩 오류: {load_error}")

if df is not None:
    # -------------------------
    # Column definitions and auto-detection
    # -------------------------
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
    REP_COL    = next((c for c in REP_CANDIDATES if c in df.columns), None)
    SUBREP_COL = next((c for c in SUBREP_CANDIDATES if c in df.columns), None)

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
        resolved = []
        for col in candidates:
            if col in df.columns:
                resolved.append(col)
            elif col == "DEN" and "DMNT" in df.columns:
                resolved.append("DMNT")
        # allow any extra numeric columns as VOCs (fallback)
        numeric_candidates = [c for c in df.columns if c not in resolved and pd.api.types.is_numeric_dtype(df[c])]
        return resolved if resolved else numeric_candidates

    voc_columns = resolve_voc_columns(df, VOC_24_CANDIDATES)
    if not voc_columns:
        st.error("VOC 수치형 컬럼을 찾지 못했습니다. 엑셀 헤더를 확인하거나 데모 데이터를 사용하세요.")
        st.stop()
    elif set(voc_columns) != set(VOC_24_CANDIDATES):
        st.info(
            (
                "데이터에서 감지된 VOC 컬럼: "
                f"{', '.join([display_name(c) for c in voc_columns])}"
            )
        )

    # Interval numeric coercion
    if INTERVAL_COL in df.columns:
        df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")

    # -------------------------
    # Sidebar filters
    # -------------------------
    st.sidebar.header("🔧 분석 옵션")

    chambers = ["전체"] + sorted(df[CHAMBER_COL].dropna().astype(str).unique().tolist()) if CHAMBER_COL in df.columns else ["전체"]
    lines    = ["전체"] + sorted(df[LINE_COL].dropna().astype(str).unique().tolist()) if LINE_COL in df.columns else ["전체"]
    chamber_sel = st.sidebar.selectbox("🏠 Chamber", chambers, index=0)
    line_sel    = st.sidebar.selectbox("🧵 Line", lines, index=0)

    treatments = sorted(df[TREAT_COL].dropna().astype(str).unique().tolist()) if TREAT_COL in df.columns else []
    treatments_for_ts = ["전체"] + treatments
    intervals_all = sorted(df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in df.columns else []
    reps_all = ["전체"] + sorted(df[REP_COL].dropna().astype(str).unique().tolist()) if REP_COL else ["전체"]
    progress_vals_all = sorted(df[PROGRESS_COL].dropna().astype(str).unique().tolist()) if PROGRESS_COL in df.columns else []

    rep_sel = st.sidebar.selectbox("🔁 Repetition", reps_all, index=0) if REP_COL else "전체"
    progress_sel = st.sidebar.multiselect("🧭 Progress(복수 선택 가능)", progress_vals_all, default=progress_vals_all)

    mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"])

    if mode != "전체 VOC 스크리닝":
        selected_voc = st.sidebar.selectbox("📌 VOC 물질 선택", [display_name(c) for c in voc_columns])
        inv_map = {display_name(c): c for c in voc_columns}
        selected_voc_internal = inv_map[selected_voc]
    else:
        selected_voc = None
        selected_voc_internal = None

    facet_by_chamber = st.sidebar.checkbox("Chamber로 분할 보기", value=False)
    facet_by_line    = st.sidebar.checkbox("Line으로 분할 보기", value=False)
    err_mode = st.sidebar.radio("오차 기준", ["SD", "SEM"], index=0)
    show_subrep_lines = st.sidebar.checkbox("소반복 라인 표시", value=bool(SUBREP_COL)) if SUBREP_COL else False

    # -------------------------
    # Apply filters
    # -------------------------
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

    filtered_df = apply_filters(df)

    # -------------------------
    # Utilities
    # -------------------------
    def add_facets(kwargs, data_frame):
        if facet_by_chamber and CHAMBER_COL in data_frame.columns:
            kwargs["facet_col"] = CHAMBER_COL
        if facet_by_line and LINE_COL in data_frame.columns:
            if "facet_col" in kwargs:
                kwargs["facet_row"] = LINE_COL
            else:
                kwargs["facet_col"] = LINE_COL
        return kwargs

    def p_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    def cld_from_nonsig(groups, pairs_ns):
        groups = list(groups)
        letters = {g: "" for g in groups}
        remaining = set(groups)
        alphabet = list("abcdefghijklmnopqrstuvwxyz")
        li = 0
        while remaining:
            letter = alphabet[li % len(alphabet)]
            bucket = []
            for g in list(remaining):
                ok = True
                for h in bucket:
                    pair = tuple(sorted((g, h)))
                    if pair not in pairs_ns and g != h:
                        ok = False
                        break
                if ok:
                    bucket.append(g)
            for g in bucket:
                letters[g] += letter
                remaining.remove(g)
            li += 1
        for g in groups:
            if letters[g] == "":
                letters[g] = "a"
        return letters

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

    # -------------------------
    # Modes
    # -------------------------
    if mode in ["처리별 VOC 비교", "시간별 VOC 변화"]:
        if mode == "처리별 VOC 비교":
            chart_type = st.sidebar.radio("차트 유형", ["막대그래프", "박스플롯"], index=0)
            selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all)

            # Safe stats import only if needed
            show_anova = False
            posthoc_choices = []
            alpha = 0.05
            letters_method_for_plot = None
            include_rep_block = False

            if chart_type == "막대그래프":
                show_anova = st.sidebar.checkbox("ANOVA 분석 표시(막대그래프 전용)", value=False)
                include_rep_block = st.sidebar.checkbox("반복을 블록요인으로 포함", value=bool(REP_COL)) if REP_COL else False
                if show_anova:
                    alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f")
                    allowed_methods = ["Tukey HSD", "Duncan"]
                    posthoc_choices = st.sidebar.multiselect("사후검정 선택(복수 가능)", allowed_methods, default=["Tukey HSD"])
                    if len(posthoc_choices) > 0:
                        letters_method_for_plot = st.sidebar.selectbox("그래프에 표시할 유의문자 기준", posthoc_choices, index=0)

        else:  # 시간별 VOC 변화
            selected_treatment = st.sidebar.selectbox("🧪 처리구 선택", ["전체"] + treatments)

        # Subset by interval for bar/box
        if mode == "처리별 VOC 비교":
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

                # sub-rep -> rep -> treatment summary
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

                if show_anova:
                    sm, smf, MultiComparison, sp, HAS_SCPH = _lazy_import_stats()
                    base_cols = [TREAT_COL, selected_voc_internal]
                    if REP_COL: base_cols.append(REP_COL)
                    anova_df = data_use[base_cols].dropna().copy()
                    if anova_df[TREAT_COL].nunique() >= 2 and all(anova_df.groupby(TREAT_COL)[selected_voc_internal].count() >= 2):
                        a_df = anova_df.rename(columns={selected_voc_internal: "y", TREAT_COL: "treat"})
                        try:
                            if include_rep_block and REP_COL and REP_COL in a_df.columns:
                                a_df["rep"] = a_df[REP_COL].astype(str)
                                model = smf.ols("y ~ C(treat) + C(rep)", data=a_df).fit()
                            else:
                                model = smf.ols("y ~ C(treat)", data=a_df).fit()
                        except Exception:
                            model = smf.ols("y ~ C(treat)", data=a_df).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)

                        try:
                            pval = float(anova_table.loc["C(treat)", "PR(>F)"])
                        except Exception:
                            pval = float("nan")
                        stars = p_to_stars(pval) if np.isfinite(pval) else "n/a"
                        fig.update_layout(title=f"{selected_voc} - 처리별 평균 비교 ({title_suffix})  |  ANOVA: p={pval:.4g} ({stars})")

                        posthoc_letters = {}
                        summary_tables = []
                        if MultiComparison and "Tukey HSD" in posthoc_choices and np.isfinite(pval):
                            mc = MultiComparison(a_df["y"], a_df["treat"])
                            tukey = mc.tukeyhsd(alpha=alpha)
                            tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                            ns_pairs = set()
                            for _, row in tukey_df.iterrows():
                                g1, g2, reject = str(row["group1"]), str(row["group2"]), bool(row["reject"])
                                if not reject:
                                    ns_pairs.add(tuple(sorted((g1, g2))))
                            treat_order = sorted(a_df["treat"].astype(str).unique().tolist())
                            for g in treat_order:
                                ns_pairs.add((g, g))
                            # build letters
                            def _cld(groups, pairs_ns):
                                groups = list(groups)
                                letters = {g: "" for g in groups}
                                remaining = set(groups)
                                alphabet = list("abcdefghijklmnopqrstuvwxyz")
                                li = 0
                                while remaining:
                                    letter = alphabet[li % len(alphabet)]
                                    bucket = []
                                    for g in list(remaining):
                                        ok = True
                                        for h in bucket:
                                            pair = tuple(sorted((g, h)))
                                            if pair not in pairs_ns and g != h:
                                                ok = False
                                                break
                                        if ok:
                                            bucket.append(g)
                                    for g in bucket:
                                        letters[g] += letter
                                        remaining.remove(g)
                                    li += 1
                                for g in groups:
                                    if letters[g] == "":
                                        letters[g] = "a"
                                return letters
                            letters = _cld(treat_order, ns_pairs)
                            posthoc_letters["Tukey HSD"] = letters

                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("ANOVA 결과")
                        st.dataframe(anova_table)
                    else:
                        st.info("ANOVA를 수행하기 위한 표본 수가 충분하지 않습니다. (처리 수준 ≥2, 각 처리 n≥2 권장)")
                else:
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
                    labels={y_for_box: f"{selected_voc} 농도 (ppb)", TREAT_COL: "처리"},
                    title=f"{selected_voc} - 처리별 분포 (박스플롯) ({title_suffix})",
                    points="outliers",
                    **({"color": PROGRESS_COL} if PROGRESS_COL in data_use.columns else {}),
                )
                fig_kwargs = add_facets(fig_kwargs, data_for_box)
                fig = px.box(data_for_box, **fig_kwargs)
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

        else:  # 시간별 VOC 변화
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

            if show_subrep_lines and SUBREP_COL and SUBREP_COL in data_use.columns:
                disp_df = per_subrep_ts.rename(columns={selected_voc_internal: "val"})
                fig_kwargs_sub = dict(
                    x=INTERVAL_COL,
                    y="val",
                    hover_data=[SUBREP_COL] + ([REP_COL] if REP_COL else []),
                    opacity=0.35,
                )
                if selected_treatment == "전체":
                    fig_kwargs_sub["color"] = TREAT_COL
                elif PROGRESS_COL in data_use.columns:
                    fig_kwargs_sub["color"] = PROGRESS_COL
                fig_kwargs_sub = add_facets(fig_kwargs_sub, disp_df)
                fig_sub = px.line(disp_df, **fig_kwargs_sub)
                fig_sub.update_traces(line=dict(width=1))
                st.plotly_chart(fig_sub, use_container_width=True)

            # Env variables if exist
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
                if selected_treatment == "전체":
                    fig_kwargs_env["color"] = TREAT_COL
                elif PROGRESS_COL in data_use.columns:
                    fig_kwargs_env["color"] = PROGRESS_COL

                fig_kwargs_env = add_facets(fig_kwargs_env, ts_env)
                fig_env = px.line(ts_env, **fig_kwargs_env)
                if tick_vals:
                    fig_env.update_xaxes(tickmode='array', tickvals=tick_vals)
                fig_env.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_env, use_container_width=True)

    else:  # 전체 VOC 스크리닝
        st.subheader("🔎 전체 VOC 스크리닝 (ANOVA 일괄 분석)")

        selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all, key="scr_interval")
        alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f", key="scr_alpha")
        do_posthoc = st.sidebar.checkbox("사후검정(Tukey/Duncan) 요약 포함", value=True)
        posthoc_method = st.sidebar.selectbox("사후검정 방법", ["Tukey HSD", "Duncan"], index=0)
        only_sig = st.sidebar.checkbox("유의 VOC만 표시 (p < α)", value=False)
        show_letters_cols = st.sidebar.checkbox("Letters 요약 문자열 포함", value=True)
        include_rep_block_scr = st.sidebar.checkbox("반복을 블록요인으로 포함(스크리닝)", value=bool(REP_COL)) if REP_COL else False

        if selected_interval == "전체":
            data_use = filtered_df.copy()
            title_suffix = "모든 시간"
        else:
            data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
            title_suffix = f"Interval: {selected_interval}h"

        sm, smf, MultiComparison, sp, HAS_SCPH = _lazy_import_stats()

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
                stars = p_to_stars(pval)
            except Exception:
                pval, stars = np.nan, "ERR"

            letters_str = ""
            if do_posthoc and np.isfinite(pval) and MultiComparison:
                treat_order = sorted(a_df["treat"].astype(str).unique().tolist())
                if posthoc_method == "Tukey HSD":
                    mc = MultiComparison(a_df["y"], a_df["treat"])
                    tukey = mc.tukeyhsd(alpha=alpha)
                    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    ns_pairs = set()
                    for _, row in tukey_df.iterrows():
                        g1, g2, reject = str(row["group1"]), str(row["group2"]), bool(row["reject"])
                        if not reject:
                            ns_pairs.add(tuple(sorted((g1, g2))))
                    for g in treat_order:
                        ns_pairs.add((g, g))
                    def _cld(groups, pairs_ns):
                        groups = list(groups)
                        letters = {g: "" for g in groups}
                        remaining = set(groups)
                        alphabet = list("abcdefghijklmnopqrstuvwxyz")
                        li = 0
                        while remaining:
                            letter = alphabet[li % len(alphabet)]
                            bucket = []
                            for g in list(remaining):
                                ok = True
                                for h in bucket:
                                    pair = tuple(sorted((g, h)))
                                    if pair not in pairs_ns and g != h:
                                        ok = False
                                        break
                                if ok:
                                    bucket.append(g)
                            for g in bucket:
                                letters[g] += letter
                                remaining.remove(g)
                            li += 1
                        for g in groups:
                            if letters[g] == "":
                                letters[g] = "a"
                        return letters
                    letters = _cld(treat_order, ns_pairs)
                    if show_letters_cols:
                        letters_str = "; ".join([f"{t}={letters.get(str(t), '')}" for t in treat_order])
                elif posthoc_method == "Duncan" and sp is not None:
                    try:
                        duncan_mat = sp.posthoc_duncan(a_df, val_col="y", group_col="treat", alpha=alpha)
                        ns_pairs = set()
                        for g1 in duncan_mat.index.astype(str):
                            for g2 in duncan_mat.columns.astype(str):
                                if g1 == g2:
                                    ns_pairs.add(tuple(sorted((g1, g2))))
                                else:
                                    p = duncan_mat.loc[g1, g2]
                                    if pd.isna(p) or p >= alpha:
                                        ns_pairs.add(tuple(sorted((g1, g2))))
                        def _cld(groups, pairs_ns):
                            groups = list(groups)
                            letters = {g: "" for g in groups}
                            remaining = set(groups)
                            alphabet = list("abcdefghijklmnopqrstuvwxyz")
                            li = 0
                            while remaining:
                                letter = alphabet[li % len(alphabet)]
                                bucket = []
                                for g in list(remaining):
                                    ok = True
                                    for h in bucket:
                                        pair = tuple(sorted((g, h)))
                                        if pair not in pairs_ns and g != h:
                                            ok = False
                                            break
                                    if ok:
                                        bucket.append(g)
                                for g in bucket:
                                    letters[g] += letter
                                    remaining.remove(g)
                                li += 1
                            for g in groups:
                                if letters[g] == "":
                                    letters[g] = "a"
                            return letters
                        letters = _cld(treat_order, ns_pairs)
                        if show_letters_cols:
                            letters_str = "; ".join([f"{t}={letters.get(str(t), '')}" for t in treat_order])
                    except Exception as e:
                        letters_str = f"Duncan err: {e}"

            results.append({
                "VOC": display_name(voc),
                "p_value": pval,
                "Significance": stars,
                "Letters": letters_str,
            })

        if not results:
            st.info("조건에 맞는 데이터가 없어 스크리닝 결과가 비어있습니다. 필터/Interval/Progress를 확인하세요.")
        else:
            res_df = pd.DataFrame(results).sort_values("p_value", na_position="last")
            if only_sig:
                res_df = res_df[res_df["p_value"] < alpha]
            st.markdown(f"**Interval: {title_suffix}**, α={alpha}")
            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "⬇️ 스크리닝 결과 CSV",
                data=res_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="voc_screening_results.csv",
                mime="text/csv",
            )

    # -------------------------
    # Raw data preview
    # -------------------------
    with st.expander("🔍 원본 데이터 보기"):
        st.dataframe(df, use_container_width=True)
