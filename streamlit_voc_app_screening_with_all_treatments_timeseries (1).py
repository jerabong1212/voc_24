import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison

# Optional: Duncan test (via scikit-posthocs)
try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except Exception:
    HAS_SCPH = False

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(page_title="VOC 실험 시각화", layout="wide")
st.title("🌿 식물 VOC 실험 결과 시각화")

# -------------------------
# 데이터 로드
# -------------------------
try:
    df = pd.read_excel("VOC_data.xlsx")
except Exception as e:
    st.error(f"데이터 로딩 오류: {e}")
    st.stop()

# -------------------------
# 컬럼명(엑셀 1행)
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

# 반복/소반복 자동 감지 (여러 표기 허용)
REP_CANDIDATES    = ["Repetition", "rep", "Rep", "repetition", "반복", "반복수"]
SUBREP_CANDIDATES = ["Sub-repetition", "subrep", "Subrep", "Sub-rep", "sub-repetition", "소반복", "소반복수"]
REP_COL    = next((c for c in REP_CANDIDATES if c in df.columns), None)
SUBREP_COL = next((c for c in SUBREP_CANDIDATES if c in df.columns), None)

# -------------------------
# VOC 24종 (사용자 제공)
# -------------------------
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
    return resolved


voc_columns = resolve_voc_columns(df, VOC_24_CANDIDATES)
if len(voc_columns) == 0:
    st.error("VOC 컬럼을 찾지 못했습니다. 파일 헤더(24종) 확인이 필요합니다.")
    st.stop()
elif len(voc_columns) < len(VOC_24_CANDIDATES):
    st.info(
        f"설정된 24종 중 데이터에서 발견한 컬럼 수: {len(voc_columns)} / {len(VOC_24_CANDIDATES)}
"
        f"감지된 VOC: {', '.join([display_name(c) for c in voc_columns])}"
    )

# -------------------------
# 처리/인터벌/반복 목록
# -------------------------
if TREAT_COL not in df.columns or INTERVAL_COL not in df.columns:
    st.error(f"필수 키 컬럼 누락: {TREAT_COL}, {INTERVAL_COL}")
    st.stop()

# 기본 처리구 목록(정렬)
treatments = sorted(df[TREAT_COL].dropna().astype(str).unique().tolist())
# 시간별 모드에서 사용할 옵션(전체 포함)
treatments_for_ts = ["전체"] + treatments

intervals = sorted(pd.to_numeric(df[INTERVAL_COL], errors="coerce").dropna().unique().tolist())

expected_intervals = [-1, 0, 1, 2, 3, 4, 5, 6, 12, 18, 24]
missing = [x for x in expected_intervals if x not in intervals]
if missing:
    st.info(f"데이터에 없는 Interval(h): {missing} (기준 간격)")

# 반복수 옵션(있을 때만)
reps_all = ["전체"] + sorted(df[REP_COL].dropna().astype(str).unique().tolist()) if REP_COL else ["전체"]

# -------------------------
# 사이드바
# -------------------------
st.sidebar.header("🔧 분석 옵션")
if CHAMBER_COL not in df.columns or LINE_COL not in df.columns:
    st.warning("Chamber/Line 컬럼이 없습니다. 파일 헤더를 확인하세요.")

chambers = ["전체"] + sorted(df[CHAMBER_COL].dropna().astype(str).unique().tolist()) if CHAMBER_COL in df.columns else ["전체"]
lines    = ["전체"] + sorted(df[LINE_COL].dropna().astype(str).unique().tolist()) if LINE_COL in df.columns else ["전체"]

chamber_sel = st.sidebar.selectbox("🏠 Chamber", chambers, index=0)
line_sel    = st.sidebar.selectbox("🧵 Line", lines, index=0)

# 반복수 선택(전역 필터)
rep_sel = st.sidebar.selectbox("🔁 Repetition", reps_all, index=0) if REP_COL else "전체"

# 소반복 표시/집계 옵션
show_subrep_lines = st.sidebar.checkbox("소반복 라인 표시", value=bool(SUBREP_COL)) if SUBREP_COL else False

progress_vals_all = sorted(df[PROGRESS_COL].dropna().astype(str).unique().tolist()) if PROGRESS_COL in df.columns else []
progress_sel = st.sidebar.multiselect("🧭 Progress(복수 선택 가능)", progress_vals_all, default=progress_vals_all)

mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"])

# 공통 VOC 선택(스크리닝 모드는 비활성화)
if mode != "전체 VOC 스크리닝":
    selected_voc = st.sidebar.selectbox("📌 VOC 물질 선택", [display_name(c) for c in voc_columns])
    inv_map = {display_name(c): c for c in voc_columns}
    selected_voc_internal = inv_map[selected_voc]
else:
    selected_voc = None
    selected_voc_internal = None

facet_by_chamber = st.sidebar.checkbox("Chamber로 분할 보기", value=False)
facet_by_line    = st.sidebar.checkbox("Line으로 분할 보기", value=False)

# 오차 기준 선택 (SD/SEM)
err_mode = st.sidebar.radio("오차 기준", ["SD", "SEM"], index=0)

# -------------------------
# 공통: 필터 적용
# -------------------------
filtered_df = df.copy()
if CHAMBER_COL in filtered_df.columns and chamber_sel != "전체":
    filtered_df = filtered_df[filtered_df[CHAMBER_COL].astype(str) == str(chamber_sel)]
if LINE_COL in filtered_df.columns and line_sel != "전체":
    filtered_df = filtered_df[filtered_df[LINE_COL].astype(str) == str(line_sel)]
if PROGRESS_COL in filtered_df.columns and progress_sel:
    filtered_df = filtered_df[filtered_df[PROGRESS_COL].astype(str).isin(progress_sel)]
if REP_COL and rep_sel != "전체":
    filtered_df = filtered_df[filtered_df[REP_COL].astype(str) == str(rep_sel)]

# -------------------------
# 유틸
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


def format_letters_string(letters_dict, treat_order):
    parts = []
    for t in treat_order:
        parts.append(f"{t}={letters_dict.get(str(t), '')}")
    return "; ".join(parts)


def sem_from_sd(sd, n):
    try:
        return sd / np.sqrt(n) if (sd is not None and n and n > 0) else np.nan
    except Exception:
        return np.nan


def attach_error_col(df_stats, err_mode):
    df_stats = df_stats.copy()
    if err_mode == "SEM":
        df_stats["err"] = df_stats.apply(lambda r: sem_from_sd(r.get("sd", np.nan), r.get("n", np.nan)), axis=1)
    else:
        df_stats["err"] = df_stats.get("sd", np.nan)
    return df_stats

# -------------------------
# 처리별 VOC 비교 / 시간별 VOC 변화
# -------------------------
if mode in ["처리별 VOC 비교", "시간별 VOC 변화"]:
    if mode == "처리별 VOC 비교":
        chart_type = st.sidebar.radio("차트 유형", ["막대그래프", "박스플롯"], index=0)
        selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals)

        show_anova = False
        posthoc_choices = []
        alpha = 0.05
        letters_method_for_plot = None

        if chart_type == "막대그래프":
            show_anova = st.sidebar.checkbox("ANOVA 분석 표시(막대그래프 전용)", value=False)
            # 반복을 블록요인으로 포함 옵션
            include_rep_block = st.sidebar.checkbox("반복을 블록요인으로 포함", value=bool(REP_COL)) if REP_COL else False
            if show_anova:
                alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f")
                allowed_methods = ["Tukey HSD"]
                if HAS_SCPH:
                    allowed_methods.append("Duncan")
                posthoc_choices = st.sidebar.multiselect("사후검정 선택(복수 가능)", allowed_methods, default=allowed_methods[:1])
                if len(posthoc_choices) > 0:
                    letters_method_for_plot = st.sidebar.selectbox("그래프에 표시할 유의문자 기준", posthoc_choices, index=0)

    else:  # 시간별 VOC 변화
        # ✅ 처리구 선택에 "전체" 옵션 추가
        selected_treatment = st.sidebar.selectbox("🧪 처리구 선택", treatments_for_ts)

    # 데이터 subset
    if mode == "처리별 VOC 비교":
        if selected_interval == "전체":
            data_use = filtered_df.copy()
            title_suffix = "모든 시간"
        else:
            data_use = filtered_df[pd.to_numeric(filtered_df[INTERVAL_COL], errors="coerce") == selected_interval].copy()
            title_suffix = f"Interval: {selected_interval}h"

        y_label = f"{selected_voc} 농도 (ppb)"
        color_kw = {"color": PROGRESS_COL} if (PROGRESS_COL in data_use.columns and data_use[PROGRESS_COL].notna().any()) else {}

        if chart_type == "막대그래프":
            group_keys = [TREAT_COL]
            if PROGRESS_COL in data_use.columns and PROGRESS_COL in data_use:
                group_keys.append(PROGRESS_COL)
            if CHAMBER_COL in data_use.columns and facet_by_chamber:
                group_keys.append(CHAMBER_COL)
            if LINE_COL in data_use.columns and facet_by_line:
                group_keys.append(LINE_COL)

            # 요약: (소반복 평균) → (반복 평균) → 처리 평균±SD/SEM
            if SUBREP_COL and SUBREP_COL in data_use.columns:
                per_subrep = data_use.groupby(group_keys + [REP_COL] + [SUBREP_COL])[selected_voc_internal].mean().reset_index() if REP_COL else data_use.groupby(group_keys + [SUBREP_COL])[selected_voc_internal].mean().reset_index()
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

                    pval = np.nan
                    try:
                        pval = float(anova_table.loc["C(treat)", "PR(>F)"])
                    except Exception:
                        for idx in anova_table.index:
                            if "C(treat)" in str(idx) or "treat" == str(idx):
                                pval = float(anova_table.loc[idx, "PR(>F)"])
                                break
                    stars = p_to_stars(pval) if not np.isnan(pval) else "n/a"
                    fig.update_layout(title=f"{selected_voc} - 처리별 평균 비교 ({title_suffix})  |  ANOVA: p={pval:.4g} ({stars})")

                    posthoc_letters = {}
                    summary_tables = []
                    means_df = grouped.groupby(TREAT_COL)["mean"].mean().reset_index().rename(columns={"mean":"Mean"})
                    counts = a_df.groupby("treat")["y"].count()
                    means_df["n"] = means_df[TREAT_COL].astype(str).map(counts).fillna(0).astype(int)
                    treat_order = means_df[TREAT_COL].astype(str).tolist()

                    if "Tukey HSD" in posthoc_choices:
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
                        letters = cld_from_nonsig(treat_order, ns_pairs)
                        posthoc_letters["Tukey HSD"] = letters

                        letters_list = [letters.get(str(g), "") for g in means_df[TREAT_COL]]
                        tukey_summary = means_df.copy()
                        tukey_summary["Letters (Tukey)"] = letters_list
                        summary_tables.append(("Tukey HSD", tukey_summary))

                        st.subheader("Tukey HSD (사후검정)")
                        st.dataframe(tukey_df)
                        st.download_button(
                            "⬇️ Tukey HSD 테이블 CSV",
                            data=tukey_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="tukey_hsd_results.csv",
                            mime="text/csv",
                        )

                    if "Duncan" in posthoc_choices:
                        if HAS_SCPH:
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
                                letters = cld_from_nonsig(treat_order, ns_pairs)
                                posthoc_letters["Duncan"] = letters

                                letters_list = [letters.get(str(g), "") for g in means_df[TREAT_COL]]
                                duncan_summary = means_df.copy()
                                duncan_summary["Letters (Duncan)"] = letters_list
                                summary_tables.append(("Duncan", duncan_summary))

                                st.subheader("Duncan (사후검정) p-value 행렬")
                                st.dataframe(duncan_mat)
                                st.download_button(
                                    "⬇️ Duncan p-value 매트릭스 CSV",
                                    data=duncan_mat.to_csv().encode("utf-8-sig"),
                                    file_name="duncan_posthoc_pvals.csv",
                                    mime="text/csv",
                                )
                            except Exception as e:
                                st.warning(f"Duncan 사후검정 계산 중 문제가 발생했습니다: {e}")
                        else:
                            st.info("Duncan 사후검정을 사용하려면 scikit-posthocs 패키지가 필요합니다. (requirements에 'scikit-posthocs' 추가)")

                    # letters 그래프 표시 (facet 미사용 시)
                    if letters_method_for_plot and (not facet_by_chamber) and (not facet_by_line):
                        letters_to_use = posthoc_letters.get(letters_method_for_plot, None)
                        if letters_to_use:
                            cat_ymax = {}
                            for tr in fig.data:
                                if getattr(tr, "type", "") == "bar":
                                    xs = list(getattr(tr, "x", []) or [])
                                    ys = list(getattr(tr, "y", []) or [])
                                    for xval, yval in zip(xs, ys):
                                        if xval is None or yval is None:
                                            continue
                                        key = str(xval)
                                        if key not in cat_ymax or yval > cat_ymax[key]:
                                            cat_ymax[key] = float(yval)
                            for xcat, yval in cat_ymax.items():
                                letter = letters_to_use.get(str(xcat), "")
                                if letter:
                                    fig.add_annotation(x=xcat, y=yval, text=letter, showarrow=False, yshift=12, font=dict(size=14))

                    st.subheader("ANOVA 결과")
                    st.dataframe(anova_table)
                    if summary_tables:
                        st.subheader("사후검정 요약 (평균 + Letters)")
                        for label, tbl in summary_tables:
                            st.markdown(f"**{label}**")
                            st.dataframe(tbl)
                    st.download_button(
                        "⬇️ ANOVA 테이블 CSV",
                        data=anova_table.to_csv().encode("utf-8-sig"),
                        file_name="anova_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("ANOVA를 수행하기 위한 표본 수가 충분하지 않습니다. (처리 수준 ≥2, 각 처리 n≥2 권장)")

            st.plotly_chart(fig, use_container_width=True)

        else:  # 박스플롯
            # 박스플롯은 기본적으로 원자료 분포를 보여줌. 필요 시 반복 평균 기반 요약 옵션 제공
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

    else:  # 시간별 VOC 변화
        # ✅ 전체 선택 시: 모든 처리구 포함
        if selected_treatment == "전체":
            data_use = filtered_df.copy()
            title_prefix = "모든 처리"
        else:
            data_use = filtered_df[filtered_df[TREAT_COL].astype(str) == str(selected_treatment)].copy()
            title_prefix = f"{selected_treatment} 처리"

        tick_vals = expected_intervals

        # 표시용 그룹키 구성
        group_keys_display = [INTERVAL_COL]
        if selected_treatment == "전체":
            group_keys_display.append(TREAT_COL)
        if PROGRESS_COL in data_use.columns:
            group_keys_display.append(PROGRESS_COL)
        if CHAMBER_COL in data_use.columns and facet_by_chamber:
            group_keys_display.append(CHAMBER_COL)
        if LINE_COL in data_use.columns and facet_by_line:
            group_keys_display.append(LINE_COL)

        # 집계 로직: 반복이 2개 이상이면 '반복 간' 평균±오차, 그렇지 않으면 '소반복 간' 평균±오차
        n_rep = data_use[REP_COL].nunique() if REP_COL and REP_COL in data_use.columns else 0

        if SUBREP_COL and SUBREP_COL in data_use.columns:
            per_subrep_ts = data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[selected_voc_internal].mean().reset_index()
        else:
            per_subrep_ts = data_use.copy()

        if n_rep and n_rep >= 2:
            # 1) 소반복 → 반복 평균, 2) 반복 간 평균±오차
            per_rep_ts = per_subrep_ts.groupby(group_keys_display + [REP_COL])[selected_voc_internal].mean().reset_index()
            final = per_rep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
            err_basis = "반복 SD/SEM"
        else:
            # 반복 1개 이하: 소반복 간 평균±오차
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
        fig_voc.update_xaxes(tickmode='array', tickvals=tick_vals)
        fig_voc.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_voc, use_container_width=True)

        # 소반복 라인 오버레이(옵션)
        if show_subrep_lines and SUBREP_COL and SUBREP_COL in data_use.columns:
            # 색상/패싯 키 동일 유지
            disp_cols = group_keys_display.copy()
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

        # 환경 변수(온도/습도)도 동일 로직 적용
        cols_exist = [c for c in [TEMP_COL, HUMID_COL] if c in data_use.columns]
        if cols_exist:
            for env_col in cols_exist:
                group_keys_env_disp = group_keys_display.copy()
                if SUBREP_COL and SUBREP_COL in data_use.columns:
                    per_subrep_env = data_use.groupby(group_keys_env_disp + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[env_col].mean().reset_index()
                else:
                    per_subrep_env = data_use.copy()

                if n_rep and n_rep >= 2:
                    per_rep_env = per_subrep_env.groupby(group_keys_env_disp + ([REP_COL] if REP_COL else []))[env_col].mean().reset_index()
                    ts_env = per_rep_env.groupby(group_keys_env_disp)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                    err_basis_env = "반복 SD/SEM"
                else:
                    ts_env = per_subrep_env.groupby(group_keys_env_disp)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
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
                fig_env.update_xaxes(tickmode='array', tickvals=tick_vals)
                fig_env.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_env, use_container_width=True)
        else:
            st.info("온도/상대습도 컬럼이 없어 환경변화 그래프는 표시하지 않습니다.")

# -------------------------
# NEW: 전체 VOC 스크리닝
# -------------------------
elif mode == "전체 VOC 스크리닝":
    st.subheader("🔎 전체 VOC 스크리닝 (ANOVA 일괄 분석)")

    selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals, key="scr_interval")
    alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f", key="scr_alpha")
    do_posthoc = st.sidebar.checkbox("사후검정(Tukey/Duncan) 요약 포함", value=True)
    posthoc_method = st.sidebar.selectbox("사후검정 방법", ["Tukey HSD"] + (["Duncan"] if HAS_SCPH else []), index=0)
    only_sig = st.sidebar.checkbox("유의 VOC만 표시 (p < α)", value=False)
    show_letters_cols = st.sidebar.checkbox("Letters 요약 문자열 포함", value=True)
    # 반복을 블록요인으로 포함 옵션(스크리닝 공통)
    include_rep_block_scr = st.sidebar.checkbox("반복을 블록요인으로 포함(스크리닝)", value=bool(REP_COL)) if REP_COL else False

    # Interval 필터
    if selected_interval == "전체":
        data_use = filtered_df.copy()
        title_suffix = "모든 시간"
    else:
        data_use = filtered_df[pd.to_numeric(filtered_df[INTERVAL_COL], errors="coerce") == selected_interval].copy()
        title_suffix = f"Interval: {selected_interval}h"

    results = []
    # 각 VOC에 대해 ANOVA 수행
    for voc in voc_columns:
        base_cols = [TREAT_COL, voc]
        if REP_COL: base_cols.append(REP_COL)
        if SUBREP_COL: base_cols.append(SUBREP_COL)
        sub = data_use[base_cols].dropna().copy()
        if sub.empty or sub[TREAT_COL].nunique() < 2:
            continue
        # 각 처리 최소 n>=2 체크
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
        if do_posthoc and np.isfinite(pval):
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
                letters = cld_from_nonsig(treat_order, ns_pairs)
                if show_letters_cols:
                    letters_str = format_letters_string(letters, treat_order)
            elif posthoc_method == "Duncan" and HAS_SCPH:
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
                    letters = cld_from_nonsig(treat_order, ns_pairs)
                    if show_letters_cols:
                        letters_str = format_letters_string(letters, treat_order)
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
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values("p_value", na_position="last")
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
# 원본 데이터 확인
# -------------------------
with st.expander("🔍 원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)
