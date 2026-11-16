import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

st.set_page_config(layout="wide")
st.title("Data Analysis For Co-pyrolisis Simulations")

st.markdown("Works in tandem with ['Processing Co-Pyrolisis Data From LAMMPS'](https://drive.google.com/drive/folders/1VZ7GhnzVKtWD6_xkPviVpw9Sr4MuUmuM?usp=drive_link). Be sure to check our guide before running the program.")


CA_COLOR = "#1f77b4"   # azul
CB_COLOR = "rgba(31,119,180,0.30)"

HA_COLOR = "#d62728"   # vermelho
HB_COLOR = "rgba(214,39,40,0.30)"

SIM_COLORS = qualitative.Plotly

STD_SUFFIX = "_DP"


ZERO_THRESHOLD = 0.5

PROPERTIES = [
    ("Temperature", "Temp"),
    ("Pressure", "Press"),
    ("Number of Molecules", "f_Species[1]"),
    ("Energy", "TotEng"),
    ("Species Present", "f_Species[2]"),
    ("Gases", "C_1-C_4"),
    ("Light Liquids", "C_5-C_12"),
    ("Light Oils", "C_13-C_20"),
    ("Heavy Oils", "C_21-C_40"),
    ("Solids", "C_41-C_200"),
]


BINS = [
    ("Gases", "C_1-C_4"),
    ("Light Liquids", "C_5-C_12"),
    ("Light Oils", "C_13-C_20"),
    ("Heavy Oils", "C_21-C_40"),
    ("Solids", "C_41-C_200"),
]

CARBON_COLUMNS = [column for _, column in BINS]
PROPERTY_LABELS = {column: label for label, column in PROPERTIES}




TIME_UNITS = [
    (1.0, "s"),
    (1e-3, "ms"),
    (1e-6, "µs"),
    (1e-9, "ns"),
    (1e-12, "ps"),
    (1e-15, "fs"),
]


def pick_unit(seconds):
    """Escolhe a unidade (fator, rótulo) mais adequada para um valor em segundos."""
    seconds = abs(seconds)
    for factor, label in TIME_UNITS:
        if seconds >= factor:
            return factor, label
    return TIME_UNITS[-1]


def fmt_time(seconds):
    """Formata um tempo em segundos usando a melhor unidade."""
    if seconds == 0:
        return "0 s"
    factor, label = pick_unit(seconds)
    return f"{seconds / factor:.4g} {label}"


def parse_positive_float(text):
    """Converte texto (aceita notação científica) em float > 0.
    Retorna (valor, erro)."""
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None, None
    try:
        value = float(text)
    except ValueError:
        return None, "Valor inválido para segundos por timestep."
    if value <= 0:
        return None, "Segundos por timestep deve ser maior que zero."
    return value, None


def add_x_column(df, seconds_per_step, factor=None, label=None):
    """Garante a coluna do eixo x. Retorna (coluna, título do eixo)."""
    if seconds_per_step is None:
        return "Timestep", "Timestep"

    if factor is None:
        max_step = df["Timestep"].max()
        max_time = (max_step if pd.notna(max_step) else 0) * seconds_per_step
        factor, label = pick_unit(
            max_time if max_time > 0 else seconds_per_step
        )

    df["Time"] = df["Timestep"] * seconds_per_step / factor
    return "Time", f"Time ({label})"



@st.cache_data(show_spinner=False)
def _read_log_bytes(data):
    return pd.read_csv(io.BytesIO(data), sep=r"\s+", engine="python")


@st.cache_data(show_spinner=False)
def _read_csv_bytes(data):
    return pd.read_csv(io.BytesIO(data))


def read_csv_ts(file):
    """Lê um CSV com coluna Timestep numérica. Retorna None se não houver arquivo."""
    if file is None:
        return None

    df = _read_csv_bytes(file.getvalue())

    if "Timestep" in df.columns:
        df["Timestep"] = pd.to_numeric(df["Timestep"], errors="coerce")

    return df


def load_merged(files):
    """Junta LOG + CH (o que estiver disponível) pelo Timestep."""
    log = files.get("LOG")
    ch = files.get("CARBONS_CSV")

    df_txt = None
    df_csv = None

    if log is not None:
        df_txt = _read_log_bytes(log.getvalue()).rename(
            columns={"Step": "Timestep"}
        )

    if ch is not None:
        df_csv = _read_csv_bytes(ch.getvalue())

    frames = [d for d in (df_txt, df_csv) if d is not None]

    if not frames:
        return None

    for d in frames:
        if "Timestep" not in d.columns:
            raise ValueError("coluna 'Timestep' (ou 'Step') não encontrada")

    cleaned = []
    for d in frames:
        d["Timestep"] = pd.to_numeric(d["Timestep"], errors="coerce")
        cleaned.append(d.dropna(subset=["Timestep"]))

    if len(cleaned) == 2:
        df = pd.merge(
            cleaned[0],
            cleaned[1],
            on="Timestep",
            how="outer"
        )
    else:
        df = cleaned[0]

    return df.sort_values("Timestep")


def load_merged_safe(label, files):
    try:
        return load_merged(files)
    except Exception as exc:
        st.error(f"Could not read the files of '{label}': {exc}")
        return None


def get_row(df, timestep):
    """Linha do timestep (ou a mais próxima)."""

    if df is None:
        return None

    row = df[df["Timestep"] == timestep]

    if not row.empty:
        return row.iloc[0]

    if not df.empty:

        idx = (
            df["Timestep"] - timestep
        ).abs().idxmin()

        return df.loc[idx]

    return None


def bin_has_species(df_abs, row_abs, carbon_bin):
    """False se o bin tem quantidade de moléculas ~0 no timestep."""

    if df_abs is None or row_abs is None:
        return True

    if carbon_bin not in df_abs.columns:
        return True

    value = row_abs[carbon_bin]

    return pd.notna(value) and value > ZERO_THRESHOLD


def property_checkboxes(prefix):
    selected = []
    for label, column in PROPERTIES:
        if st.checkbox(label, key=f"{prefix}_prop_{column}"):
            selected.append(column)
    return selected


def bin_checkboxes(prefix):
    selected = []
    for label, column in BINS:
        if st.checkbox(label, key=f"{prefix}_bin_{column}"):
            selected.append(column)
    return selected




if "simulations" not in st.session_state:
    st.session_state["simulations"] = {}

if "active_simulation" not in st.session_state:
    st.session_state["active_simulation"] = None


@st.dialog("Upload your files", width="medium", dismissible=True, icon=":material/upload_file:", on_dismiss="ignore")
def Upload_Files():

    sim_name = st.text_input(
        "Simulation name",
        help="Nome usado para identificar esse grupo de 4 arquivos."
    )

    CARBONS_CSV = st.file_uploader("CH file", type=["csv"], key="ch")
    LOG = st.file_uploader("LAMMPS LOG", key="log")
    CARBON_Fractions_CSV = st.file_uploader("Carbon Fractions file", type=["csv"], key="ca")
    HYDROGEN_FRACTION_CSV = st.file_uploader("Hydrogen Fractions file", type=["csv"], key="ha")

    st.markdown("**Global fraction (optional)**")

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        dlg_global_ca = st.number_input(
            "CA",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            format="%.2f",
            key="dlg_global_ca"
        )

    with gcol2:
        dlg_global_ha = st.number_input(
            "HA",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            format="%.2f",
            key="dlg_global_ha"
        )

    st.markdown("**Time conversion (optional)**")

    sec_text = st.text_input(
        "Seconds per timestep",
        placeholder="ex.: 2.5e-16",
        help="Quanto vale 1 timestep em segundos. Aceita notação científica. "
             "Se vazio, os gráficos continuam em timesteps.",
        key="dlg_sec_per_step"
    )

    if st.button("Submit"):

        seconds_value, sec_error = parse_positive_float(sec_text)

        if not sim_name:
            st.warning("Dê um nome para a simulação antes de enviar.")

        elif sec_error:
            st.warning(sec_error)

        else:
            st.session_state["simulations"][sim_name] = {
                "LOG": LOG,
                "CARBONS_CSV": CARBONS_CSV,
                "CARBON_Fractions_CSV": CARBON_Fractions_CSV,
                "HYDROGEN_FRACTION_CSV": HYDROGEN_FRACTION_CSV,
                "GLOBAL_CA": dlg_global_ca,
                "GLOBAL_HA": dlg_global_ha,
                "SECONDS_PER_STEP": seconds_value,
            }

            st.session_state["active_simulation"] = sim_name

            st.rerun()


if st.button("File Upload"):
    Upload_Files()


sims = st.session_state["simulations"]

EMPTY_FILES = {
    "LOG": None,
    "CARBONS_CSV": None,
    "CARBON_Fractions_CSV": None,
    "HYDROGEN_FRACTION_CSV": None,
    "GLOBAL_CA": None,
    "GLOBAL_HA": None,
    "SECONDS_PER_STEP": None,
}




st.header("Charts for solo simulation Analysis")



if sims:

    sim_names = list(sims.keys())

    if st.session_state["active_simulation"] not in sim_names:
        st.session_state["active_simulation"] = sim_names[0]

    sel_col, rm_col, _ = st.columns([3, 2, 5], vertical_alignment="bottom")

    with sel_col:
        selected_sim = st.selectbox(
            "Active simulation",
            sim_names,
            index=sim_names.index(st.session_state["active_simulation"])
        )

    st.session_state["active_simulation"] = selected_sim

    with rm_col:
        if st.button("Remove active simulation"):
            del sims[selected_sim]
            st.session_state["active_simulation"] = None
            st.rerun()

    current_files = sims[selected_sim]

else:
    st.info("No simulation uploaded yet.")
    current_files = EMPTY_FILES

LOG = current_files.get("LOG")
CARBONS_CSV = current_files.get("CARBONS_CSV")
CARBON_Fractions_CSV = current_files.get("CARBON_Fractions_CSV")
HYDROGEN_FRACTION_CSV = current_files.get("HYDROGEN_FRACTION_CSV")
global_ca = current_files.get("GLOBAL_CA")
global_ha = current_files.get("GLOBAL_HA")
seconds_per_step = current_files.get("SECONDS_PER_STEP")


def timestep_label(timestep):
    """Rótulo 'timestep (tempo)' quando há conversão; senão só o timestep."""
    if seconds_per_step is None:
        return f"{timestep:g}"
    return f"{timestep:g} ({fmt_time(timestep * seconds_per_step)})"


if sims:
    st.caption(f"Active simulation: **{st.session_state['active_simulation']}**")

    info_parts = []
    if seconds_per_step is not None:
        info_parts.append(f"1 timestep = {seconds_per_step:g} s")
    if global_ca is not None:
        info_parts.append(f"CA global = {global_ca:.2%}")
    if global_ha is not None:
        info_parts.append(f"HA global = {global_ha:.2%}")
    if info_parts:
        st.caption(" | ".join(info_parts))




col1, col2 = st.columns([4, 1])

with col2:

    st.write("Select Properties to be shown in graph")

    selected_columns = property_checkboxes("solo")


with col1:

    df = None

    if LOG is not None or CARBONS_CSV is not None:
        df = load_merged_safe(
            st.session_state["active_simulation"],
            current_files
        )

    if df is not None and selected_columns:

        x_col, x_title = add_x_column(df, seconds_per_step)

        fig = go.Figure()

        right_columns = [
            col for col in selected_columns
            if col in CARBON_COLUMNS and col in df.columns
        ]

        left_columns = [
            col for col in selected_columns
            if col not in CARBON_COLUMNS and col in df.columns
        ]

        for column in left_columns:

            df_plot = df[
                [x_col, column]
            ].dropna()

            fig.add_trace(
                go.Scatter(
                    x=df_plot[x_col],
                    y=df_plot[column],
                    name=column,
                    mode="lines"
                )
            )

        for column in right_columns:

            df_plot = df[
                [x_col, column]
            ].dropna()

            fig.add_trace(
                go.Scatter(
                    x=df_plot[x_col],
                    y=df_plot[column],
                    name=column,
                    mode="markers",
                    yaxis="y2"
                )
            )

        if left_columns and right_columns:

            fig.update_layout(
                xaxis=dict(title=x_title),
                yaxis=dict(title="Properties"),
                yaxis2=dict(
                    title="Carbons",
                    overlaying="y",
                    side="right"
                ),
                hovermode="x unified"
            )

        else:

            fig.update_layout(
                xaxis=dict(title=x_title),
                yaxis=dict(title="Properties"),
                hovermode="x unified"
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )




col1, col2, col3 = st.columns([1, 3, 3])




with col1:

    fraction_types = []

    st.write("### Select Fraction types to be taken into analysis")

    Carbon_Fractions = st.checkbox("Carbon Fractions")

    if Carbon_Fractions:
        fraction_types.append("Carbon")

    Hydrogen_Fractions = st.checkbox("Hydrogen Fractions")

    if Hydrogen_Fractions:
        fraction_types.append("Hydrogen")

    st.write("### Select Carbon bins")

    carbon_bins = bin_checkboxes("solo_fraction")

    st.write("### Standard deviation")

    show_std_ca = st.checkbox("Show standard deviation (CA)")
    show_std_ha = st.checkbox("Show standard deviation (HA)")


df_absolute = None
df_ca = None
df_ha = None

try:
    df_absolute = read_csv_ts(CARBONS_CSV)

    if "Carbon" in fraction_types:
        df_ca = read_csv_ts(CARBON_Fractions_CSV)

    if "Hydrogen" in fraction_types:
        df_ha = read_csv_ts(HYDROGEN_FRACTION_CSV)

except Exception as exc:
    st.error(f"Could not read the CSV files: {exc}")




timesteps = []

if df_ca is not None:
    timesteps.extend(df_ca["Timestep"].dropna().unique().tolist())

if df_ha is not None:
    timesteps.extend(df_ha["Timestep"].dropna().unique().tolist())

timesteps = sorted(set(timesteps))

timestep_selected = None



with col2:

    if not carbon_bins:

        st.info("Select at least one carbon bin.")

    elif df_absolute is None:

        st.warning("Upload the CARBONS CSV file.")

    elif not timesteps:

        st.warning("No timestep available.")

    else:

        timestep_selected = st.select_slider(
            "Time",
            options=timesteps,
            value=timesteps[0],
            format_func=timestep_label
        )

        row_absolute = get_row(
            df_absolute,
            timestep_selected
        )

        absolute_values = []

        for carbon_bin in carbon_bins:

            if carbon_bin in df_absolute.columns:
                absolute_values.append(row_absolute[carbon_bin])
            else:
                absolute_values.append(None)

        fig_absolute = go.Figure()

        fig_absolute.add_trace(
            go.Bar(
                x=carbon_bins,
                y=absolute_values,
                name="Carbons"
            )
        )

        fig_absolute.update_layout(
            title=(
                f"Absolute Values - "
                f"Timestep {timestep_label(timestep_selected)}"
            ),
            xaxis_title="Carbon Bins",
            yaxis_title="Number of Species",
            showlegend=False
        )

        st.plotly_chart(
            fig_absolute,
            use_container_width=True
        )



with col3:

    if not fraction_types:

        st.info("Select at least one fraction type.")

    elif not carbon_bins:

        st.info("Select at least one carbon bin.")

    elif not timesteps:

        st.warning("No timestep available.")

    elif timestep_selected is None:

        st.info("Upload the CARBONS CSV to use the time slider.")

    else:


        row_abs_for_filter = get_row(df_absolute, timestep_selected)

        bin_visible = [
            bin_has_species(df_absolute, row_abs_for_filter, b)
            for b in carbon_bins
            ]

        hidden_bins = [
            b for b, v in zip(carbon_bins, bin_visible) if not v
            ]

        if hidden_bins:
            st.caption(
                "Bins omitidos (sem moléculas): "
                + ", ".join(hidden_bins)
            )

        fig_fraction = go.Figure()


        if df_ca is not None:

            row_ca = get_row(df_ca, timestep_selected)

            ca_values = []

            for carbon_bin, visible in zip(carbon_bins, bin_visible):

                if visible and carbon_bin in df_ca.columns:
                    value = row_ca[carbon_bin]
                    ca_values.append(0.0 if pd.isna(value) else value)
                else:
                    ca_values.append(None)

            error_y_ca = None

            if show_std_ca:

                std_cols_ca = [
                    f"{carbon_bin}{STD_SUFFIX}"
                    for carbon_bin in carbon_bins
                ]

                if all(c in df_ca.columns for c in std_cols_ca):

                    error_y_ca = dict(
                        type="data",
                        array=[row_ca[c] for c in std_cols_ca],
                        visible=True,
                        color="white",
                        thickness=1.5,
                        width=6
                    )

                else:
                    st.warning(
                        f"Colunas de desvio padrão não encontradas para CA "
                        f"(esperado sufixo '{STD_SUFFIX}')."
                    )

            fig_fraction.add_trace(
                go.Bar(
                    x=carbon_bins,
                    y=ca_values,
                    name="CA",
                    offsetgroup="Carbon",
                    legendgroup="Carbon",
                    marker_color=CA_COLOR,
                    text=[
                        f"{v:.2%}" if pd.notna(v) else ""
                        for v in ca_values
                    ],
                    textposition="inside",
                    error_y=error_y_ca
                )
            )

            cb_values = [
                1 - value if pd.notna(value) else None
                for value in ca_values
            ]

            fig_fraction.add_trace(
                go.Bar(
                    x=carbon_bins,
                    y=cb_values,
                    name="CB",
                    offsetgroup="Carbon",
                    legendgroup="Carbon",
                    marker_color=CB_COLOR,
                    showlegend=False
                )
            )


        if df_ha is not None:

            row_ha = get_row(df_ha, timestep_selected)

            ha_values = []

            for carbon_bin, visible in zip(carbon_bins, bin_visible):

                if visible and carbon_bin in df_ha.columns:
                    value = row_ha[carbon_bin]
                    ha_values.append(0.0 if pd.isna(value) else value)
                else:
                    ha_values.append(None)

            error_y_ha = None

            if show_std_ha:

                std_cols_ha = [
                    f"{carbon_bin}{STD_SUFFIX}"
                    for carbon_bin in carbon_bins
                ]

                if all(c in df_ha.columns for c in std_cols_ha):

                    error_y_ha = dict(
                        type="data",
                        array=[row_ha[c] for c in std_cols_ha],
                        visible=True,
                        color="white",
                        thickness=1.5,
                        width=6
                    )

                else:
                    st.warning(
                        f"Colunas de desvio padrão não encontradas para HA "
                        f"(esperado sufixo '{STD_SUFFIX}')."
                    )

            fig_fraction.add_trace(
                go.Bar(
                    x=carbon_bins,
                    y=ha_values,
                    name="HA",
                    offsetgroup="Hydrogen",
                    legendgroup="Hydrogen",
                    marker_color=HA_COLOR,
                    text=[
                        f"{v:.2%}" if pd.notna(v) else ""
                        for v in ha_values
                    ],
                    textposition="inside",
                    error_y=error_y_ha
                )
            )

            hb_values = [
                1 - value if pd.notna(value) else None
                for value in ha_values
            ]

            fig_fraction.add_trace(
                go.Bar(
                    x=carbon_bins,
                    y=hb_values,
                    name="HB",
                    offsetgroup="Hydrogen",
                    legendgroup="Hydrogen",
                    marker_color=HB_COLOR,
                    showlegend=False
                )
            )


        if df_ca is not None and global_ca is not None:

            fig_fraction.add_hline(
                y=global_ca,
                line_color=CA_COLOR,
                line_width=2,
                line_dash="dash",
                annotation_text=f"CA global: {global_ca:.2%}",
                annotation_position="top left"
            )

        if df_ha is not None and global_ha is not None:

            fig_fraction.add_hline(
                y=global_ha,
                line_color=HA_COLOR,
                line_width=2,
                line_dash="dash",
                annotation_text=f"HA global: {global_ha:.2%}",
                annotation_position="bottom left"
            )

        fig_fraction.update_layout(
            title=(
                f"Fractions - "
                f"Timestep {timestep_label(timestep_selected)}"
            ),
            xaxis_title="Carbon Bins",
            yaxis_title="Fraction",
            yaxis=dict(range=[0, 1]),
            barmode="stack",
            legend=dict(title="Fractions")
        )

        st.plotly_chart(
            fig_fraction,
            use_container_width=True
        )



st.divider()
st.header("Charts for multi simulation Analysis")

all_sim_names = list(sims.keys())


def sim_color(name):
    return SIM_COLORS[all_sim_names.index(name) % len(SIM_COLORS)]


multi_selected = []

if not all_sim_names:

    st.info("Upload at least one simulation to use the multi simulation analysis.")

else:

    # remove da seleção simulações que foram apagadas
    st.session_state["multi_sims"] = [
        s for s in st.session_state.get("multi_sims", [])
        if s in all_sim_names
    ]

    multi_selected = st.multiselect(
        "Select simulations to compare",
        all_sim_names,
        key="multi_sims"
    )

    if not multi_selected:
        st.info("Select at least one simulation.")


if multi_selected:

    st.subheader("Properties over time")

    pcol1, pcol2 = st.columns([4, 1])

    with pcol2:

        x_mode = st.radio(
            "X axis",
            ["Timestep", "Time"],
            key="multi_x_mode",
            help="'Time' exige 'segundos por timestep' em todas as simulações selecionadas."
        )

        st.write("Select Properties to be shown in graph")

        m_selected_columns = property_checkboxes("multi")

    with pcol1:

        if m_selected_columns:

            merged = {}

            for name in multi_selected:
                df_m = load_merged_safe(name, sims[name])
                if df_m is not None:
                    merged[name] = df_m

            no_data = [n for n in multi_selected if n not in merged]

            if no_data:
                st.caption(
                    "Sem LOG/CH para: " + ", ".join(no_data)
                )

            sec_by_sim = {
                n: sims[n].get("SECONDS_PER_STEP") for n in merged
            }

            use_time = x_mode == "Time"

            if use_time and any(v is None for v in sec_by_sim.values()):
                missing_sec = [n for n, v in sec_by_sim.items() if v is None]
                st.warning(
                    "Sem 'segundos por timestep' em: "
                    + ", ".join(missing_sec)
                    + ". Usando Timestep no eixo x."
                )
                use_time = False

            if use_time and merged:

                max_time = 0

                for n, df_m in merged.items():
                    max_step = df_m["Timestep"].max()
                    if pd.notna(max_step):
                        max_time = max(max_time, max_step * sec_by_sim[n])

                unit_factor, unit_label = pick_unit(
                    max_time if max_time > 0 else min(sec_by_sim.values())
                )

                x_col = "Time"
                x_title = f"Time ({unit_label})"

                for n, df_m in merged.items():
                    df_m["Time"] = df_m["Timestep"] * sec_by_sim[n] / unit_factor

            else:
                x_col = "Timestep"
                x_title = "Timestep"

            for column in m_selected_columns:

                label = PROPERTY_LABELS.get(column, column)

                fig_m = go.Figure()
                plotted = False

                for n, df_m in merged.items():

                    if column not in df_m.columns:
                        continue

                    df_plot = df_m[[x_col, column]].dropna()

                    if df_plot.empty:
                        continue

                    fig_m.add_trace(
                        go.Scatter(
                            x=df_plot[x_col],
                            y=df_plot[column],
                            name=n,
                            mode=(
                                "lines+markers"
                                if column in CARBON_COLUMNS
                                else "lines"
                            ),
                            line=dict(color=sim_color(n)),
                            marker=dict(color=sim_color(n))
                        )
                    )

                    plotted = True

                if not plotted:
                    st.caption(
                        f"Sem dados de {label} nas simulações selecionadas."
                    )
                    continue

                fig_m.update_layout(
                    title=label,
                    xaxis_title=x_title,
                    yaxis_title=column,
                    hovermode="x unified",
                    legend=dict(title="Simulations")
                )

                st.plotly_chart(
                    fig_m,
                    use_container_width=True,
                    key=f"multi_line_{column}"
                )



    st.subheader("Bar charts at a selected timestep")

    mcol1, mcol2, mcol3 = st.columns([1, 3, 3])

    with mcol1:

        m_fraction_types = []

        st.write("### Select Fraction types to be taken into analysis")

        if st.checkbox("Carbon Fractions", key="multi_carbon_fractions"):
            m_fraction_types.append("Carbon")

        if st.checkbox("Hydrogen Fractions", key="multi_hydrogen_fractions"):
            m_fraction_types.append("Hydrogen")

        st.write("### Select Carbon bins")

        m_bins = bin_checkboxes("multi")

        st.write("### Standard deviation")

        m_show_std = st.checkbox(
            "Show standard deviation (CA/HA)",
            key="multi_show_std"
        )


    multi_data = {}

    for name in multi_selected:

        files = sims[name]

        try:
            multi_data[name] = {
                "abs": read_csv_ts(files.get("CARBONS_CSV")),
                "Carbon": (
                    read_csv_ts(files.get("CARBON_Fractions_CSV"))
                    if "Carbon" in m_fraction_types else None
                ),
                "Hydrogen": (
                    read_csv_ts(files.get("HYDROGEN_FRACTION_CSV"))
                    if "Hydrogen" in m_fraction_types else None
                ),
            }
        except Exception as exc:
            st.error(f"Could not read the CSV files of '{name}': {exc}")
            multi_data[name] = {"abs": None, "Carbon": None, "Hydrogen": None}

    m_timesteps = set()

    for data in multi_data.values():
        for df_x in data.values():
            if df_x is not None and "Timestep" in df_x.columns:
                m_timesteps.update(df_x["Timestep"].dropna().unique().tolist())

    m_timesteps = sorted(m_timesteps)


    secs = {sims[n].get("SECONDS_PER_STEP") for n in multi_selected}
    multi_sec = secs.pop() if len(secs) == 1 and None not in secs else None

    def multi_timestep_label(timestep):
        if multi_sec is None:
            return f"{timestep:g}"
        return f"{timestep:g} ({fmt_time(timestep * multi_sec)})"

    m_timestep = None


    with mcol2:

        if not m_bins:

            st.info("Select at least one carbon bin.")

        elif not m_timesteps:

            st.warning("No timestep available (upload the CSV files).")

        else:

            m_timestep = st.select_slider(
                "Time (compare simulations)",
                options=m_timesteps,
                value=m_timesteps[0],
                format_func=multi_timestep_label
            )

            st.caption("Each simulation uses its nearest available timestep.")

            fig_abs_m = go.Figure()

            no_abs = []

            for name in multi_selected:

                df_abs = multi_data[name].get("abs")
                row_abs = get_row(df_abs, m_timestep)

                if row_abs is None:
                    no_abs.append(name)
                    continue

                values = [
                    row_abs[b] if b in df_abs.columns else None
                    for b in m_bins
                ]

                fig_abs_m.add_trace(
                    go.Bar(
                        x=m_bins,
                        y=values,
                        name=name,
                        marker_color=sim_color(name)
                    )
                )

            if no_abs:
                st.caption("Sem CARBONS CSV para: " + ", ".join(no_abs))

            fig_abs_m.update_layout(
                title=f"Absolute Values - Timestep {multi_timestep_label(m_timestep)}",
                xaxis_title="Carbon Bins",
                yaxis_title="Number of Species",
                barmode="group",
                legend=dict(title="Simulations")
            )

            st.plotly_chart(
                fig_abs_m,
                use_container_width=True,
                key="multi_absolute_chart"
            )


    with mcol3:

        if not m_fraction_types:

            st.info("Select at least one fraction type.")

        elif not m_bins:

            st.info("Select at least one carbon bin.")

        elif m_timestep is None:

            st.info("No timestep available.")

        else:

            fig_frac_m = go.Figure()

            hidden_report = {}
            missing_report = []
            std_missing = []

            for name in multi_selected:

                color = sim_color(name)

                df_abs = multi_data[name].get("abs")
                row_abs = get_row(df_abs, m_timestep)

                visible = [
                    bin_has_species(df_abs, row_abs, b) for b in m_bins
                ]

                hidden = [
                    b for b, v in zip(m_bins, visible) if not v
                ]

                if hidden:
                    hidden_report[name] = hidden

                for ftype in m_fraction_types:

                    letter = "C" if ftype == "Carbon" else "H"
                    frac_df = multi_data[name].get(ftype)

                    if frac_df is None:
                        missing_report.append(f"{name} ({ftype})")
                        continue

                    row = get_row(frac_df, m_timestep)

                    if row is None:
                        continue

                    values = [
                        (0.0 if pd.isna(row[b]) else row[b])
                        if (vis and b in frac_df.columns) else None
                        for b, vis in zip(m_bins, visible)
                    ]

                    error_y = None

                    if m_show_std:

                        std_cols = [f"{b}{STD_SUFFIX}" for b in m_bins]

                        if all(c in frac_df.columns for c in std_cols):
                            error_y = dict(
                                type="data",
                                array=[row[c] for c in std_cols],
                                visible=True,
                                color="white",
                                thickness=1.5,
                                width=6
                            )
                        else:
                            std_missing.append(f"{name} ({ftype})")

                    group = f"{name}|{ftype}"
                    pattern = "" if ftype == "Carbon" else "/"

                    fig_frac_m.add_trace(
                        go.Bar(
                            x=m_bins,
                            y=values,
                            name=f"{name} · {letter}A",
                            offsetgroup=group,
                            legendgroup=group,
                            marker=dict(
                                color=color,
                                pattern=dict(shape=pattern)
                            ),
                            text=[
                                f"{v:.2%}" if pd.notna(v) else ""
                                for v in values
                            ],
                            textposition="inside",
                            error_y=error_y
                        )
                    )

                    complement = [
                        1 - v if pd.notna(v) else None for v in values
                    ]

                    fig_frac_m.add_trace(
                        go.Bar(
                            x=m_bins,
                            y=complement,
                            name=f"{name} · {letter}B",
                            offsetgroup=group,
                            legendgroup=group,
                            marker=dict(
                                color=color,
                                opacity=0.3,
                                pattern=dict(shape=pattern)
                            ),
                            showlegend=False
                        )
                    )

            if hidden_report:
                st.caption(
                    "Bins omitidos (sem moléculas): "
                    + "; ".join(
                        f"{n}: {', '.join(bs)}"
                        for n, bs in hidden_report.items()
                    )
                )

            if missing_report:
                st.caption(
                    "Sem arquivo de fração para: " + ", ".join(missing_report)
                )

            if std_missing:
                st.warning(
                    f"Colunas de desvio padrão não encontradas "
                    f"(esperado sufixo '{STD_SUFFIX}') em: "
                    + ", ".join(std_missing)
                )

            fig_frac_m.update_layout(
                title=f"Fractions - Timestep {multi_timestep_label(m_timestep)}",
                xaxis_title="Carbon Bins",
                yaxis_title="Fraction",
                yaxis=dict(range=[0, 1]),
                barmode="stack",
                legend=dict(title="Fractions")
            )

            st.plotly_chart(
                fig_frac_m,
                use_container_width=True,
                key="multi_fraction_chart"
            )
