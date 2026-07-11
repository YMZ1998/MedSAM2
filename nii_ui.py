from html import escape


def product_css():
    return """
    <style>
    :root {
        --bg: #f5f8fc;
        --panel: #ffffff;
        --ink: #16243a;
        --muted: #65748b;
        --line: #dce5f0;
        --blue: #2f6fe4;
        --blue-soft: #eaf2ff;
        --cyan: #10b9bd;
        --green: #17894a;
        --orange: #f29a38;
        --viewer: #050a10;
    }
    header[data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], #MainMenu { display: none; }
    .stApp { background: var(--bg); color: var(--ink); }
    .block-container { max-width: none; padding: 16px 24px 20px; }
    [data-testid="stSidebar"] {
        width: 310px; min-width: 310px;
        background: #f8faff; border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] section { padding-top: 8px; }
    [data-testid="stSidebarCollapseButton"], button[title="Close sidebar"],
    button[data-testid="baseButton-header"], button[data-testid="baseButton-headerNoPadding"] {
        display: none;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .brand { display: flex; align-items: center; gap: 12px; margin: 0 0 18px; }
    .brand-logo {
        display: grid; place-items: center; width: 50px; height: 50px;
        border-radius: 8px; background: #2f6fe4; color: #fff;
        font-size: 30px; font-weight: 900; box-shadow: 0 10px 24px rgba(47,111,228,.24);
    }
    .brand-name { color: var(--ink); font-size: 1.48rem; line-height: 1.05; font-weight: 820; }
    .brand-sub { color: var(--blue); font-weight: 780; }
    .section-label {
        display: flex; align-items: center; gap: 10px; margin: 14px 0 8px;
        color: #46608a; font-size: .71rem; font-weight: 800; text-transform: uppercase;
    }
    .section-label:after { content: ""; flex: 1; height: 1px; background: var(--line); }
    .file-card {
        display: grid; grid-template-columns: 30px 1fr 22px; align-items: center; gap: 9px;
        border: 1px solid var(--line); background: #fff; border-radius: 7px;
        padding: 11px 12px; margin: 8px 0 2px;
    }
    .file-icon { color: #7385a3; font-size: 1.25rem; }
    .file-name { color: var(--ink); font-size: .88rem; font-weight: 760; overflow-wrap: anywhere; }
    .file-size { color: var(--muted); font-size: .76rem; margin-top: 2px; }
    .file-ok { color: var(--green); font-size: 1.05rem; font-weight: 900; }
    .runtime {
        border: 1px solid #bce1cc; background: #e6f5ec; color: #12653a;
        border-radius: 7px; padding: 10px 12px; margin: 8px 0 10px;
        font-size: .82rem; line-height: 1.45; font-weight: 650;
    }
    .runtime.offline { border-color: #efc2c8; background: #fff0f2; color: #9f2938; }
    .status-grid {
        display: grid; grid-template-columns: .9fr 1fr 1fr .9fr .8fr .7fr 1.45fr;
        background: #fff; border: 1px solid var(--line); border-radius: 7px;
        margin: 0 0 12px; overflow: hidden;
    }
    .status-item { min-width: 0; padding: 13px 17px; border-right: 1px solid var(--line); }
    .status-item:last-child { border-right: 0; }
    .status-label { color: #576881; font-size: .69rem; font-weight: 740; margin-bottom: 7px; }
    .status-value { color: var(--ink); font-size: .91rem; font-weight: 780; overflow-wrap: anywhere; }
    .status-detail { color: var(--muted); font-size: .72rem; margin-top: 4px; }
    .viewer-title { display: flex; align-items: center; gap: 10px; margin: 2px 0 10px; }
    .viewer-title h2 { margin: 0; color: #31518c; font-size: .98rem; font-weight: 780; }
    .step {
        display: grid; place-items: center; width: 31px; height: 31px; border-radius: 50%;
        background: var(--blue-soft); color: var(--blue); font-weight: 850;
    }
    .viewer-placeholder {
        min-height: 510px; display: grid; place-items: center; border: 1px solid #111a25;
        border-radius: 7px; background: radial-gradient(circle at 50% 46%, #202934 0, #0c131c 31%, #03070c 70%);
        color: #91a0b2; font-weight: 700;
    }
    .note {
        border-left: 3px solid var(--blue); border-radius: 6px; padding: 9px 11px;
        margin: 9px 0 0; background: #edf5ff; color: #24569e; font-size: .81rem;
    }
    .note.warn { border-left-color: var(--orange); background: #fff6ed; color: #8a481f; }
    .result-card, .quick-card {
        min-height: 92px; border: 1px solid var(--line); border-radius: 7px;
        background: #fff; padding: 12px 14px; margin-top: 10px;
    }
    .card-title { color: #52647e; font-size: .78rem; font-weight: 780; margin-bottom: 11px; }
    .result-grid { display: grid; grid-template-columns: repeat(3, 1fr); }
    .result-item { padding-right: 10px; border-right: 1px solid var(--line); }
    .result-item + .result-item { padding-left: 10px; }
    .result-item:last-child { border-right: 0; }
    .result-label { color: var(--muted); font-size: .66rem; }
    .result-value { color: var(--ink); font-size: .91rem; font-weight: 800; margin-top: 5px; white-space: nowrap; }
    .empty-thumbs {
        display: grid; grid-template-columns: repeat(7, 1fr); gap: 7px; margin-top: 9px;
        padding: 9px; border: 1px solid #111a25; border-radius: 7px; background: var(--viewer);
    }
    .empty-thumb { height: 70px; border: 1px solid #172333; border-radius: 5px; background: #0b121b; }
    .info-strip {
        margin-top: 14px; padding: 12px 15px; border: 1px solid #d4e5ff;
        border-radius: 7px; background: #eaf3ff; color: #245fc5; font-size: .84rem;
    }
    div[data-testid="stImage"] img, iframe[title="streamlit_drawable_canvas.st_canvas"] {
        border: 1px solid #111a25; border-radius: 7px; background: var(--viewer);
        box-shadow: 0 12px 28px rgba(16,31,48,.12);
    }
    div[data-testid="stImage"] { margin-bottom: 0; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { min-height: 108px; padding: 12px; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] { margin-bottom: -8px; }
    .stButton > button, .stDownloadButton > button {
        width: 100%; min-height: 2.35rem; border-radius: 6px; font-weight: 730;
    }
    .stButton > button[kind="primary"] { background: var(--blue); border-color: var(--blue); }
    [data-testid="stSlider"] { margin-top: -8px; }
    .thumb-row [data-testid="stImage"] img { border-radius: 5px 5px 0 0; box-shadow: none; }
    @media (max-width: 1180px) {
        .status-grid { grid-template-columns: repeat(4, minmax(0,1fr)); }
        .status-item { border-bottom: 1px solid var(--line); }
        div[data-testid="stHorizontalBlock"]:has(.viewer-placeholder),
        div[data-testid="stHorizontalBlock"]:has(iframe[title="streamlit_drawable_canvas.st_canvas"]) {
            flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"]:has(.viewer-placeholder) > div[data-testid="column"],
        div[data-testid="stHorizontalBlock"]:has(iframe[title="streamlit_drawable_canvas.st_canvas"]) > div[data-testid="column"] {
            min-width: 100%; flex: 1 1 100%;
        }
    }
    @media (max-width: 760px) {
        .block-container { padding: 12px; }
        .status-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
        .viewer-placeholder { min-height: 340px; }
    }
    </style>
    """


def render_brand():
    return """
    <div class="brand">
      <div class="brand-logo">&#10010;</div>
      <div class="brand-name">MedSAM2<br><span class="brand-sub">NIfTI Studio</span></div>
    </div>
    """


def render_section_label(text):
    return f'<div class="section-label">{escape(text)}</div>'


def render_file_card(name, size_bytes):
    if not name:
        return ""
    return f"""
    <div class="file-card">
      <div class="file-icon">&#9635;</div>
      <div><div class="file-name">{escape(name)}</div><div class="file-size">{size_bytes / 1048576:.1f} MB</div></div>
      <div class="file-ok">&#10003;</div>
    </div>
    """


def render_status_grid(items):
    cells = []
    for label, value, detail in items:
        cells.append(
            '<div class="status-item">'
            f'<div class="status-label">{escape(str(label))}</div>'
            f'<div class="status-value">{escape(str(value))}</div>'
            f'<div class="status-detail">{escape(str(detail))}</div></div>'
        )
    return f'<div class="status-grid">{"".join(cells)}</div>'


def render_viewer_title(step, title):
    return f'<div class="viewer-title"><div class="step">{int(step)}</div><h2>{escape(title)}</h2></div>'


def render_mask_summary(summary):
    return f"""
    <div class="result-card">
      <div class="card-title">Mask statistics</div>
      <div class="result-grid">
        <div class="result-item"><div class="result-label">Volume</div><div class="result-value">{summary['volume_cm3']:.3f} cm3</div></div>
        <div class="result-item"><div class="result-label">Voxels</div><div class="result-value">{summary['voxels']:,}</div></div>
        <div class="result-item"><div class="result-label">Coverage</div><div class="result-value">{summary['coverage']:.2f}%</div></div>
      </div>
    </div>
    """


def render_empty_thumbnails():
    return '<div class="empty-thumbs">' + '<div class="empty-thumb"></div>' * 7 + '</div>'


def render_placeholder_status(model_name, device):
    return render_status_grid(
        [
            ("Case", "No study", "Waiting"),
            ("Dimensions", "- x - x -", "D x H x W"),
            ("Spacing", "- x - x -", "mm"),
            ("Voxel Count", "-", "voxels"),
            ("Slice", "- / -", "Load volume"),
            ("Device", device.upper(), "Ready"),
            ("Model", model_name, "Ready"),
        ]
    )
