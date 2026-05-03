import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

import db
from extractors.llm import extract_companies
from extractors.substack import get_substack_content
from extractors.youtube import get_transcript
from scanner.wma_scanner import get_cached_results, run_scan

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

_THEME_LABELS = {
    "ai_infrastructure": "AI Infrastructure",
    "edge_inference": "Edge Inference",
    "biotech_healthtech": "Biotech / HealthTech",
    "ai_applications": "AI Applications",
    "ai_other": "AI — Other",
    "not_ai": "Non-AI",
    "unknown": "Unknown",
}

_SENTIMENT_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _companies_df(companies: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(companies)
    rename = {
        "company_name": "Company",
        "ticker": "Ticker",
        "sub_theme": "Theme",
        "sentiment": "Sentiment",
        "context": "Context",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Theme" in df.columns:
        df["Theme"] = df["Theme"].map(lambda x: _THEME_LABELS.get(x, x))
    return df


@st.cache_data(ttl=900)
def _fetch_live_wma(tickers: tuple) -> dict:
    """Fetch current price + 200-week MA for a tuple of tickers. Cached 15 min."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers),
            period="5y",
            interval="1wk",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return {}

    if raw.empty:
        return {}

    if "Close" in raw.columns:
        close = raw["Close"]
    else:
        try:
            close = raw.xs("Close", axis=1, level=0)
        except Exception:
            return {}

    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    result = {}
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if len(series) < 2:
            continue
        wma = series.rolling(200, min_periods=20).mean().iloc[-1]
        current = series.iloc[-1]
        if pd.isna(current):
            continue
        if pd.isna(wma) or wma == 0:
            result[ticker] = {"price": round(float(current), 2), "wma": None, "pct": None}
        else:
            pct = ((current - wma) / wma) * 100
            result[ticker] = {
                "price": round(float(current), 2),
                "wma": round(float(wma), 2),
                "pct": round(float(pct), 2),
            }
    return result


def _render_price_chart(ticker: str):
    try:
        raw = yf.download(
            ticker, period="5y", interval="1wk", auto_adjust=True, progress=False
        )
        if raw.empty:
            st.warning(f"No price data for {ticker}")
            return

        close = raw["Close"].squeeze()
        wma = close.rolling(200, min_periods=20).mean()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                name="Price",
                line={"color": "#1f77b4", "width": 1.5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wma.index,
                y=wma.values,
                name="200-week MA",
                line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            )
        )
        fig.update_layout(
            title=f"{ticker} — Weekly Price vs 200 WMA",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            margin={"t": 50, "b": 40, "l": 40, "r": 20},
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")


# ── Sidebar nav ───────────────────────────────────────────────────────────────

st.sidebar.title("📈 Market Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Sources", "Intelligence", "200 WMA Scanner", "Watchlist"],
    label_visibility="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SOURCES
# ─────────────────────────────────────────────────────────────────────────────

if page == "Sources":
    st.title("Sources")
    st.caption(
        "Add YouTube videos, Substack articles, or paste raw text. "
        "Claude extracts the companies and AI themes."
    )

    add_tab, history_tab = st.tabs(["➕ Add Source", "📋 History"])

    with add_tab:
        source_type = st.selectbox(
            "Source type",
            ["YouTube Video", "Substack Article", "Paste Text"],
        )

        if source_type == "YouTube Video":
            url = st.text_input(
                "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )
            if st.button("Extract", type="primary", disabled=not url):
                with st.status("Processing YouTube video…", expanded=True) as status:
                    st.write("Fetching transcript…")
                    try:
                        content, video_id = get_transcript(url)
                        st.write(f"✅ Transcript fetched ({len(content):,} chars)")
                        st.write("Extracting companies with Claude…")
                        companies = extract_companies(content)
                        source_id = db.save_source(
                            url, "youtube", f"YouTube: {video_id}", content
                        )
                        db.save_extracted_companies(source_id, companies)
                        status.update(
                            label=f"Done — {len(companies)} companies found",
                            state="complete",
                        )
                        if companies:
                            st.dataframe(
                                _companies_df(companies),
                                use_container_width=True,
                                hide_index=True,
                            )
                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(str(e))

        elif source_type == "Substack Article":
            url = st.text_input(
                "Substack URL",
                placeholder="https://newsletter.substack.com/p/...",
            )
            if st.button("Extract", type="primary", disabled=not url):
                with st.status("Processing Substack article…", expanded=True) as status:
                    st.write("Fetching article…")
                    try:
                        content = get_substack_content(url)
                        st.write(f"✅ Article fetched ({len(content):,} chars)")
                        st.write("Extracting companies with Claude…")
                        companies = extract_companies(content)
                        source_id = db.save_source(
                            url,
                            "substack",
                            f"Substack: {url.split('/')[-1][:60]}",
                            content,
                        )
                        db.save_extracted_companies(source_id, companies)
                        status.update(
                            label=f"Done — {len(companies)} companies found",
                            state="complete",
                        )
                        if companies:
                            st.dataframe(
                                _companies_df(companies),
                                use_container_width=True,
                                hide_index=True,
                            )
                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(str(e))

        else:  # Paste Text
            title = st.text_input(
                "Title (optional)",
                placeholder="e.g. Tweet thread from @investor",
            )
            content = st.text_area("Paste content", height=280)
            if st.button("Extract", type="primary", disabled=not content):
                with st.status("Extracting companies…", expanded=True) as status:
                    try:
                        companies = extract_companies(content)
                        source_id = db.save_source(
                            "", "paste", title or "Pasted content", content
                        )
                        db.save_extracted_companies(source_id, companies)
                        status.update(
                            label=f"Done — {len(companies)} companies found",
                            state="complete",
                        )
                        if companies:
                            st.dataframe(
                                _companies_df(companies),
                                use_container_width=True,
                                hide_index=True,
                            )
                    except Exception as e:
                        status.update(label="Error", state="error")
                        st.error(str(e))

    with history_tab:
        sources = db.get_all_sources()
        if not sources:
            st.info("No sources yet. Add one above.")
        else:
            for src in sources:
                companies = db.get_companies_by_source(src["id"])
                label = (
                    f"**{src['title']}** — {src['type'].upper()} "
                    f"— {src['created_at'][:10]} — {len(companies)} companies"
                )
                with st.expander(label):
                    if src["url"]:
                        st.markdown(f"[Open source]({src['url']})")
                    if companies:
                        st.dataframe(
                            _companies_df(companies),
                            use_container_width=True,
                            hide_index=True,
                        )
                    if st.button("Delete source", key=f"del_{src['id']}"):
                        db.delete_source(src["id"])
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Intelligence":
    st.title("Intelligence")
    st.caption(
        "All companies extracted from your sources, organized by AI sub-theme."
    )

    all_companies = db.get_all_extracted_companies()
    if not all_companies:
        st.info("No companies extracted yet. Add sources first.")
    else:
        df = pd.DataFrame(all_companies)
        df["sub_theme_label"] = df["sub_theme"].map(
            lambda x: _THEME_LABELS.get(x, x)
        )

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique Tickers", df["ticker"].nunique())
        c2.metric("Sources", df["source_id"].nunique())
        c3.metric("Bullish Mentions", int((df["sentiment"] == "bullish").sum()))
        c4.metric("Bearish Mentions", int((df["sentiment"] == "bearish").sum()))

        st.markdown("---")

        themes = ["All"] + sorted(df["sub_theme_label"].unique().tolist())
        selected_theme = st.selectbox("Filter by theme", themes)

        filtered = (
            df
            if selected_theme == "All"
            else df[df["sub_theme_label"] == selected_theme]
        )

        for theme_key, group in filtered.groupby("sub_theme_label"):
            unique_tickers = group["ticker"].unique()
            st.subheader(f"{theme_key}  ·  {len(unique_tickers)} companies")

            for ticker, ticker_group in group.groupby("ticker"):
                company_name = ticker_group["company_name"].iloc[0]
                mentions = len(ticker_group)
                bullish = int((ticker_group["sentiment"] == "bullish").sum())
                bearish = int((ticker_group["sentiment"] == "bearish").sum())

                col1, col2 = st.columns([5, 1])
                with col1:
                    header = (
                        f"**{ticker}** — {company_name} — {mentions} mention(s)"
                        + (f"  🟢 {bullish}" if bullish else "")
                        + (f"  🔴 {bearish}" if bearish else "")
                    )
                    with st.expander(header):
                        for _, row in ticker_group.iterrows():
                            icon = _SENTIMENT_EMOJI.get(row["sentiment"], "")
                            st.markdown(
                                f"{icon} **{row['source_title']}** ({row['created_at'][:10]})"
                            )
                            if row["context"]:
                                st.caption(row["context"])
                with col2:
                    if st.button(
                        "＋ Watch",
                        key=f"intel_add_{ticker}",
                        help=f"Add {ticker} to watchlist",
                    ):
                        themes_list = ticker_group["sub_theme"].unique().tolist()
                        db.add_to_watchlist(ticker, company_name, themes_list, "content")
                        st.toast(f"{ticker} added to watchlist")

            st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 200 WMA SCANNER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "200 WMA Scanner":
    st.title("200 WMA Scanner")
    st.caption(
        "S&P 500 + Russell 2000 companies trading below their 200-week moving average. "
        "A full scan takes ~5–10 min and results are cached."
    )

    last_updated = db.get_wma_last_updated()
    if last_updated:
        st.info(f"Last scan: {last_updated[:16]} UTC")
    else:
        st.warning("No scan data yet — run a scan below.")

    scan_col, toggle_col = st.columns([2, 1])
    with scan_col:
        if st.button("🔄 Run Full Scan", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _cb(pct: float, msg: str):
                progress_bar.progress(min(pct, 1.0))
                status_text.text(msg)

            with st.spinner("Scanning S&P 500 + Russell 2000…"):
                total = run_scan(progress_callback=_cb)

            progress_bar.progress(1.0)
            status_text.text(f"Complete — {total} tickers scanned")
            st.success("Scan complete!")
            st.rerun()

    with toggle_col:
        show_above = st.checkbox("Include above 200 WMA", value=False)

    results = get_cached_results(below_only=not show_above)

    if not results:
        st.info("No results yet. Run a scan to populate.")
    else:
        df = pd.DataFrame(results)

        f1, f2, f3 = st.columns(3)
        with f1:
            index_filter = st.multiselect(
                "Index",
                ["sp500", "russell2000"],
                default=["sp500", "russell2000"],
            )
        with f2:
            max_pct = st.slider(
                "Max % from 200 WMA",
                min_value=-100,
                max_value=0,
                value=-5,
                help="Negative = below the 200 WMA. -5 means at least 5% below.",
            )
        with f3:
            sort_by = st.selectbox(
                "Sort", ["% from WMA (worst first)", "Ticker A–Z"]
            )

        filtered = df[df["index_name"].isin(index_filter)]
        filtered = filtered[filtered["pct_from_wma"] <= max_pct]

        if sort_by == "% from WMA (worst first)":
            filtered = filtered.sort_values("pct_from_wma")
        else:
            filtered = filtered.sort_values("ticker")

        st.markdown(f"**{len(filtered):,} companies** match the current filters")

        disp = filtered[
            ["ticker", "company_name", "index_name", "current_price", "wma_200", "pct_from_wma"]
        ].copy()
        disp.columns = ["Ticker", "Company", "Index", "Price", "200 WMA", "% from WMA"]
        disp["% from WMA"] = disp["% from WMA"].map(lambda x: f"{x:+.1f}%")
        disp["Price"] = disp["Price"].map(lambda x: f"${x:,.2f}")
        disp["200 WMA"] = disp["200 WMA"].map(lambda x: f"${x:,.2f}")

        selection = st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )

        selected_rows = selection.get("selection", {}).get("rows", [])
        if selected_rows:
            sel_df = filtered.iloc[selected_rows]
            if st.button(
                f"＋ Add {len(selected_rows)} to Watchlist", type="primary"
            ):
                for _, row in sel_df.iterrows():
                    db.add_to_watchlist(
                        row["ticker"],
                        row["company_name"],
                        [],
                        "wma_screen",
                    )
                st.toast(f"{len(selected_rows)} companies added to watchlist")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Watchlist":
    st.title("Watchlist")
    st.caption(
        "Your tracked companies. Add them from Intelligence or the WMA Scanner."
    )

    watchlist = db.get_watchlist()

    if not watchlist:
        st.info("Your watchlist is empty. Add companies from Intelligence or the Scanner.")
    else:
        tickers_tuple = tuple(w["ticker"] for w in watchlist)
        live_data = _fetch_live_wma(tickers_tuple)

        rows = []
        for w in watchlist:
            t = w["ticker"]
            ld = live_data.get(t, {})
            rows.append(
                {
                    "Ticker": t,
                    "Company": w["company_name"],
                    "Themes": ", ".join(
                        _THEME_LABELS.get(th, th) for th in (w["sub_themes"] or [])
                    ),
                    "Source": w["source_type"],
                    "Price": ld.get("price"),
                    "200 WMA": ld.get("wma"),
                    "% from WMA": ld.get("pct"),
                    "Added": w["added_at"][:10],
                }
            )

        df = pd.DataFrame(rows)

        wl_filter = st.selectbox(
            "Filter",
            ["All", "Below 200 WMA", "Above 200 WMA", "From content", "From WMA screen"],
        )

        filtered_wl = df.copy()
        if wl_filter == "Below 200 WMA":
            filtered_wl = filtered_wl[
                filtered_wl["% from WMA"].apply(lambda x: x is not None and x < 0)
            ]
        elif wl_filter == "Above 200 WMA":
            filtered_wl = filtered_wl[
                filtered_wl["% from WMA"].apply(lambda x: x is not None and x >= 0)
            ]
        elif wl_filter == "From content":
            filtered_wl = filtered_wl[filtered_wl["Source"].isin(["content", "both"])]
        elif wl_filter == "From WMA screen":
            filtered_wl = filtered_wl[filtered_wl["Source"].isin(["wma_screen", "both"])]

        disp = filtered_wl.copy()
        disp["Price"] = disp["Price"].apply(
            lambda x: f"${x:,.2f}" if x is not None else "—"
        )
        disp["200 WMA"] = disp["200 WMA"].apply(
            lambda x: f"${x:,.2f}" if x is not None else "—"
        )
        disp["% from WMA"] = disp["% from WMA"].apply(
            lambda x: f"{x:+.1f}%" if x is not None else "—"
        )

        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown("---")

        selected_ticker = st.selectbox(
            "Select a company for chart + notes",
            [w["ticker"] for w in watchlist],
        )

        if selected_ticker:
            wl_item = next(
                (w for w in watchlist if w["ticker"] == selected_ticker), None
            )
            if wl_item:
                chart_col, info_col = st.columns([3, 1])
                with chart_col:
                    _render_price_chart(selected_ticker)
                with info_col:
                    st.markdown(f"### {selected_ticker}")
                    st.caption(wl_item["company_name"])

                    ld = live_data.get(selected_ticker, {})
                    if ld.get("price"):
                        delta_str = (
                            f"{ld['pct']:+.1f}% vs 200 WMA"
                            if ld.get("pct") is not None
                            else None
                        )
                        st.metric("Price", f"${ld['price']:,.2f}", delta=delta_str)

                    if wl_item["sub_themes"]:
                        st.markdown(
                            "**Themes:** "
                            + ", ".join(
                                _THEME_LABELS.get(t, t) for t in wl_item["sub_themes"]
                            )
                        )

                    st.markdown(f"**Source:** {wl_item['source_type']}")

                    notes = st.text_area(
                        "Notes",
                        value=wl_item.get("notes") or "",
                        height=150,
                        key=f"notes_{selected_ticker}",
                    )
                    if st.button("Save notes", key=f"save_{selected_ticker}"):
                        db.update_watchlist_notes(selected_ticker, notes)
                        st.toast("Notes saved")

                    st.markdown("")
                    if st.button(
                        "🗑 Remove from watchlist",
                        key=f"rm_{selected_ticker}",
                        type="secondary",
                    ):
                        db.remove_from_watchlist(selected_ticker)
                        st.rerun()
