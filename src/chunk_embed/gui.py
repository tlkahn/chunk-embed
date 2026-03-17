"""PySide6 GUI for the chunk-embed ingest + search pipeline."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QThread, Signal, QSize, QUrl, Qt
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from chunk_embed.embed import MODEL_DIR
from chunk_embed.models import ALL_CHUNK_TYPES, TEXTUAL_TYPES, ChunkSummary, DocumentInfo, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-select combo box
# ---------------------------------------------------------------------------

class CheckableComboBox(QComboBox):
    """QComboBox that allows selecting multiple items via checkboxes.

    Displays checked item names joined by ", " in the line-edit.
    An empty selection is equivalent to "(all)".
    """

    selection_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("(all)")
        self.model().dataChanged.connect(self._update_display)
        self._updating = False
        # Intercept clicks on the popup list to toggle checkboxes.
        self.view().viewport().installEventFilter(self)

    # -- public API --

    def checked_items(self) -> list[str]:
        """Return the list of checked item texts."""
        items: list[str] = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                items.append(item.text())
        return items

    def set_items(self, items: list[str], previously_checked: list[str] | None = None) -> None:
        """Replace all items and optionally restore checked state."""
        self._updating = True
        self.clear()
        for text in items:
            self.addItem(text)
            item = self.model().item(self.count() - 1)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if previously_checked and text in previously_checked:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        self._updating = False
        self._update_display()

    # -- overrides --

    def addItem(self, text, userData=None):  # noqa: N802
        super().addItem(text, userData)
        item = self.model().item(self.count() - 1)
        if item is not None and not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)

    def hidePopup(self):  # noqa: N802
        # Only close the popup when clicking outside, not on item toggle.
        if not self.view().underMouse():
            super().hidePopup()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                item = self.model().item(index.row())
                if item is not None and (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                    new_state = (
                        Qt.CheckState.Unchecked
                        if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    item.setCheckState(new_state)
                    return True  # consume the event
        return super().eventFilter(obj, event)

    # -- internal --

    def _update_display(self) -> None:
        if self._updating:
            return
        checked = self.checked_items()
        self.lineEdit().setText(", ".join(checked) if checked else "")
        self.lineEdit().setToolTip(", ".join(checked) if checked else "(all)")
        self.selection_changed.emit()


# ---------------------------------------------------------------------------
# Log forwarding to Qt
# ---------------------------------------------------------------------------

class _LogSignalBridge(QObject):
    """Thin QObject that owns a signal for cross-thread log delivery."""
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that emits formatted records via a Qt signal."""

    def __init__(self, bridge: _LogSignalBridge) -> None:
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        self._bridge.message.emit(self.format(record))


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------

@dataclass
class DepStatus:
    name: str
    ok: bool
    detail: str
    install_hint: str
    help_page: str = ""


def _check_dependencies(database_url: str) -> list[DepStatus]:
    results: list[DepStatus] = []

    # BGE-M3 model
    model_ok = MODEL_DIR.is_dir()
    results.append(DepStatus(
        name="BGE-M3 model",
        ok=model_ok,
        detail=str(MODEL_DIR) if model_ok else f"{MODEL_DIR} not found",
        install_hint="Auto-downloaded on first embed, or manually place in local_bge_m3/",
        help_page="help-bge-m3.html",
    ))

    # Editor ($VISUAL / $EDITOR)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        editor_path = shutil.which(Path(editor).name)
        results.append(DepStatus(
            name="Editor",
            ok=editor_path is not None,
            detail=editor or "set but not found in PATH",
            install_hint="Ensure $VISUAL or $EDITOR points to a valid executable",
            help_page="help-editor.html",
        ))
    else:
        results.append(DepStatus(
            name="Editor",
            ok=False,
            detail="$VISUAL and $EDITOR not set — source links open without line positioning",
            install_hint='export EDITOR=code  (or vim, nvim, emacs, subl, etc.)',
            help_page="help-editor.html",
        ))

    # PostgreSQL + pgvector
    db_detail = ""
    db_ok = False
    try:
        import psycopg
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname='vector'"
            ).fetchone()
            if row:
                db_ok = True
                db_detail = f"Connected, pgvector extension present"
            else:
                db_detail = "Connected but pgvector extension missing"
    except Exception as exc:
        db_detail = str(exc)
    results.append(DepStatus(
        name="PostgreSQL + pgvector",
        ok=db_ok,
        detail=db_detail,
        install_hint='brew install postgresql; createdb chunk_embed; psql chunk_embed -c "CREATE EXTENSION vector"',
        help_page="help-postgresql.html",
    ))

    return results

# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class IngestWorker(QThread):
    """Run the ingest pipeline off the main thread."""

    progress = Signal(str, int, int)  # stage, current, total
    log = Signal(str)
    finished = Signal(str)  # summary message
    error = Signal(str)

    def __init__(
        self,
        file_paths: list[str],
        split: bool,
        database_url: str,
        dry_run: bool,
        batch_size: int = 32,
        allowed_types: frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.split = split
        self.database_url = database_url
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.allowed_types = allowed_types

    def run(self) -> None:
        try:
            self._run_pipeline()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run_pipeline(self) -> None:
        from chunk_embed.embed import BgeM3Embedder
        from chunk_embed.pipeline import ingest_one_file

        # Load embedder with log bridge so model-loading messages reach the GUI
        bridge = _LogSignalBridge()
        bridge.message.connect(self.log.emit)
        handler = QtLogHandler(bridge)
        handler.setFormatter(logging.Formatter("%(message)s"))
        embed_logger = logging.getLogger("chunk_embed.embed")
        st_logger = logging.getLogger("sentence_transformers")
        embed_logger.addHandler(handler)
        st_logger.addHandler(handler)
        try:
            self.log.emit("Loading embedding model…")
            self.progress.emit("Loading model", 0, 0)
            embedder = BgeM3Embedder()
        finally:
            embed_logger.removeHandler(handler)
            st_logger.removeHandler(handler)

        total_files = len(self.file_paths)
        succeeded = 0
        failed = 0

        for i, fp in enumerate(self.file_paths, 1):
            prefix = f"[{i}/{total_files}] {Path(fp).name}"
            self.log.emit(f"{prefix}: starting…")
            try:
                result = ingest_one_file(
                    file_path=Path(fp),
                    source=None,
                    embedder=embedder,
                    split=self.split,
                    batch_size=self.batch_size,
                    database_url=self.database_url,
                    dry_run=self.dry_run,
                    allowed_types=self.allowed_types,
                    on_log=lambda msg, _p=prefix: self.log.emit(f"{_p}: {msg}"),
                    on_embed_progress=lambda done, tot, _p=prefix: self.progress.emit(f"{_p} — Embedding", done, tot),
                    on_store_progress=lambda done, tot, _p=prefix: self.progress.emit(f"{_p} — Storing", done, tot),
                )
                if result.dry_run:
                    self.log.emit(f"{prefix}: dry run — {result.num_embeddings} embeddings produced")
                else:
                    self.log.emit(f"{prefix}: done — doc {result.doc_id}, {result.num_chunks} chunks stored")
                succeeded += 1
            except Exception as exc:
                self.log.emit(f"{prefix}: ERROR — {exc}")
                failed += 1

        parts = [f"{succeeded} succeeded"]
        if failed:
            parts.append(f"{failed} failed")
        self.finished.emit(f"Done: {', '.join(parts)} ({total_files} files)")


class QueryWorker(QThread):
    """Run a semantic search off the main thread."""

    log = Signal(str)
    finished = Signal(list)  # list[SearchResult]
    error = Signal(str)

    def __init__(
        self,
        query_text: str,
        database_url: str,
        top_k: int = 10,
        source_filter: list[str] | None = None,
        chunk_type_filter: list[str] | None = None,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_text = query_text
        self.database_url = database_url
        self.top_k = top_k
        self.source_filter = source_filter or []
        self.chunk_type_filter = chunk_type_filter or []
        self.threshold = threshold

    def run(self) -> None:
        try:
            self._run_query()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run_query(self) -> None:
        from chunk_embed.embed import BgeM3Embedder
        from chunk_embed.store import search_chunks
        import psycopg
        from pgvector.psycopg import register_vector

        bridge = _LogSignalBridge()
        bridge.message.connect(self.log.emit)
        handler = QtLogHandler(bridge)
        handler.setFormatter(logging.Formatter("%(message)s"))
        embed_logger = logging.getLogger("chunk_embed.embed")
        st_logger = logging.getLogger("sentence_transformers")
        embed_logger.addHandler(handler)
        st_logger.addHandler(handler)
        try:
            self.log.emit("Loading embedding model…")
            embedder = BgeM3Embedder()
        finally:
            embed_logger.removeHandler(handler)
            st_logger.removeHandler(handler)

        self.log.emit("Embedding query…")
        query_embedding = embedder.embed([self.query_text])[0]

        self.log.emit("Searching…")
        with psycopg.connect(self.database_url) as conn:
            register_vector(conn)
            results = search_chunks(
                conn,
                query_embedding,
                top_k=self.top_k,
                source_paths=self.source_filter or None,
                chunk_types=self.chunk_type_filter or None,
                threshold=self.threshold,
            )

        self.finished.emit(results)


class FilterOptionsWorker(QThread):
    """Fetch distinct source paths and chunk types from the database."""

    finished = Signal(list, list)  # sources, chunk_types
    error = Signal(str)

    def __init__(self, database_url: str) -> None:
        super().__init__()
        self.database_url = database_url

    def run(self) -> None:
        try:
            import psycopg
            from chunk_embed.store import get_distinct_sources, get_distinct_chunk_types

            with psycopg.connect(self.database_url, connect_timeout=3) as conn:
                sources = get_distinct_sources(conn)
                chunk_types = get_distinct_chunk_types(conn)
            self.finished.emit(sources, chunk_types)
        except Exception as exc:
            self.error.emit(str(exc))


class DocsListWorker(QThread):
    """Fetch all documents from the database."""

    finished = Signal(list)  # list[DocumentInfo]
    error = Signal(str)

    def __init__(self, database_url: str) -> None:
        super().__init__()
        self.database_url = database_url

    def run(self) -> None:
        try:
            import psycopg
            from chunk_embed.store import list_documents

            with psycopg.connect(self.database_url, connect_timeout=3) as conn:
                docs = list_documents(conn)
            self.finished.emit(docs)
        except Exception as exc:
            self.error.emit(str(exc))


class DocsDetailWorker(QThread):
    """Fetch chunk summary for a single document."""

    finished = Signal(int, list)  # (document_id, list[ChunkSummary])
    error = Signal(str)

    def __init__(self, database_url: str, document_id: int) -> None:
        super().__init__()
        self.database_url = database_url
        self.document_id = document_id

    def run(self) -> None:
        try:
            import psycopg
            from chunk_embed.store import get_chunk_summary

            with psycopg.connect(self.database_url, connect_timeout=3) as conn:
                summaries = get_chunk_summary(conn, self.document_id)
            self.finished.emit(self.document_id, summaries)
        except Exception as exc:
            self.error.emit(str(exc))


class DocsDeleteWorker(QThread):
    """Delete documents by source path."""

    finished = Signal(int)  # count deleted
    error = Signal(str)

    def __init__(self, database_url: str, source_paths: list[str]) -> None:
        super().__init__()
        self.database_url = database_url
        self.source_paths = source_paths

    def run(self) -> None:
        try:
            import psycopg
            from chunk_embed.store import delete_documents

            with psycopg.connect(self.database_url) as conn:
                count = delete_documents(conn, self.source_paths)
                conn.commit()
            self.finished.emit(count)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("chunk-embed")
        self.setMinimumSize(QSize(720, 560))
        self._worker: QThread | None = None
        self._filter_worker: FilterOptionsWorker | None = None
        self._last_results: list[SearchResult] = []
        self._docs_worker: QThread | None = None
        self._docs_detail_worker: QThread | None = None
        self._docs_documents: list[DocumentInfo] = []

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_setup_tab()   # index 0
        self._build_ingest_tab()  # index 1
        self._build_search_tab()  # index 2
        self._build_docs_tab()    # index 3

        # Status bar with embedded progress bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(250)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Initial dependency check
        self._run_dep_check(startup=True)

    # ---- Setup tab ----

    def _build_setup_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Dependencies"))

        # Database URL (single shared widget)
        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("Database URL:"))
        self.db_url = QLineEdit()
        self.db_url.setText(os.environ.get("DATABASE_URL", "postgresql://localhost/chunk_embed"))
        db_row.addWidget(self.db_url, 1)
        self.test_conn_btn = QPushButton("Test Connection")
        self.test_conn_btn.clicked.connect(self._test_connection)
        db_row.addWidget(self.test_conn_btn)
        layout.addLayout(db_row)

        # Dependency status table
        self.dep_table = QTableWidget()
        self.dep_table.setColumnCount(4)
        self.dep_table.setHorizontalHeaderLabels(["Dependency", "Status", "Detail", "Install / Fix"])
        header = self.dep_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.dep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dep_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.dep_table.verticalHeader().setVisible(False)
        self.dep_table.itemSelectionChanged.connect(self._update_dep_label_colors)
        layout.addWidget(self.dep_table, 1)

        # Re-check button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.recheck_btn = QPushButton("Re-check All")
        self.recheck_btn.clicked.connect(self._run_dep_check)
        btn_row.addWidget(self.recheck_btn)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Setup")

    _HELP_DIR = Path(__file__).resolve().parents[2] / "resources" / "doc"

    def _populate_dep_table(self, statuses: list[DepStatus]) -> None:
        self.dep_table.setRowCount(len(statuses))
        green = QColor(0, 160, 0)
        red = QColor(200, 0, 0)
        for i, dep in enumerate(statuses):
            self.dep_table.setItem(i, 0, QTableWidgetItem(dep.name))
            status_item = QTableWidgetItem("OK" if dep.ok else "Missing")
            status_item.setForeground(green if dep.ok else red)
            self.dep_table.setItem(i, 1, status_item)
            self.dep_table.setItem(i, 2, QTableWidgetItem(dep.detail))
            # Install/Fix column: hint text + "More…" link
            help_file = self._HELP_DIR / dep.help_page if dep.help_page else None
            if help_file and help_file.exists():
                url = QUrl.fromLocalFile(str(help_file))
                label = QLabel()
                label.linkActivated.connect(self._open_help_link)
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setWordWrap(True)
                label.setProperty("hint_text", dep.install_hint)
                label.setProperty("hint_url", url.toString())
                self.dep_table.setCellWidget(i, 3, label)
            else:
                self.dep_table.setItem(i, 3, QTableWidgetItem(dep.install_hint))
        self._update_dep_label_colors()
        self.dep_table.resizeRowsToContents()

    def _update_dep_label_colors(self) -> None:
        pal = self.dep_table.palette()
        sel_text = pal.color(pal.ColorRole.HighlightedText).name()
        norm_text = pal.color(pal.ColorRole.Text).name()
        norm_link = pal.color(pal.ColorRole.Link).name()
        selected_rows = {idx.row() for idx in self.dep_table.selectionModel().selectedRows()}
        for row in range(self.dep_table.rowCount()):
            label = self.dep_table.cellWidget(row, 3)
            if not isinstance(label, QLabel):
                continue
            hint = label.property("hint_text")
            url = label.property("hint_url")
            if row in selected_rows:
                text_col, link_col = sel_text, sel_text
            else:
                text_col, link_col = norm_text, norm_link
            label.setText(
                f'<span style="color:{text_col}">{hint}</span>'
                f'  <a href="{url}" style="color:{link_col}">More\u2026</a>'
            )

    def _open_help_link(self, _url: str) -> None:
        label = self.sender()
        if label is not None:
            webbrowser.open(label.property("hint_url"))

    def _run_dep_check(self, startup: bool = False) -> None:
        statuses = _check_dependencies(self.db_url.text())
        self._populate_dep_table(statuses)
        all_ok = all(s.ok for s in statuses)

        # Enable/disable Ingest, Search, and Documents tabs
        self.tabs.setTabEnabled(1, all_ok)  # Ingest
        self.tabs.setTabEnabled(2, all_ok)  # Search
        self.tabs.setTabEnabled(3, all_ok)  # Documents

        if all_ok:
            if startup:
                self.tabs.setCurrentIndex(1)  # jump to Ingest on first launch
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("All dependencies satisfied")
            self._refresh_filters()
            self._refresh_docs()
        else:
            self.tabs.setCurrentIndex(0)  # stay on Setup
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Some dependencies are missing — see Setup tab")

    def _test_connection(self) -> None:
        """Re-check only the PostgreSQL dependency and update its row."""
        statuses = _check_dependencies(self.db_url.text())
        self._populate_dep_table(statuses)
        # Find the DB row and report
        for s in statuses:
            if s.name == "PostgreSQL + pgvector":
                if hasattr(self, "status_bar"):
                    if s.ok:
                        self.status_bar.showMessage("Database connection OK")
                    else:
                        self.status_bar.showMessage(f"Database: {s.detail}")
                break
        all_ok = all(s.ok for s in statuses)
        self.tabs.setTabEnabled(1, all_ok)
        self.tabs.setTabEnabled(2, all_ok)
        self.tabs.setTabEnabled(3, all_ok)

    # ---- Ingest tab ----

    def _build_ingest_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Source: [Browse… ▾]  filename.md
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        browse_menu = QMenu(self)
        browse_menu.addAction(QAction("Files…", self, triggered=self._browse_files))
        browse_menu.addAction(QAction("Folder…", self, triggered=self._browse_folder))
        file_btn = QPushButton("Browse")
        file_btn.setMenu(browse_menu)
        src_row.addWidget(file_btn)
        self.ingest_source = QLabel("No file selected")
        self.ingest_source.setStyleSheet("color: gray;")
        self.ingest_source.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        src_row.addWidget(self.ingest_source, 1)
        layout.addLayout(src_row)

        # Split + dry run + ingest button
        opt_row = QHBoxLayout()
        self.split_check = QCheckBox("Sentence split")
        self.split_check.setChecked(True)
        opt_row.addWidget(self.split_check)
        self.dry_run_check = QCheckBox("Dry run")
        opt_row.addWidget(self.dry_run_check)

        opt_row.addWidget(QLabel("Types:"))
        self.ingest_type_filter = CheckableComboBox()
        all_types_sorted = sorted(ALL_CHUNK_TYPES)
        self.ingest_type_filter.set_items(
            all_types_sorted,
            previously_checked=sorted(TEXTUAL_TYPES),
        )
        opt_row.addWidget(self.ingest_type_filter)

        opt_row.addStretch()
        self.ingest_btn = QPushButton("Ingest")
        self.ingest_btn.setDefault(True)
        self.ingest_btn.clicked.connect(self._start_ingest)
        opt_row.addWidget(self.ingest_btn)
        layout.addLayout(opt_row)

        # Log area
        self.ingest_log = QTextEdit()
        self.ingest_log.setReadOnly(True)
        layout.addWidget(self.ingest_log, 1)

        self.tabs.addTab(tab, "Ingest")

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select files", "",
            "Supported Files (*.json *.md *.markdown *.mdown *.mkd);;JSON Files (*.json);;Markdown Files (*.md *.markdown *.mdown *.mkd);;All Files (*)",
        )
        if paths:
            self._ingest_file_paths = paths
            if len(paths) == 1:
                self.ingest_source.setText(paths[0])
            else:
                self.ingest_source.setText(f"{len(paths)} files selected")
            self.ingest_source.setStyleSheet("")

    def _browse_folder(self) -> None:
        from chunk_embed.pipeline import resolve_paths

        directory = QFileDialog.getExistingDirectory(self, "Select folder")
        if directory:
            expanded = resolve_paths([directory])
            if not expanded:
                self.ingest_source.setText(f"{Path(directory).name}/ (0 eligible files)")
                self.ingest_source.setStyleSheet("color: gray;")
                self._ingest_file_paths = []
                return
            self._ingest_file_paths = [str(p) for p in expanded]
            self.ingest_source.setText(f"{Path(directory).name}/ ({len(expanded)} files)")
            self.ingest_source.setStyleSheet("")

    def _start_ingest(self) -> None:
        file_paths = getattr(self, "_ingest_file_paths", [])
        if not file_paths:
            self.status_bar.showMessage("No files selected")
            return

        self.ingest_log.clear()
        self.ingest_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        checked = self.ingest_type_filter.checked_items()
        ingest_types = frozenset(checked) if checked else None

        self._worker = IngestWorker(
            file_paths=file_paths,
            split=self.split_check.isChecked(),
            database_url=self.db_url.text(),
            dry_run=self.dry_run_check.isChecked(),
            allowed_types=ingest_types,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(lambda msg: self.ingest_log.append(msg))
        self._worker.finished.connect(self._on_ingest_done)
        self._worker.error.connect(self._on_ingest_error)
        self._worker.start()

    def _cleanup_worker(self) -> None:
        """Wait for the worker thread to fully stop before dropping the reference."""
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

    def _on_progress(self, stage: str, current: int, total: int) -> None:
        if total == 0:
            # Indeterminate / pulsating mode
            self.progress_bar.setMaximum(0)
            self.status_bar.showMessage(f"{stage}…")
        else:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.status_bar.showMessage(f"{stage}: {current}/{total}")

    def _on_ingest_done(self, summary: str) -> None:
        self.ingest_log.append(summary)
        self.status_bar.showMessage(summary)
        self.progress_bar.hide()
        self.ingest_btn.setEnabled(True)
        self._cleanup_worker()
        self._refresh_filters()
        self._refresh_docs()

    def _on_ingest_error(self, msg: str) -> None:
        self.ingest_log.append(f"ERROR: {msg}")
        self.status_bar.showMessage(f"Error: {msg}")
        self.progress_bar.hide()
        self.ingest_btn.setEnabled(True)
        self._cleanup_worker()

    # ---- Documents tab ----

    def _build_docs_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Documents table
        self.docs_table = QTableWidget()
        self.docs_table.setColumnCount(5)
        self.docs_table.setHorizontalHeaderLabels(["ID", "Mode", "Chunks", "Created", "Source"])
        header = self.docs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.docs_table.setColumnWidth(0, 50)
        self.docs_table.setColumnWidth(1, 70)
        self.docs_table.setColumnWidth(2, 60)
        self.docs_table.setColumnWidth(3, 140)
        header.sectionResized.connect(lambda: self.docs_table.resizeRowsToContents())
        self.docs_table.setWordWrap(True)
        self.docs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.docs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.docs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.docs_table.verticalHeader().setVisible(False)
        self.docs_table.itemSelectionChanged.connect(self._on_doc_selection_changed)
        layout.addWidget(self.docs_table, 2)

        # Chunk breakdown detail
        layout.addWidget(QLabel("Chunk breakdown"))
        self.docs_detail_table = QTableWidget()
        self.docs_detail_table.setColumnCount(3)
        self.docs_detail_table.setHorizontalHeaderLabels(["Type", "Count", "Total Chars"])
        detail_header = self.docs_detail_table.horizontalHeader()
        detail_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        detail_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        detail_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.docs_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.docs_detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.docs_detail_table.verticalHeader().setVisible(False)
        layout.addWidget(self.docs_detail_table, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.docs_refresh_btn = QPushButton("Refresh")
        self.docs_refresh_btn.clicked.connect(self._refresh_docs)
        btn_row.addWidget(self.docs_refresh_btn)
        self.docs_delete_btn = QPushButton("Delete")
        self.docs_delete_btn.setEnabled(False)
        self.docs_delete_btn.clicked.connect(self._delete_selected_docs)
        btn_row.addWidget(self.docs_delete_btn)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Documents")

    def _on_doc_selection_changed(self) -> None:
        selected = self.docs_table.selectionModel().selectedRows()
        self.docs_delete_btn.setEnabled(len(selected) > 0)
        if len(selected) == 1:
            row = selected[0].row()
            if row < len(self._docs_documents):
                doc = self._docs_documents[row]
                self._fetch_doc_detail(doc.id)
                return
        self.docs_detail_table.setRowCount(0)

    def _fetch_doc_detail(self, document_id: int) -> None:
        if self._docs_detail_worker is not None:
            return
        self._docs_detail_worker = DocsDetailWorker(self.db_url.text(), document_id)
        self._docs_detail_worker.finished.connect(self._on_docs_detail_done)
        self._docs_detail_worker.error.connect(self._on_docs_detail_error)
        self._docs_detail_worker.finished.connect(self._cleanup_docs_detail_worker)
        self._docs_detail_worker.error.connect(self._cleanup_docs_detail_worker)
        self._docs_detail_worker.start()

    def _refresh_docs(self) -> None:
        if self._docs_worker is not None:
            return
        self._docs_worker = DocsListWorker(self.db_url.text())
        self._docs_worker.finished.connect(self._on_docs_list_done)
        self._docs_worker.error.connect(self._on_docs_list_error)
        self._docs_worker.finished.connect(self._cleanup_docs_worker)
        self._docs_worker.error.connect(self._cleanup_docs_worker)
        self._docs_worker.start()

    def _on_docs_list_done(self, documents: list) -> None:
        self._docs_documents = documents
        self.docs_table.setRowCount(len(documents))
        for i, doc in enumerate(documents):
            self.docs_table.setItem(i, 0, QTableWidgetItem(str(doc.id)))
            self.docs_table.setItem(i, 1, QTableWidgetItem(doc.mode))
            self.docs_table.setItem(i, 2, QTableWidgetItem(str(doc.total_chunks)))
            created_str = doc.created_at.strftime("%Y-%m-%d %H:%M")
            self.docs_table.setItem(i, 3, QTableWidgetItem(created_str))
            source_item = QTableWidgetItem(doc.source_path)
            source_font = source_item.font()
            source_font.setPointSizeF(source_font.pointSizeF() * 0.85)
            source_item.setFont(source_font)
            self.docs_table.setItem(i, 4, source_item)
        self.docs_detail_table.setRowCount(0)

    def _on_docs_list_error(self, msg: str) -> None:
        self.status_bar.showMessage(f"Documents: {msg}")

    def _on_docs_detail_done(self, doc_id: int, summaries: list) -> None:
        self.docs_detail_table.setRowCount(len(summaries))
        for i, s in enumerate(summaries):
            self.docs_detail_table.setItem(i, 0, QTableWidgetItem(s.chunk_type))
            self.docs_detail_table.setItem(i, 1, QTableWidgetItem(str(s.count)))
            self.docs_detail_table.setItem(i, 2, QTableWidgetItem(str(s.total_chars)))

    def _on_docs_detail_error(self, msg: str) -> None:
        self.status_bar.showMessage(f"Chunk detail: {msg}")

    def _delete_selected_docs(self) -> None:
        selected = self.docs_table.selectionModel().selectedRows()
        if not selected:
            return
        paths = []
        for idx in selected:
            row = idx.row()
            if row < len(self._docs_documents):
                paths.append(self._docs_documents[row].source_path)
        if not paths:
            return
        n = len(paths)
        answer = QMessageBox.question(
            self,
            "Delete documents",
            f"Delete {n} document{'s' if n != 1 else ''} and all associated chunks?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.docs_delete_btn.setEnabled(False)
        if self._docs_worker is not None:
            return
        self._docs_worker = DocsDeleteWorker(self.db_url.text(), paths)
        self._docs_worker.finished.connect(self._cleanup_docs_worker)
        self._docs_worker.error.connect(self._cleanup_docs_worker)
        self._docs_worker.finished.connect(self._on_docs_delete_done)
        self._docs_worker.error.connect(self._on_docs_delete_error)
        self._docs_worker.start()

    def _on_docs_delete_done(self, count: int) -> None:
        self.status_bar.showMessage(f"Deleted {count} document{'s' if count != 1 else ''}")
        self._refresh_docs()
        self._refresh_filters()

    def _on_docs_delete_error(self, msg: str) -> None:
        self.status_bar.showMessage(f"Delete error: {msg}")
        self.docs_delete_btn.setEnabled(True)

    def _cleanup_docs_worker(self) -> None:
        if self._docs_worker is not None:
            self._docs_worker.wait()
            self._docs_worker = None

    def _cleanup_docs_detail_worker(self) -> None:
        if self._docs_detail_worker is not None:
            self._docs_detail_worker.wait()
            self._docs_detail_worker = None

    # ---- Filter dropdowns ----

    @staticmethod
    def _get_filter_value(combo: CheckableComboBox) -> list[str]:
        return combo.checked_items()

    def _refresh_filters(self) -> None:
        if self._filter_worker is not None:
            return  # already running
        self._filter_worker = FilterOptionsWorker(self.db_url.text())
        self._filter_worker.finished.connect(self._on_filters_loaded)
        self._filter_worker.error.connect(self._on_filters_error)
        self._filter_worker.finished.connect(self._cleanup_filter_worker)
        self._filter_worker.error.connect(self._cleanup_filter_worker)
        self._filter_worker.start()

    def _on_filters_loaded(self, sources: list, chunk_types: list) -> None:
        self._repopulate_combo(self.source_filter, sources)
        self._repopulate_combo(self.chunk_type_filter, chunk_types)

    @staticmethod
    def _repopulate_combo(combo: CheckableComboBox, items: list[str]) -> None:
        previously_checked = combo.checked_items()
        combo.set_items(items, previously_checked)

    def _on_filters_error(self, msg: str) -> None:
        logger.warning("Failed to load filter options: %s", msg)

    def _cleanup_filter_worker(self) -> None:
        if self._filter_worker is not None:
            self._filter_worker.wait()
            self._filter_worker = None

    # ---- Search tab ----

    def _build_search_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Query input
        q_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter search query…")
        self.query_input.returnPressed.connect(self._start_query)
        q_row.addWidget(self.query_input, 1)
        self.query_btn = QPushButton("Search")
        self.query_btn.setDefault(False)
        self.query_btn.clicked.connect(self._start_query)
        q_row.addWidget(self.query_btn)
        layout.addLayout(q_row)

        # Filters row
        filt_row = QHBoxLayout()

        filt_row.addWidget(QLabel("Top-K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 200)
        self.top_k_spin.setValue(10)
        filt_row.addWidget(self.top_k_spin)

        filt_row.addWidget(QLabel("Source:"))
        self.source_filter = CheckableComboBox()
        self.source_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        filt_row.addWidget(self.source_filter)

        filt_row.addWidget(QLabel("Type:"))
        self.chunk_type_filter = CheckableComboBox()
        self.chunk_type_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        filt_row.addWidget(self.chunk_type_filter)

        filt_row.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        filt_row.addWidget(self.threshold_spin)

        layout.addLayout(filt_row)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["#", "Score", "Source", "Text", "Headings"])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.results_table.setColumnWidth(2, 200)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.results_table.setColumnWidth(4, 150)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setWordWrap(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.cellClicked.connect(self._on_result_cell_clicked)
        self.results_table.viewport().setMouseTracking(True)
        self.results_table.viewport().installEventFilter(self)
        layout.addWidget(self.results_table, 1)

        # Export button
        export_row = QHBoxLayout()
        export_row.addStretch()
        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        self.tabs.addTab(tab, "Search")

    def _start_query(self) -> None:
        query_text = self.query_input.text().strip()
        if not query_text:
            self.status_bar.showMessage("Enter a query")
            return

        self.query_btn.setEnabled(False)
        self.status_bar.showMessage("Searching…")
        self.progress_bar.setMaximum(0)  # indeterminate
        self.progress_bar.show()

        self._worker = QueryWorker(
            query_text=query_text,
            database_url=self.db_url.text(),
            top_k=self.top_k_spin.value(),
            source_filter=self._get_filter_value(self.source_filter),
            chunk_type_filter=self._get_filter_value(self.chunk_type_filter),
            threshold=self.threshold_spin.value(),
        )
        self._worker.log.connect(lambda msg: self.status_bar.showMessage(msg))
        self._worker.finished.connect(self._on_query_done)
        self._worker.error.connect(self._on_query_error)
        self._worker.start()

    def _on_query_done(self, results: list) -> None:
        self._last_results = results
        self.progress_bar.hide()
        self.query_btn.setEnabled(True)

        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{r.similarity:.4f}"))

            source_item = QTableWidgetItem(f"{r.source_path}:{r.source_line_start}")
            source_item.setForeground(QColor(0, 102, 204))
            source_font = source_item.font()
            source_font.setUnderline(True)
            source_font.setPointSizeF(source_font.pointSizeF() * 0.85)
            source_item.setFont(source_font)
            self.results_table.setItem(i, 2, source_item)

            text_preview = r.text[:300] + "…" if len(r.text) > 300 else r.text
            text_item = QTableWidgetItem(text_preview)
            text_font = QFont()
            text_font.setPointSizeF(text_font.pointSizeF() * 1.05)
            text_item.setFont(text_font)
            self.results_table.setItem(i, 3, text_item)

            heading_item = QTableWidgetItem(" > ".join(r.heading_context))
            heading_font = heading_item.font()
            heading_font.setPointSizeF(heading_font.pointSizeF() * 0.85)
            heading_item.setFont(heading_font)
            heading_item.setForeground(QColor(130, 130, 130))
            self.results_table.setItem(i, 4, heading_item)
        self.results_table.resizeRowsToContents()

        n = len(results)
        self.status_bar.showMessage(f"{n} result{'s' if n != 1 else ''} found")
        self.export_btn.setEnabled(n > 0)
        self._cleanup_worker()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        tw = self.results_table.width()
        max_source_w = tw // 6
        if self.results_table.columnWidth(2) > max_source_w:
            self.results_table.setColumnWidth(2, max_source_w)
        max_heading_w = tw // 8
        if self.results_table.columnWidth(4) > max_heading_w:
            self.results_table.setColumnWidth(4, max_heading_w)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.results_table.viewport() and event.type() == QEvent.Type.MouseMove:
            item = self.results_table.itemAt(event.position().toPoint())
            if item is not None and item.column() == 2:
                obj.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                obj.unsetCursor()
        return super().eventFilter(obj, event)

    def _on_result_cell_clicked(self, row: int, col: int) -> None:
        if col == 2 and row < len(self._last_results):
            r = self._last_results[row]
            self._open_source(r.source_path, r.source_line_start)

    def _open_source(self, source_path: str, line: int = 1) -> None:
        """Open the source file in the user's editor at the given line."""
        p = str(Path(source_path).resolve())
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if editor:
            editor_base = Path(editor).name
            try:
                if editor_base in ("code", "code-insiders"):
                    subprocess.Popen([editor, "--goto", f"{p}:{line}"])
                elif editor_base in ("subl", "sublime_text"):
                    subprocess.Popen([editor, f"{p}:{line}"])
                else:
                    # vim, nvim, emacs, nano, etc. all accept +line
                    subprocess.Popen([editor, f"+{line}", p])
                return
            except OSError:
                pass  # fall through to default
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _on_query_error(self, msg: str) -> None:
        self.status_bar.showMessage(f"Error: {msg}")
        self.progress_bar.hide()
        self.query_btn.setEnabled(True)
        self._cleanup_worker()

    def _export_results(self) -> None:
        if not self._last_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "results.json", "JSON Files (*.json)",
        )
        if not path:
            return

        from chunk_embed.format import format_results_json
        Path(path).write_text(format_results_json(self._last_results))
        self.status_bar.showMessage(f"Exported to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from chunk_embed._paths import prepend_bundled_bin_to_path
    prepend_bundled_bin_to_path()
    app = QApplication(sys.argv)
    app.setApplicationName("chunk-embed")
    icon_path = Path(__file__).resolve().parents[2] / "icons" / "chunk_embed.icns"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
