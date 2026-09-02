from __future__ import annotations
from config import settings
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:

    def __init__(
        self,
        log_file: str | Path = "log.txt",
    ) -> None:
        self.agent_name = settings.AGENT_NAME
        self.model_name = settings.LLM_MODEL
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.top_results = settings.TOP_K
        self.vector_size = settings.VECTOR_SIZE
        self.log_file = Path(log_file)

        self.session_start: Optional[float] = None
        self.session_started_at: Optional[datetime] = None
        self.session_active = False

        self._ensure_log_file()
        self._start_session()

    # ------------------------------------------------------------------
    # File / Session management
    # ------------------------------------------------------------------

    def _ensure_log_file(self) -> None:
        """Create the log file if it does not already exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Agent ASCII art.
        self._write(self._generate_ascii_art())

        if not self.log_file.exists():
            self.log_file.touch()

    def _write(self, text: str = "") -> None:
        """Append a line to the log file."""
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(f"{text}\n")

    def _write_empty_comment(self) -> None:
        """Write the required empty-space marker."""
        self._write(self.EMPTY_COMMENT)

    def _start_session(self) -> None:
        """Initialise a new logging session."""
        self.session_start = time.perf_counter()
        self.session_started_at = datetime.now()
        self.session_active = True

        # Separate this session from previous sessions.
        self._write()
        self._write()

        self._write("---| Session Started |---")

        self._write("> Initialising.....")

        self._write()
        self._write("--~--")
        self._write("Settings: ")
        self._write(f"\t LLM Model in use: {self.model_name}")
        self._write(f"\t Embedding Model in use: {self.embedding_model_name}")
        self._write(f"\t Vector Size: {self.vector_size}")
        self._write(f"\t Context Size: {self.top_results} tokens")
        self._write("--~--")
        self._write()

        self._write(f"> Contacting Database at {settings.QDRANT_URL} | Collection: {settings.QDRANT_COLLECTION}")
        self._write(f"> Contacting Model at {settings.OLLAMA_BASE_URL}")

        self._write(
            f"> Started at "
            f"{self.session_started_at.strftime('%H:%M')} @ "
            f"{self.session_started_at.strftime('%d/%m/%Y')}"
        )

    # ------------------------------------------------------------------
    # ASCII art
    # ------------------------------------------------------------------

    def _generate_ascii_art(self) -> str:
        border = "=" * (len(self.agent_name) + 12)

        return (
            f"+{border}+\n"
            f"|    {self.agent_name}    |\n"
            f"+{border}+"
        )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _elapsed(self) -> str:
        """Return seconds elapsed since the session began."""
        if self.session_start is None:
            return "0s"

        elapsed = time.perf_counter() - self.session_start

        if elapsed < 60:
            return f"{elapsed:.1f}s"

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if minutes == 1:
            return f"1m {seconds}s"

        return f"{minutes}m {seconds}s"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str) -> None:
        """
        Log a normal pipeline activity.
        """
        self._write(
            f"> [ LOG ] {message} {{ {self._elapsed()} }}"
        )

    def warn(self, message: str) -> None:
        """
        Log a recoverable warning.

        Warnings indicate something went wrong or something potentially
        problematic was detected, but the pipeline can continue.
        """
        self._write(
            f"> [ WARN ] {message} {{ {self._elapsed()} }}"
        )

    def error(self, message: str) -> None:
        """
        Log a serious error.

        This does not raise an exception itself. The calling pipeline
        decides whether the error should terminate execution.
        """
        self._write(
            f"> [ ERROR ] {message} {{ {self._elapsed()} }}"
        )

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def pipeline_completed(self) -> None:
        """
        Mark the pipeline as successfully completed and close the session.
        """
        if not self.session_active:
            return

        elapsed = self._elapsed()

        self._write(
            f"> Pipeline Completed in {elapsed} ✓"
        )

        self._write(
            f"> Ending at "
            f"{datetime.now().strftime('%H:%M')} @ "
            f"{datetime.now().strftime('%d/%m/%Y')}"
        )

        self._write("> Exiting Code...")

        self._write("---| Session Ended |---")

        self.session_active = False

    def pipeline_failed(self) -> None:
        """
        Mark the pipeline as failed and close the session.
        """
        if not self.session_active:
            return

        elapsed = self._elapsed()

        self._write(
            f"> Pipeline Failed after {elapsed} ⨉"
        )

        self._write("> Exiting Code...")

        self._write("---| Session Ended |---")

        self.session_active = False

    # ------------------------------------------------------------------
    # Exception helper
    # ------------------------------------------------------------------

    def exception(
        self,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """
        Log an exception as an ERROR.

        Example:
            logger.exception(
                "Failed to ingest document",
                exc
            )
        """
        if exc is not None:
            message = f"{message}: {type(exc).__name__}: {exc}"

        self.error(message)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def __enter__(self) -> "Logger":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        """
        Automatically close the session if used as a context manager.

        An exception means the pipeline failed.
        No exception means the pipeline completed successfully.
        """
        if self.session_active:
            if exc_type is None:
                self.pipeline_completed()
            else:
                self.exception(
                    "Unhandled pipeline exception",
                    exc_value,
                )
                self.pipeline_failed()

        # Do not suppress exceptions.
        return False
