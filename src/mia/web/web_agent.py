"""Headless/interactive Selenium web agent that executes structured plans."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore

try:
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.chrome.service import Service as ChromeService  # type: ignore
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
    _HAS_SELENIUM = True
except ImportError:  # pragma: no cover - optional dependency
    webdriver = None  # type: ignore
    ChromeService = None  # type: ignore
    ChromeDriverManager = None  # type: ignore
    _HAS_SELENIUM = False

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:
    from readability import Document  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class WebAgentStepResult:
    """Represents the output produced by a web agent step."""

    action: str
    success: bool
    message: str
    payload: Optional[Dict[str, Any]] = None


class WebAgentError(RuntimeError):
    """Raised when the web agent cannot fulfil a plan."""


class WebAgent:
    """Selenium powered agent executing declarative navigation plans."""

    def __init__(
        self,
        *,
        download_dir: str = "downloads",
        screenshot_dir: str = "screenshots",
        headless_default: bool = True,
        wait_timeout: float = 15.0,
    ) -> None:
        if not _HAS_SELENIUM:
            raise RuntimeError("Selenium and webdriver-manager are required for WebAgent. Install with 'pip install selenium webdriver-manager'.")

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.headless_default = headless_default
        self.wait_timeout = wait_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_plan(self, plan: Any, *, headless: Optional[bool] = None, viewport: Optional[tuple[int, int]] = None) -> List[WebAgentStepResult]:
        if not _HAS_SELENIUM:
            raise WebAgentError("Selenium not available. Install selenium and webdriver-manager to use web agent.")
        
        steps = self._normalise_plan(plan)
        if not steps:
            raise WebAgentError("Plan is empty or malformed")

        headless = self.headless_default if headless is None else headless
        options = webdriver.ChromeOptions()  # type: ignore
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size={}x{}".format(*(viewport or (1440, 900))))
        prefs = {
            "download.default_directory": str(self.download_dir.resolve()),
            "download.prompt_for_download": False,
            "profile.default_content_settings.popups": 0,
        }
        options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)  # type: ignore
        driver.set_page_load_timeout(int(self.wait_timeout))
        wait = WebDriverWait(driver, self.wait_timeout)

        results: List[WebAgentStepResult] = []
        try:
            for idx, step in enumerate(steps, start=1):
                action = step.get("action", "").lower()
                try:
                    payload = None
                    if action == "open":
                        payload = self._step_open(driver, step)
                    elif action == "click":
                        payload = self._step_click(driver, wait, step)
                    elif action == "type":
                        payload = self._step_type(driver, wait, step)
                    elif action == "wait":
                        payload = self._step_wait(driver, wait, step)
                    elif action == "scroll":
                        payload = self._step_scroll(driver, step)
                    elif action == "screenshot":
                        payload = self._step_screenshot(driver, idx, step)
                    elif action == "scrape":
                        payload = self._step_scrape(driver, step)
                    else:
                        raise WebAgentError(f"Unsupported action '{action}'")
                    results.append(WebAgentStepResult(action=action, success=True, message="ok", payload=payload))
                except Exception as exc:  # pragma: no cover - headless runtime failure
                    logger.exception("WebAgent step %s failed: %s", idx, exc)
                    results.append(WebAgentStepResult(action=action, success=False, message=str(exc)))
                    if step.get("halt_on_error", True):
                        break
            return results
        finally:
            driver.quit()

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _step_open(self, driver, step: Dict[str, Any]) -> Dict[str, Any]:
        url = step.get("url")
        if not url:
            raise WebAgentError("open step requires 'url'")
        driver.get(url)
        return {"url": url}

    def _resolve_locator(self, step: Dict[str, Any]) -> tuple[str, str]:
        if "css" in step:
            return By.CSS_SELECTOR, step["css"]
        if "xpath" in step:
            return By.XPATH, step["xpath"]
        if "name" in step:
            return By.NAME, step["name"]
        if "id" in step:
            return By.ID, step["id"]
        raise WebAgentError("Step requires locator (css/xpath/name/id)")

    def _step_click(self, driver, wait, step: Dict[str, Any]) -> Dict[str, Any]:
        locator = self._resolve_locator(step)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.click()
        return {"clicked": locator[1]}

    def _step_type(self, driver, wait, step: Dict[str, Any]) -> Dict[str, Any]:
        locator = self._resolve_locator(step)
        value = step.get("text", "")
        element = wait.until(EC.visibility_of_element_located(locator))
        if step.get("clear", True):
            element.clear()
        element.send_keys(value)
        return {"typed": value, "target": locator[1]}

    def _step_wait(self, driver, wait, step: Dict[str, Any]) -> Dict[str, Any]:
        if step.get("until") == "visible":
            locator = self._resolve_locator(step)
            wait.until(EC.visibility_of_element_located(locator))
            return {"waited_for": locator[1]}
        seconds = float(step.get("seconds", 1.0))
        time.sleep(max(0.0, seconds))
        return {"sleep": seconds}

    def _step_scroll(self, driver, step: Dict[str, Any]) -> Dict[str, Any]:
        position = step.get("to", "end")
        if position == "end":
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        else:
            driver.execute_script("window.scrollTo(0, arguments[0]);", int(position))
        return {"scroll": position}

    def _step_screenshot(self, driver, index: int, step: Dict[str, Any]) -> Dict[str, Any]:
        filename = step.get("path") or self.screenshot_dir / f"step_{index:02d}.png"
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(filename))
        return {"screenshot": str(filename.resolve())}

    def _step_scrape(self, driver, step: Dict[str, Any]) -> Dict[str, Any]:
        html = driver.page_source
        parsed = {
            "title": driver.title,
            "url": driver.current_url,
        }

        if Document is not None:  # readability extraction
            try:
                doc = Document(html)
                parsed["summary_html"] = doc.summary()
                parsed["summary_text"] = doc.summary().strip()
            except Exception as exc:  # pragma: no cover - optional failure
                logger.debug("Readability extraction failed: %s", exc)

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            parsed["text"] = soup.get_text("\n", strip=True)
        else:
            parsed["text"] = html

        if step.get("extract_tables") and pd is not None:
            try:
                tables = pd.read_html(html)
                parsed["tables"] = [table.to_dict(orient="records") for table in tables]
            except ValueError:
                parsed["tables"] = []
            except Exception as exc:  # pragma: no cover - optional failure
                logger.debug("Table extraction failed: %s", exc)
        elif step.get("extract_tables"):
            parsed["tables"] = []

        return parsed

    # ------------------------------------------------------------------
    # Unified search + scraping + summarization workflow
    # ------------------------------------------------------------------
    def unified_web_workflow(self, query: str, max_results: int = 5, summarize: bool = True) -> List[WebAgentStepResult]:
        """
        Unified workflow: search web, scrape results, and optionally summarize.
        
        Args:
            query: Search query
            max_results: Maximum number of results to process
            summarize: Whether to generate summaries
            
        Returns:
            List of step results from the workflow
        """
        results = []
        
        # Step 1: Search for the query
        search_plan = [
            {"action": "open", "url": f"https://www.google.com/search?q={query.replace(' ', '+')}"},
            {"action": "wait", "seconds": 2},
            {"action": "scrape", "extract_tables": False}
        ]
        
        search_results = self.run_plan(search_plan)
        results.extend(search_results)
        
        if not search_results or not any(r.success for r in search_results):
            return results
        
        # Extract search result URLs from the scraped content
        scraped_data = None
        for result in reversed(search_results):
            if result.success and result.payload and "text" in result.payload:
                scraped_data = result.payload
                break
        
        if not scraped_data:
            return results
        
        # Parse search results to extract URLs
        urls = self._extract_search_result_urls(scraped_data["text"])
        urls = urls[:max_results]  # Limit results
        
        # Step 2: Scrape each URL
        for i, url in enumerate(urls, 1):
            scrape_plan = [
                {"action": "open", "url": url},
                {"action": "wait", "seconds": 1},
                {"action": "scrape", "extract_tables": True}
            ]
            
            scrape_results = self.run_plan(scrape_plan)
            results.extend(scrape_results)
            
            # Step 3: Summarize if requested
            if summarize and scrape_results:
                last_result = scrape_results[-1]
                if last_result.success and last_result.payload:
                    summary = self._generate_content_summary(last_result.payload, query)
                    if summary:
                        results.append(WebAgentStepResult(
                            action="summarize",
                            success=True,
                            message=f"Summary for {url}",
                            payload={"summary": summary, "url": url, "query": query}
                        ))
        
        return results

    def _extract_search_result_urls(self, search_text: str) -> List[str]:
        """Extract URLs from Google search results text."""
        import re
        
        # Simple regex to find URLs in search results
        url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, search_text)
        
        # Filter out Google-specific URLs and duplicates
        filtered_urls = []
        seen = set()
        for url in urls:
            if not any(domain in url for domain in ['google.com', 'googleusercontent.com', 'gstatic.com']):
                if url not in seen:
                    filtered_urls.append(url)
                    seen.add(url)
        
        return filtered_urls[:10]  # Return top 10 unique URLs

    def _generate_content_summary(self, content: Dict[str, Any], query: str) -> Optional[str]:
        """Generate a summary of scraped content related to the query."""
        if not content or "text" not in content:
            return None
        
        text = content["text"]
        if len(text) < 100:
            return text  # Too short to summarize
        
        # Simple extractive summarization
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on query terms
        query_terms = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            words = set(sentence.lower().split())
            score = len(words.intersection(query_terms))
            scored_sentences.append((score, sentence))
        
        # Select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:3] if s[0] > 0]
        
        if top_sentences:
            return '. '.join(top_sentences) + '.'
        
        # Fallback: return first few sentences
        return '. '.join(sentences[:2]) + '.' if sentences else None
    def _normalise_plan(self, plan: Any) -> List[Dict[str, Any]]:
        if isinstance(plan, str):
            plan = json.loads(plan)
        if isinstance(plan, dict) and "steps" in plan:
            plan = plan["steps"]
        if not isinstance(plan, Iterable):
            raise WebAgentError("Plan must be a list of steps or an object with 'steps'")
        steps = []
        for step in plan:
            if isinstance(step, dict):
                steps.append(step)
        return steps


__all__ = ["WebAgent", "WebAgentError", "WebAgentStepResult"]
