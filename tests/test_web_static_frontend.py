import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class WebStaticFrontendTestCase(unittest.TestCase):
    def _load_html(self) -> str:
        return Path("complex_document_rag/web_static/index.html").read_text(encoding="utf-8")

    def _css_rule(self, html: str, selector: str) -> str:
        style_match = re.search(r"<style>(.*)</style>", html, re.S)
        self.assertIsNotNone(style_match, "expected inline <style> in frontend template")
        rule_match = re.search(rf"{re.escape(selector)}\s*\{{(.*?)\}}", style_match.group(1), re.S)
        self.assertIsNotNone(rule_match, f"expected CSS rule for {selector}")
        return rule_match.group(1)

    def test_inline_script_is_valid_javascript(self):
        node_path = shutil.which("node")
        if node_path is None:
            self.skipTest("node is required for frontend script syntax validation")

        html = self._load_html()
        script_match = re.search(r"<script>(.*)</script>\s*</body>", html, re.S)
        self.assertIsNotNone(script_match, "expected inline <script> in frontend template")

        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as handle:
            handle.write(script_match.group(1))
            handle.flush()
            result = subprocess.run(
                [node_path, "--check", handle.name],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_answer_assets_are_width_constrained(self):
        html = self._load_html()

        card_rule = self._css_rule(html, ".card")
        self.assertIn("min-width: 0", card_rule)
        self.assertIn("max-width: 100%", card_rule)
        self.assertIn("overflow: hidden", card_rule)

        thumb_rule = self._css_rule(html, ".thumb")
        self.assertIn("max-width: 100%", thumb_rule)
        self.assertIn("height: auto", thumb_rule)

        table_shell_rule = self._css_rule(html, ".table-shell")
        self.assertIn("overflow-x: auto", table_shell_rule)
        self.assertIn("max-width: 100%", table_shell_rule)

    def test_tables_prefer_fit_before_horizontal_scroll(self):
        html = self._load_html()

        table_rule = self._css_rule(html, ".table-shell table")
        self.assertIn("max-width: 100%", table_rule)
        self.assertIn("table-layout: fixed", table_rule)

        cell_rule = self._css_rule(html, ".table-shell th, .table-shell td")
        self.assertIn("overflow-wrap: anywhere", cell_rule)

    def test_streaming_respects_manual_scroll_position(self):
        html = self._load_html()

        self.assertIn("const AUTO_SCROLL_THRESHOLD_PX = 80;", html)
        self.assertIn("function isNearBottom()", html)
        self.assertIn("function scrollChatToBottom", html)
        self.assertIn("if (force || shouldAutoScroll)", html)
        self.assertIn('chatArea.addEventListener("scroll"', html)

    def test_reasoning_panel_is_collapsible_and_above_answer(self):
        html = self._load_html()

        reasoning_rule = self._css_rule(html, ".reasoning-body")
        self.assertIn("font-style: italic", reasoning_rule)

        self.assertIn('<details id="reasoningPanel" class="reasoning-panel" hidden>', html)
        self.assertIn('<summary class="reasoning-label">模型思考过程</summary>', html)
        self.assertLess(html.index('id="reasoningPanel"'), html.index('id="answerBox"'))
        self.assertIn("reasoningPanel.open = false;", html)

    def test_jump_to_bottom_button_is_available_when_user_leaves_bottom(self):
        html = self._load_html()

        button_rule = self._css_rule(html, ".jump-bottom-btn")
        self.assertIn("position: absolute", button_rule)
        self.assertIn("bottom:", button_rule)
        self.assertIn("right:", button_rule)

        self.assertIn('id="jumpToBottomBtn"', html)
        self.assertIn("function updateJumpToBottomVisibility()", html)
        self.assertIn("jumpToBottomBtn.hidden = shouldAutoScroll || !canScroll;", html)
        self.assertIn('jumpToBottomBtn.addEventListener("click"', html)
        self.assertIn('scrollChatToBottom("smooth", true)', html)

    def test_submit_does_not_force_scroll_and_preserves_current_view_preference(self):
        html = self._load_html()

        self.assertNotIn('const isNewConversation = messagesArea.style.display === "none";', html)
        self.assertIn('const shouldFollowNewResponse = messagesArea.style.display !== "none" && isNearBottom();', html)
        self.assertIn("shouldAutoScroll = shouldFollowNewResponse;", html)
        self.assertNotIn('setTimeout(() => scrollChatToBottom("smooth", true), 50);', html)

    def test_jump_button_is_docked_near_input_and_toggle_copy_is_clear(self):
        html = self._load_html()

        button_rule = self._css_rule(html, ".jump-bottom-btn")
        self.assertIn("right: 24px", button_rule)
        self.assertIn("bottom: 28px", button_rule)

        self.assertIn('title="回到底部"', html)
        self.assertIn('title="开启后会基于检索结果生成最终回答；关闭后只展示检索证据"', html)
        self.assertIn("生成最终回答", html)
        self.assertIn('href="/ingest"', html)

    def test_enter_submission_respects_ime_composition(self):
        html = self._load_html()

        self.assertIn("let isQueryInputComposing = false;", html)
        self.assertIn('queryInput.addEventListener("compositionstart"', html)
        self.assertIn('queryInput.addEventListener("compositionend"', html)
        self.assertIn("if (event.isComposing || isQueryInputComposing || event.keyCode === 229)", html)

    def test_submit_button_can_abort_inflight_generation(self):
        html = self._load_html()

        button_rule = self._css_rule(html, ".send-btn.is-stop")
        self.assertIn("background: #8b3e22", button_rule)

        self.assertIn("let currentRequestAbortController = null;", html)
        self.assertIn("let isRequestInFlight = false;", html)
        self.assertIn("function syncSubmitButtonState()", html)
        self.assertIn("currentRequestAbortController?.abort();", html)
        self.assertIn("signal: requestAbortController.signal", html)
        self.assertIn("requestStatus.textContent = \"已停止\";", html)

    def test_mermaid_blocks_are_supported_in_answers(self):
        html = self._load_html()

        self.assertIn("https://cdn.jsdelivr.net/npm/mermaid", html)
        self.assertIn("function renderMermaidDiagrams", html)
        self.assertIn('answerBox.querySelectorAll("pre.mermaid")', html)
        self.assertIn("await mermaid.run", html)
        self.assertIn("renderMermaidDiagrams(answerBox)", html)


if __name__ == "__main__":
    unittest.main()
