from __future__ import annotations

import unittest

from cultural_mood_tracker.sources.base import redact_url


class SourceSecurityTests(unittest.TestCase):
    def test_query_credentials_are_redacted(self) -> None:
        url = "https://example.test/search?page=1&api_key=secret-value&language=en"

        redacted = redact_url(url)

        self.assertNotIn("secret-value", redacted)
        self.assertEqual(
            redacted,
            "https://example.test/search?page=1&api_key=<redacted>&language=en",
        )


if __name__ == "__main__":
    unittest.main()
