from __future__ import annotations

import unittest

from tools.vm.egress_probe import redact_url


class EgressProbeRedactionTests(unittest.TestCase):
    def test_redacts_userinfo_query_and_fragment(self) -> None:
        self.assertEqual(
            redact_url(
                "https://agent:secret@example.com:8443/download/file.pdf"
                "?token=private&part=1#fragment"
            ),
            "https://<redacted>@example.com:8443/download/file.pdf?<redacted>",
        )

    def test_preserves_credential_free_url(self) -> None:
        self.assertEqual(
            redact_url("https://example.com/download/file.pdf"),
            "https://example.com/download/file.pdf",
        )

    def test_malformed_authority_is_not_echoed(self) -> None:
        self.assertEqual(
            redact_url("https://user:secret@example.com:bad/path"),
            "<redacted-url>",
        )


if __name__ == "__main__":
    unittest.main()
