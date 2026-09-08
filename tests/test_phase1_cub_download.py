from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ibkd_seg.phase1.cub_data import _download


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        content_range: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def getcode(self) -> int:
        return self.status


class Phase1CubDownloadTest(unittest.TestCase):
    def test_early_eof_is_resumed_with_a_valid_range_response(self) -> None:
        payload = b"0123456789"
        responses = [
            FakeResponse(payload[:4], status=200),
            FakeResponse(
                payload[4:],
                status=206,
                content_range="bytes 4-9/10",
            ),
        ]
        requests = []

        def fake_urlopen(request, *, timeout):
            requests.append((request, timeout))
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "archive.tgz"
            with (
                mock.patch(
                    "ibkd_seg.phase1.cub_data.urllib.request.urlopen",
                    side_effect=fake_urlopen,
                ),
                mock.patch("ibkd_seg.phase1.cub_data.time.sleep"),
            ):
                _download(
                    "https://example.test/archive.tgz",
                    destination,
                    expected_md5=hashlib.md5(payload).hexdigest(),
                    expected_bytes=len(payload),
                    retries=2,
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".tgz.part").exists())

        self.assertIsNone(requests[0][0].get_header("Range"))
        self.assertEqual(requests[1][0].get_header("Range"), "bytes=4-")

    def test_complete_existing_partial_is_verified_without_network(self) -> None:
        payload = b"already complete"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "archive.tgz"
            partial = destination.with_suffix(".tgz.part")
            partial.write_bytes(payload)
            with mock.patch(
                "ibkd_seg.phase1.cub_data.urllib.request.urlopen"
            ) as urlopen:
                _download(
                    "https://example.test/archive.tgz",
                    destination,
                    expected_md5=hashlib.md5(payload).hexdigest(),
                    expected_bytes=len(payload),
                )

            urlopen.assert_not_called()
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_ignored_resume_range_does_not_append_duplicate_bytes(self) -> None:
        payload = b"0123456789"
        requests = []
        responses = [
            FakeResponse(payload, status=200),
            FakeResponse(
                payload[4:],
                status=206,
                content_range="bytes 4-9/10",
            ),
        ]

        def fake_urlopen(request, *, timeout):
            requests.append(request)
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "archive.tgz"
            partial = destination.with_suffix(".tgz.part")
            partial.write_bytes(payload[:4])
            with (
                mock.patch(
                    "ibkd_seg.phase1.cub_data.urllib.request.urlopen",
                    side_effect=fake_urlopen,
                ),
                mock.patch("ibkd_seg.phase1.cub_data.time.sleep"),
            ):
                _download(
                    "https://example.test/archive.tgz",
                    destination,
                    expected_md5=hashlib.md5(payload).hexdigest(),
                    expected_bytes=len(payload),
                    retries=2,
                )

            self.assertEqual(destination.read_bytes(), payload)

        self.assertEqual(requests[0].get_header("Range"), "bytes=4-")
        self.assertEqual(requests[1].get_header("Range"), "bytes=4-")


if __name__ == "__main__":
    unittest.main()
