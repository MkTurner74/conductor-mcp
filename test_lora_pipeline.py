"""
Tests for the parts of the LoRA pipeline that cost money or lose data.

Run with:  python test_lora_pipeline.py

Deliberately stdlib-only (unittest, no pytest) so it runs anywhere the server
runs, including inside the deploy container, with nothing extra installed.

Everything here is a bug that actually happened, or one that would only ever
show up on somebody else's machine.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import lora_pipeline as lp


class DefaultWorkdirTests(unittest.TestCase):
    """
    The staging path decides whether the render node can read its own dataset.

    Job 00014 reported success and produced no weights because the container
    running the submission was root: the dataset staged to /root/.conductor-lora,
    uploaded to /root/... on the node, and the task -- which runs as `conductor`
    -- got PermissionError on a directory it could not enter. The identical code
    from a laptop worked. Nothing about it was visible in the job status.
    """

    def test_root_home_never_used(self):
        for home in ("/root", "/root/", "/"):
            with mock.patch("os.path.expanduser", return_value=home), \
                 mock.patch.dict("os.environ", {}, clear=False):
                mock.patch.dict("os.environ", {"LORA_WORKDIR": ""}).start()
                import os
                os.environ.pop("LORA_WORKDIR", None)
                got = lp.default_workdir()
                self.assertFalse(
                    got.startswith("/root"),
                    f"home={home!r} produced {got!r} -- the render node cannot read /root",
                )

    def test_normal_home_is_used(self):
        import os
        os.environ.pop("LORA_WORKDIR", None)
        with mock.patch("os.path.expanduser", return_value="/home/someone"):
            self.assertEqual(lp.default_workdir(), os.path.join("/home/someone", ".conductor-lora"))

    def test_env_override_wins(self):
        import os
        with mock.patch.dict(os.environ, {"LORA_WORKDIR": "/somewhere/else"}):
            self.assertEqual(lp.default_workdir(), "/somewhere/else")


class NodePathTests(unittest.TestCase):
    """Conductor strips the drive letter; getting it wrong fails only on the node."""

    def test_windows_drive_letter_stripped(self):
        self.assertEqual(
            lp.node_path(r"C:\Users\mktur\.conductor-lora\x\dataset"),
            "/Users/mktur/.conductor-lora/x/dataset",
        )

    def test_posix_path_unchanged(self):
        self.assertEqual(lp.node_path("/Users/samsyn/.conductor-lora/x"), "/Users/samsyn/.conductor-lora/x")


class OutputNameTests(unittest.TestCase):
    """
    Conductor names output files `relative_path`. Filtering on `name` or `path`
    silently matches nothing, so a job that produced a perfectly good
    .safetensors reports "no weights". Cost a confused round trip on job 00013.
    """

    def test_reads_relative_path(self):
        self.assertEqual(lp.output_name({"relative_path": "model.safetensors"}), "model.safetensors")

    def test_falls_back_without_inventing(self):
        self.assertEqual(lp.output_name({}), "")


class ProvenanceFieldTests(unittest.TestCase):
    """
    A Portal that lacks one of our fields must not lose the other seven.

    Writing an unattached field returns 400 for the WHOLE metadata call, so a
    missing `prov_created_by` would silently cost the prompt, the model and the
    job id too. It is skipped and reported instead -- and starts working by
    itself the moment somebody adds the field in the admin UI.
    """

    def _run(self, coro):
        return asyncio.run(coro)

    def test_missing_field_is_skipped_not_fatal(self):
        with mock.patch.object(lp, "provenance_group_fields",
                               new=mock.AsyncMock(return_value={"prov_kind", "prov_prompt2"})):
            fields, skipped = self._run(lp._writable_provenance_fields({
                "provenance_kind": "generated_image",
                "prompt": "a car",
                "created_by": "mark@example.com",
            }))
        self.assertEqual({f["name"] for f in fields}, {"prov_kind", "prov_prompt2"})
        self.assertEqual(len(skipped), 1)
        self.assertIn("created_by", skipped[0])

    def test_unmapped_key_raises_rather_than_vanishing(self):
        # An unmapped key is OUR bug. A provenance record that quietly loses a
        # field is worse than one that fails loudly.
        with mock.patch.object(lp, "provenance_group_fields",
                               new=mock.AsyncMock(return_value={"prov_kind"})):
            with self.assertRaises(KeyError):
                self._run(lp._writable_provenance_fields({"not_a_real_key": "x"}))

    def test_unreadable_group_attempts_the_write(self):
        # An empty available-set means we could not read the group at all. A
        # transient read failure must not silently strip every field.
        with mock.patch.object(lp, "provenance_group_fields",
                               new=mock.AsyncMock(return_value=set())):
            fields, skipped = self._run(lp._writable_provenance_fields({"prompt": "a car"}))
        self.assertEqual(len(fields), 1)
        self.assertEqual(skipped, [])

    def test_generated_images_record_the_full_story(self):
        # Model, LoRA name, trigger word, prompt, user, job, source -- everything
        # needed to answer "what made this, from what, and who asked" without
        # leaving the asset panel.
        with mock.patch.object(lp, "provenance_group_fields",
                               new=mock.AsyncMock(return_value=set(lp.PROVENANCE_FIELD_IDS.values()))):
            fields, skipped = self._run(lp._writable_provenance_fields({
                "provenance_kind": "generated_image",
                "label": "Aston Martin F1 Livery v1",
                "trigger_word": "amf1",
                "prompt": "amf1 livery, studio lighting",
                "base_model": "sdxl1-kohya",
                "job_id": "00015",
                "source_asset_ids": "VX-4422",
                "created_by": "mark@example.com",
            }))
        self.assertEqual(skipped, [])
        by_name = {f["name"]: f["value"] for f in fields}
        self.assertEqual(by_name["prov_base_model"], "sdxl1-kohya")
        self.assertEqual(by_name["prov_label"], "Aston Martin F1 Livery v1")
        self.assertEqual(by_name["prov_prompt2"], "amf1 livery, studio lighting")
        # The "3" is load-bearing: prov_created_by and prov_created_by2 exist on
        # the group as burnt earlier attempts. This test fails if anyone tidies
        # the mapping back to the bare name.
        self.assertEqual(by_name["prov_created_by3"], "mark@example.com")
        self.assertEqual(by_name["prov_source_assets"], "VX-4422")


class TrainingFailureDetailTests(unittest.TestCase):
    """
    A job can report success and still have failed. Saying "no .safetensors"
    points at the wrong layer; the cause is in the log we already fetched.
    """

    def test_extracts_the_real_exception(self):
        log = (
            "FutureWarning: something deprecated\n"
            "2026-08-31 14:50:27 INFO     Using DreamBooth method.\n"
            "Traceback (most recent call last):\n"
            "  File \"train_network.py\", line 521, in train\n"
            "PermissionError: [Errno 13] Permission denied: '/root/.conductor-lora/x/dataset'\n"
        )

        async def fake_get(url, **kw):  # noqa: ANN001
            raise AssertionError("should not be called")

        class FakeResponse:
            def __init__(self, text): self.text = text
            def raise_for_status(self): return None

        class FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url):
                return FakeResponse("train_rc=1" if "status" in url else log)

        outputs = {"downloads": [{"files": [
            {"relative_path": "train_status.txt", "url": "https://x/status"},
            {"relative_path": "train.log", "url": "https://x/log"},
        ]}]}

        with mock.patch.object(lp.httpx, "AsyncClient", FakeClient):
            detail = asyncio.run(lp.training_failure_detail(outputs))

        self.assertIn("train_rc=1", detail)
        self.assertIn("PermissionError", detail)

    def test_says_nothing_when_there_is_nothing_to_say(self):
        self.assertIsNone(asyncio.run(lp.training_failure_detail({"downloads": []})))


class PromptQuotingTests(unittest.TestCase):
    """
    A prompt has spaces. That is the entire bug.

    Job 00015 -- the first inference run ever -- reported SUCCESS and produced
    no images. The prompt was interpolated into a command already wrapped in
    bash -c '...', where a backslash is literal, so the node received
    `--prompt \\"amf1 livery, studio...` and argparse died on the leftovers.
    """

    def test_prompt_is_not_interpolated_into_the_command(self):
        self.assertNotIn("{prompt}", lp.INFER_COMMAND_TEMPLATE,
                         "the prompt must not be substituted into a shell string")
        self.assertIn('--prompt "$LORA_PROMPT"', lp.INFER_COMMAND_TEMPLATE)

    def test_template_still_formats_without_a_prompt_argument(self):
        # If the placeholder came back, this raises KeyError -- which is a
        # better failure than a job that succeeds and generates nothing.
        cmd = lp.INFER_COMMAND_TEMPLATE.format(
            launcher="python", lora_path="/x/l.safetensors", output_path="/out",
            count=4, width=1024, height=1024, steps=30, seed=42,
            strength=1.25, guidance=8.0,
        )
        self.assertIn('--prompt "$LORA_PROMPT"', cmd)
        # The knobs that decide whether the LoRA is actually visible in the
        # result. Missing --network_mul means kohya's polite 1.0 default, which
        # is what "it is not sticking to the LoRA" looks like.
        self.assertIn("--network_mul 1.25", cmd)
        self.assertIn("--scale 8.0", cmd)

    def test_punctuation_survives(self):
        # Quotes used to be stripped because they broke the shell. They no
        # longer can, so a real prompt keeps its meaning and its punctuation.
        raw = chr(10).join(['a "quoted" prompt', "  with a break"])
        self.assertEqual(" ".join(raw.split()), 'a "quoted" prompt with a break')

class InferenceDiagnosisTests(unittest.TestCase):
    """The diagnostic has to know inference filenames, not just training ones."""

    def test_reads_infer_log_and_argparse_error(self):
        # argparse does not raise -- it prints usage and exits 2. That is what
        # a mis-quoted prompt looks like, so the parser has to recognise it
        # alongside the exceptions or the cause stays invisible.
        log = chr(10).join([
            "usage: sdxl_gen_img.py [-h] [--ckpt CKPT]",
            "sdxl_gen_img.py: error: unrecognized arguments: livery, studio lighting",
        ])

        class FakeResponse:
            def __init__(self, text): self.text = text
            def raise_for_status(self): return None

        class FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url):
                return FakeResponse("infer_rc=2" if "status" in url else log)

        outputs = {"downloads": [{"files": [
            {"relative_path": "infer_status.txt", "url": "https://x/status"},
            {"relative_path": "infer.log", "url": "https://x/log"},
        ]}]}

        with mock.patch.object(lp.httpx, "AsyncClient", FakeClient):
            detail = asyncio.run(lp.training_failure_detail(outputs))

        self.assertIn("infer_rc=2", detail)
        self.assertIn("unrecognized arguments", detail)

if __name__ == "__main__":
    unittest.main(verbosity=2)
