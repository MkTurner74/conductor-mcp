"""
Tests for the Conductor-for-AI path — the decisions that are expensive to get
wrong, not the HTTP plumbing.

Run with:  python test_conductor_ai.py

Same rules as test_lora_pipeline.py: stdlib only (unittest, no pytest), so it
runs anywhere the server runs. Nothing here makes a network call.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import conductor_ai_client as ai
import conductor_ai_pipeline as pipeline


def run(coro):
    return asyncio.run(coro)


class StatusClassificationTests(unittest.TestCase):
    """
    The single most dangerous function here.

    A generation wrongly read as finished sends the run on to ingest images
    that do not exist; one wrongly read as still-running costs a poll. So an
    UNRECOGNISED status must never be treated as success — the Conductor
    green-job lesson, applied before it can bite twice.
    """

    def test_known_success_words(self):
        for s in ("success", "SUCCESS", "Completed", "finished", "done"):
            self.assertEqual(pipeline._classify(s), "success", s)

    def test_known_failure_words(self):
        for s in ("failed", "Cancelled", "ERROR", "timed_out"):
            self.assertEqual(pipeline._classify(s), "failed", s)

    def test_unknown_status_is_running_not_success(self):
        for s in ("provisioning", "queued", "warming_model", "", "42"):
            self.assertEqual(pipeline._classify(s), "running", s)


class MachineBlockTests(unittest.TestCase):
    """A half-filled machine block is worse than none: the service picks a sane
    default, and a wrong GPU name is a submission-time rejection."""

    def test_nothing_chosen_sends_nothing(self):
        self.assertIsNone(pipeline._machine())
        self.assertIsNone(pipeline._machine("", 0, ""))

    def test_gpu_type_only(self):
        self.assertEqual(pipeline._machine("A5000"), {"gpu": {"type": "A5000"}})

    def test_count_is_coerced_to_int(self):
        self.assertEqual(pipeline._machine("A5000", "2"), {"gpu": {"type": "A5000", "count": 2}})


class ImageCollectionTests(unittest.TestCase):
    """Outputs come back as nested {asset: {...}} rows, and the URL lives under
    either presigned_url or url depending on the endpoint. Reading the wrong
    key reports "no images" for a generation that produced plenty — exactly the
    relative_path-vs-name mistake that cost a round trip on the render side."""

    def _outputs(self, rows):
        return mock.patch.object(ai, "inference_outputs", mock.AsyncMock(return_value={"data": rows}))

    def test_reads_presigned_url_and_plain_url(self):
        rows = [
            {"asset": {"presigned_url": "https://x/1.png", "file_name": "1.png"}},
            {"asset": {"url": "https://x/2.png", "file_name": "2.png"}},
        ]
        with mock.patch.object(pipeline, "project_id", mock.AsyncMock(return_value="p1")), self._outputs(rows):
            got = run(pipeline.generation_images("inf1"))
        self.assertEqual(got["count"], 2)
        self.assertEqual([i["url"] for i in got["images"]], ["https://x/1.png", "https://x/2.png"])

    def test_skips_rows_with_no_url_and_non_images(self):
        rows = [
            {"asset": {"file_name": "3.png"}},                                  # no url at all
            {"asset": {"url": "https://x/log.txt", "file_name": "log.txt"}},    # not an image
            {"asset": {"url": "https://x/ok.jpg", "file_name": "ok.jpg"}},
        ]
        with mock.patch.object(pipeline, "project_id", mock.AsyncMock(return_value="p1")), self._outputs(rows):
            got = run(pipeline.generation_images("inf1"))
        self.assertEqual(got["count"], 1)
        self.assertEqual(got["images"][0]["file_name"], "ok.jpg")

    def test_unnamed_file_is_kept(self):
        """A url with no file_name is still an image — dropping it would lose
        output on a service that does not have to name its files."""
        rows = [{"asset": {"url": "https://x/anon", "file_name": ""}}]
        with mock.patch.object(pipeline, "project_id", mock.AsyncMock(return_value="p1")), self._outputs(rows):
            got = run(pipeline.generation_images("inf1"))
        self.assertEqual(got["count"], 1)


class ModelResolutionTests(unittest.TestCase):
    # The real registered names, read off the live account 2026-09-08. Note
    # that NONE of them contain the string "sdxl" or the bare word "flux" —
    # that is the whole reason MODEL_ALIASES exists, so the fixture has to keep
    # the awkward real names rather than tidy ones.
    MODELS = [
        {"id": "m-1", "name": "FLUX.1-schnell-t5fp8-07182025", "model_name": "FLUX.1-schnell-t5fp8", "model_family": ""},
        {"id": "m-2", "name": "stable-diffusion-xl-base-1.0-07212025", "model_name": "stable-diffusion-xl-base-1.0", "model_family": ""},
        {"id": "m-3", "name": "stable-diffusion-3.5-large-12012025", "model_name": "stable-diffusion-3.5-large", "model_family": ""},
    ]

    def _listing(self):
        return mock.patch.object(pipeline, "list_models",
                                 mock.AsyncMock(return_value={"project_id": "p1", "models": self.MODELS}))

    def test_blank_takes_the_first(self):
        with self._listing():
            self.assertEqual(run(pipeline.resolve_model_id("")), "m-1")

    def test_exact_id_wins(self):
        with self._listing():
            self.assertEqual(run(pipeline.resolve_model_id("m-2")), "m-2")

    def test_short_names_resolve_through_the_alias_map(self):
        """"sdxl" must reach stable-diffusion-xl-base-1.0. Falling through to
        the first model instead is the silent failure this guards: a different
        model, generating perfectly good wrong pictures."""
        with self._listing():
            self.assertEqual(run(pipeline.resolve_model_id("sdxl")), "m-2")
            self.assertEqual(run(pipeline.resolve_model_id("SDXL")), "m-2")
            self.assertEqual(run(pipeline.resolve_model_id("flux")), "m-1")
            self.assertEqual(run(pipeline.resolve_model_id("sd35-large")), "m-3")

    def test_full_and_partial_real_names(self):
        with self._listing():
            self.assertEqual(run(pipeline.resolve_model_id("stable-diffusion-xl-base-1.0-07212025")), "m-2")
            self.assertEqual(run(pipeline.resolve_model_id("stable-diffusion-3.5-large")), "m-3")

    def test_no_match_names_the_options(self):
        """The error has to say what WAS available, or the next step is guessing."""
        with self._listing():
            with self.assertRaises(ai.ConductorAIError) as ctx:
                run(pipeline.resolve_model_id("dall-e"))
        self.assertIn("stable-diffusion-xl-base-1.0-07212025", str(ctx.exception))

    def test_no_models_at_all_says_so(self):
        with mock.patch.object(pipeline, "list_models",
                               mock.AsyncMock(return_value={"project_id": "p1", "models": []})):
            with self.assertRaises(ai.ConductorAIError) as ctx:
                run(pipeline.resolve_model_id(""))
        self.assertIn("no base models", str(ctx.exception).lower())


class SubmissionShapeTests(unittest.TestCase):
    """What actually goes on the wire. lora_details must be ABSENT, not empty,
    for a base-model-only generation — an empty list is a different request."""

    def _submit(self):
        return mock.patch.object(ai, "submit_inference",
                                 mock.AsyncMock(return_value={"id": "inf-9", "latest_status": "queued"}))

    def _common(self):
        return (
            mock.patch.object(pipeline, "project_id", mock.AsyncMock(return_value="p1")),
            mock.patch.object(pipeline, "resolve_model_id", mock.AsyncMock(return_value="m-1")),
        )

    def test_no_lora_sends_no_lora_details(self):
        a, b = self._common()
        with a, b, self._submit() as sub:
            got = run(pipeline.submit_generation(prompt="a car"))
        self.assertTrue(got["ok"])
        self.assertEqual(got["inference_id"], "inf-9")
        self.assertIsNone(sub.call_args.args[3] if len(sub.call_args.args) > 3 else sub.call_args.kwargs.get("lora_details"))

    def test_lora_carries_trigger_and_weight(self):
        a, b = self._common()
        with a, b, self._submit() as sub:
            run(pipeline.submit_generation(prompt="sks car", lora_id="l-1", trigger_word="sks", weight=1.3))
        details = sub.call_args.args[3] if len(sub.call_args.args) > 3 else sub.call_args.kwargs["lora_details"]
        self.assertEqual(details, [{"id": "l-1", "weight": 1.3, "trigger_word": "sks"}])

    def test_settings_are_numbers_not_strings(self):
        """Every value arrives from the canvas as a string; the API takes ints
        and floats, and a quoted "30" is a 400 at submission time."""
        a, b = self._common()
        with a, b, self._submit() as sub:
            run(pipeline.submit_generation(prompt="a car", steps="20", seed="7",
                                           width="512", height="768", prompt_adherence="4"))
        settings = sub.call_args.args[2] if len(sub.call_args.args) > 2 else sub.call_args.kwargs["settings"]
        self.assertEqual(settings["num_steps"], 20)
        self.assertEqual(settings["seed"], 7)
        self.assertEqual(settings["width"], 512)
        self.assertEqual(settings["height"], 768)
        self.assertEqual(settings["prompt_adherence"], 4.0)

    def test_empty_prompt_is_refused_before_spending(self):
        got = run(pipeline.submit_generation(prompt="   "))
        self.assertFalse(got["ok"])


class AdherenceScaleTests(unittest.TestCase):
    """This API's prompt_adherence is 1-4 (default 3). The kohya inference tool
    next to it takes 7.5+ classifier-free guidance. Carrying the familiar
    number across sends every generation a value past the model's maximum, so
    the default is pinned by a test."""

    def test_default_is_on_this_apis_scale(self):
        self.assertEqual(pipeline.DEFAULT_ADHERENCE, 3.0)
        self.assertLessEqual(pipeline.DEFAULT_ADHERENCE, 4.0)


class AuthConfigTests(unittest.TestCase):
    def test_key_is_stripped(self):
        """A trailing \\r from a Windows-generated env value already killed one
        deployment at the edge."""
        with mock.patch.dict("os.environ", {"CONDUCTOR_AI_API_KEY": "abc123\r\n"}):
            self.assertEqual(ai.api_key(), "abc123")
            self.assertTrue(ai.configured())

    def test_url_gets_a_scheme(self):
        with mock.patch.dict("os.environ", {"CONDUCTOR_AI_API_URL": "backend.example.com"}):
            self.assertEqual(ai.api_url(), "https://backend.example.com")

    def test_header_styles(self):
        self.assertEqual(ai._header("raw", "k"), {"Authorization": "k"})
        self.assertEqual(ai._header("bearer", "k"), {"Authorization": "Bearer k"})

    def test_missing_key_says_which_key(self):
        with mock.patch.dict("os.environ", {"CONDUCTOR_AI_API_KEY": ""}):
            with self.assertRaises(ai.ConductorAIError) as ctx:
                run(ai._request("GET", "/core/v1/projects"))
        self.assertIn("CONDUCTOR_AI_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
