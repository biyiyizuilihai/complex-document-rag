import os
import tempfile
import unittest

from data_models import ExternalReference, ImageDescription
from complex_document_rag.pipeline_utils import (
    build_description_payload,
    list_document_files,
    list_image_filenames,
    make_image_id,
    resolve_source_image_path,
)


class PipelineUtilsTests(unittest.TestCase):
    def test_list_image_filenames_ignores_hidden_and_non_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, ".gitkeep"), "w").close()
            open(os.path.join(tmpdir, "diagram.png"), "w").close()
            open(os.path.join(tmpdir, "notes.txt"), "w").close()

            self.assertEqual(list_image_filenames(tmpdir), ["diagram.png"])
            self.assertEqual(make_image_id(0), "img_00000")

    def test_build_description_payload_keeps_source_image_metadata(self):
        with tempfile.TemporaryDirectory() as project_root:
            images_dir = os.path.join(project_root, "images")
            os.makedirs(images_dir)
            image_path = os.path.join(images_dir, "diagram.png")
            open(image_path, "w").close()

            desc = ImageDescription(
                summary="登录流程",
                detailed_description="用户登录后系统校验令牌。",
                nodes=["用户登录", "令牌校验"],
                external_references=[
                    ExternalReference(target="接口文档", context="节点引用接口文档")
                ],
                tags=["登录"],
            )

            payload = build_description_payload(desc, image_path, project_root)

            self.assertEqual(payload["source_image_path"], os.path.join("images", "diagram.png"))
            self.assertEqual(payload["source_image_filename"], "diagram.png")
            self.assertEqual(payload["summary"], "登录流程")
            self.assertEqual(payload["external_references"][0]["target"], "接口文档")

    def test_resolve_source_image_path_prefers_stored_metadata(self):
        with tempfile.TemporaryDirectory() as project_root:
            images_dir = os.path.join(project_root, "images")
            os.makedirs(images_dir)
            image_path = os.path.join(images_dir, "diagram.png")
            open(image_path, "w").close()

            desc_data = {
                "source_image_path": os.path.join("images", "diagram.png"),
                "source_image_filename": "diagram.png",
            }

            resolved_path = resolve_source_image_path("img_00000", desc_data, project_root)

            self.assertEqual(resolved_path, os.path.abspath(image_path))

    def test_list_document_files_ignores_hidden_placeholders(self):
        with tempfile.TemporaryDirectory() as docs_dir:
            open(os.path.join(docs_dir, ".gitkeep"), "w").close()
            open(os.path.join(docs_dir, "sample.md"), "w").close()

            self.assertEqual(list_document_files(docs_dir), ["sample.md"])


if __name__ == "__main__":
    unittest.main()
