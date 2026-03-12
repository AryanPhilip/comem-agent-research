"""Experience-memory loading and retrieval."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import faiss
import numpy as np

from memory.help_functions import CLIPMultimodalSimilarity, CLIPTextSimilarity
from memory.runtime import (
    MemoryBundle,
    MemoryQuery,
    MemoryRetriever,
    TrajectoryCondenser,
    TrajectoryRecord,
)

LOG_PATH = "memory"
os.makedirs(LOG_PATH, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join(LOG_PATH, "memory.log")))


class Memory:
    """Memory bank that supports text, continuous, and hybrid retrieval."""

    def __init__(
        self,
        training_data_path="training_data",
        agent=None,
        faiss_index_path=None,
        multimodal=False,
        bank_size=None,
    ):
        self.training_data_path = training_data_path
        self.multimodal = multimodal
        self.agent = agent
        self.bank_size = bank_size

        self.clip_similarity = (
            CLIPMultimodalSimilarity() if multimodal else CLIPTextSimilarity()
        )
        self.condenser = TrajectoryCondenser()

        self.records: list[TrajectoryRecord] = []
        self.embeddings = None
        self.faiss_index = None
        self.retriever: Optional[MemoryRetriever] = None

        if faiss_index_path is None:
            logger.info("Generating new memory index")
            self._load_all_conversations()
            self._create_faiss_index()
            if self.faiss_index is not None:
                os.makedirs("memory_index", exist_ok=True)
                self.save_index(
                    f"memory_index/{'multimodal' if multimodal else 'text'}_{self.faiss_index.ntotal}"
                )
        else:
            logger.info("Loading memory index from %s", faiss_index_path)
            self.load_index(faiss_index_path)

        self._rebuild_retriever()

    @property
    def memories(self) -> list[dict]:
        """Backward-compatible dict view."""
        return [record.to_dict() for record in self.records]

    def _rebuild_retriever(self) -> None:
        self.retriever = MemoryRetriever(
            records=self.records,
            faiss_index=self.faiss_index,
            embeddings=self.embeddings,
            clip_similarity=self.clip_similarity,
            multimodal=self.multimodal,
        )

    def _iter_trajectory_files(self):
        root = Path(self.training_data_path)
        if not root.exists():
            return

        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for domain_dir in sorted(dataset_dir.iterdir()):
                if not domain_dir.is_dir():
                    continue
                for model_dir in sorted(domain_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    for run_dir in sorted(model_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        for status in ("success", "positive", "negative"):
                            status_dir = run_dir / status
                            if not status_dir.is_dir():
                                continue
                            for file_path in sorted(status_dir.glob("*.jsonl")):
                                yield dataset_dir.name, domain_dir.name, model_dir.name, status, file_path

    def _derive_site(self, payload: dict, metadata: dict) -> str:
        if metadata.get("site"):
            return str(metadata["site"])
        if payload.get("conversation_summary", {}).get("site"):
            return str(payload["conversation_summary"]["site"])
        final_url = metadata.get("final_url") or payload.get("conversation_summary", {}).get("final_url", "")
        if final_url:
            return urlparse(final_url).netloc or final_url
        return ""

    def _derive_failure_tags(self, payload: dict, status: str, success: bool) -> list[str]:
        evaluation = payload.get("evaluation", {}).get("evaluation", {})
        failure_tags = payload.get("metadata", {}).get("failure_tags")
        if isinstance(failure_tags, list) and failure_tags:
            return [str(tag) for tag in failure_tags]

        tags = []
        error_type = evaluation.get("Error_Type")
        if error_type:
            tags.append(str(error_type))
        if evaluation.get("Redundant"):
            tags.append("redundant_steps")
        if not success and status == "negative":
            tags.append("negative_part")
        if evaluation.get("First_Error_Step"):
            tags.append("first_error_step")
        return tags

    def _load_all_conversations(self):
        logger.info("Loading all conversations from %s", self.training_data_path)
        records: list[TrajectoryRecord] = []

        for dataset, domain, model, status, file_path in self._iter_trajectory_files() or []:
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                logger.info("Failed to load %s: %s", file_path, exc)
                continue

            task_description = payload.get("task_description", "").strip()
            rounds = payload.get("rounds", [])
            total_rounds = payload.get("total_rounds", len(rounds))
            if not task_description or total_rounds < 1:
                continue

            metadata = payload.get("metadata", {})
            success = bool(
                metadata.get(
                    "success",
                    payload.get("conversation_summary", {}).get("success", status != "negative"),
                )
            )
            actions, images = self.condenser.extract_actions_and_images(rounds)
            base64_image = images[0] if images else None
            failure_tags = self._derive_failure_tags(payload, status=status, success=success)
            semantic_summary = self.condenser.summarize_actions(actions)
            reflection_summary = self.condenser.summarize_reflection(
                task_description=task_description,
                success=success,
                failure_tags=failure_tags,
                actions=actions,
            )

            record = TrajectoryRecord(
                file_path=str(file_path),
                task_description=task_description,
                prefixed_query=f"{dataset}_{domain}: {task_description}",
                dataset=metadata.get("dataset", dataset),
                domain=metadata.get("domain", domain),
                site=self._derive_site(payload, metadata),
                model=metadata.get("model", model),
                success=success,
                step_count=int(metadata.get("step_count", total_rounds)),
                timestamp=str(
                    metadata.get(
                        "timestamp",
                        payload.get("conversation_end", payload.get("conversation_start", "")),
                    )
                ),
                failure_tags=failure_tags,
                base64_image=base64_image,
                actions=actions,
                images=images,
                semantic_summary=semantic_summary,
                reflection_summary=reflection_summary,
                example_text=self.condenser.build_example_text(
                    task_description=task_description,
                    semantic_summary=semantic_summary,
                    reflection_summary=reflection_summary,
                    actions=actions,
                ),
            )
            records.append(record)

        if self.bank_size is not None:
            records = records[: self.bank_size]

        self.records = records
        logger.info("Loaded %d memory records", len(self.records))

    def _create_faiss_index(self):
        if not self.records:
            logger.info("No records available to build the memory index")
            return

        prefixed_queries = [record.prefixed_query for record in self.records]
        base64_images = [record.base64_image for record in self.records]

        if self.multimodal:
            self.embeddings = self.clip_similarity.get_multimodal_embeddings(
                prefixed_queries,
                base64_images,
            )
        else:
            self.embeddings = self.clip_similarity.get_text_embeddings(prefixed_queries)

        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings.astype("float32"))
        logger.info("Created FAISS index with %d vectors", self.faiss_index.ntotal)

    def build_query(
        self,
        current_question: str,
        current_image: Optional[str] = None,
        dataset: Optional[str] = None,
        domain: Optional[str] = None,
        site: Optional[str] = None,
        action_history: Optional[list[str]] = None,
        failure_state: str = "",
        current_url: str = "",
    ) -> MemoryQuery:
        return MemoryQuery(
            task=current_question,
            dataset=dataset or "",
            domain=domain or "",
            site=site or "",
            current_url=current_url or "",
            action_history=list(action_history or []),
            failure_state=failure_state,
            current_image=current_image,
        )

    def retrieve_records(
        self,
        query: MemoryQuery,
        similar_num: int = 3,
        dataset_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
    ) -> list[TrajectoryRecord]:
        if not self.retriever:
            return []
        results = self.retriever.retrieve(
            query=query,
            top_k=similar_num,
            dataset_filter=dataset_filter,
            domain_filter=domain_filter,
        )
        for result in results:
            logger.info(
                "Retrieval score %.4f (sim %.4f): %s",
                result.rerank_score,
                result.similarity,
                result.record.prefixed_query,
            )
        return [result.record for result in results]

    def build_memory_bundle(self, query: MemoryQuery, similar_num: int = 3) -> MemoryBundle:
        records = self.retrieve_records(
            query=query,
            similar_num=similar_num,
            dataset_filter=query.dataset or None,
            domain_filter=query.domain or None,
        )
        bundle = MemoryBundle(selected_records=records)
        for record in records:
            bundle.semantic_notes.append(record.semantic_summary)
            bundle.file_id_list.append(Path(record.file_path).stem)
            if record.actions:
                bundle.experience_texts.append(record.actions)
                bundle.experience_images.append(record.images[: len(record.actions)])
            bundle.prompt_text += record.example_text + "\n\n"
        bundle.prompt_text = bundle.prompt_text.strip()
        return bundle

    def retrieve_similar_conversations(
        self,
        current_question,
        current_image=None,
        model=None,
        similar_num=3,
    ):
        query = self.build_query(
            current_question=current_question,
            current_image=current_image,
        )
        return [record.file_path for record in self.retrieve_records(query, similar_num=similar_num)]

    def construct_experience_memory(
        self,
        current_question,
        agent,
        current_image=None,
        dataset=None,
        domain=None,
        similar_num=3,
        site=None,
        action_history=None,
        failure_state="",
        current_url="",
    ):
        query = self.build_query(
            current_question=current_question,
            current_image=current_image,
            dataset=dataset,
            domain=domain,
            site=site,
            action_history=action_history,
            failure_state=failure_state,
            current_url=current_url,
        )
        bundle = self.build_memory_bundle(query, similar_num=similar_num)
        return (
            bundle.prompt_text,
            bundle.experience_texts,
            bundle.experience_images,
            bundle.file_id_list,
        )

    def retrieve_similar_conversations_with_filter(
        self,
        current_question,
        current_image=None,
        dataset=None,
        domain=None,
        similar_num=3,
    ):
        query = self.build_query(
            current_question=current_question,
            current_image=current_image,
            dataset=dataset,
            domain=domain,
        )
        return [
            record.file_path
            for record in self.retrieve_records(
                query,
                similar_num=similar_num,
                dataset_filter=dataset,
                domain_filter=domain,
            )
        ]

    def get_available_datasets_and_domains(self):
        result = {}
        for record in self.records:
            result.setdefault(record.dataset, set()).add(record.domain)
        return {dataset: sorted(domains) for dataset, domains in result.items()}

    def save_index(self, filepath):
        if self.faiss_index is None:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        if self.embeddings is not None:
            np.save(f"{filepath}.embeddings.npy", self.embeddings)
        with open(f"{filepath}.json", "w", encoding="utf-8") as handle:
            json.dump({"records": [record.to_dict() for record in self.records]}, handle, indent=2)

    def load_index(self, filepath):
        try:
            self.faiss_index = faiss.read_index(f"{filepath}.faiss")
            embeddings_path = f"{filepath}.embeddings.npy"
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
            with open(f"{filepath}.json", "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.records = [TrajectoryRecord.from_dict(item) for item in payload.get("records", [])]
            if self.bank_size is not None:
                self.records = self.records[: self.bank_size]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[: self.bank_size]
                    dimension = self.embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    self.faiss_index.add(self.embeddings.astype("float32"))
        except Exception as exc:
            logger.info("Failed to load memory index %s: %s", filepath, exc)
            self._load_all_conversations()
            self._create_faiss_index()
        self._rebuild_retriever()
