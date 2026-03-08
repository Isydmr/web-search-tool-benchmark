import logging
import ast
import json
import nltk
import numpy as np
import re
import time
from typing import Any
try:
    import torch
except Exception:  # pragma: no cover - broken local torch install
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:  # pragma: no cover - optional when torch stack unavailable
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from app.config import Config

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional when torch stack unavailable
    SentenceTransformer = None
from strsimpy.normalized_levenshtein import Levenshtein, NormalizedLevenshtein
from strsimpy.ngram import NGram

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception as exc:  # pragma: no cover - best effort download
        logger.warning("Could not download punkt tokenizer: %s", exc)

class Evaluation:
    def __init__(self):
        logger.info("Initializing Evaluation component...")
        if torch is not None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = None
        logger.info(f"Using device: {self.device}")

        self.model_name = Config.NLI_MODEL_NAME
        self.model_revision = Config.NLI_MODEL_REVISION
        self.tokenizer = None
        self.model = None
        try:
            logger.info("Loading model: %s @ %s", self.model_name, self.model_revision)
            if AutoTokenizer is None or AutoModelForSequenceClassification is None or self.device is None:
                raise RuntimeError("Torch/transformers stack unavailable")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=False,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                trust_remote_code=False,
            ).to(self.device)
        except Exception as exc:
            logger.warning("Falling back without NLI model: %s", exc)

        logger.info("Initializing OpenAI client...")
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        logger.info("Loading sentence transformer model...")
        self.sentence_transformer_name = Config.SENTENCE_TRANSFORMER_MODEL
        self.sentence_transformer_revision = Config.SENTENCE_TRANSFORMER_REVISION
        self.sentence_transformer_model = None
        try:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers unavailable")
            logger.info(
                "Loading sentence transformer: %s @ %s",
                self.sentence_transformer_name,
                self.sentence_transformer_revision,
            )
            self.sentence_transformer_model = SentenceTransformer(
                self.sentence_transformer_name,
                revision=self.sentence_transformer_revision,
            )
        except Exception as exc:
            logger.warning("Falling back without sentence transformer: %s", exc)
        
        logger.info("Initializing text similarity metrics...")
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.levenshtein = Levenshtein()
        self.bigram = NGram(2)
        
        logger.info("Evaluation component initialized successfully")

    def tokenize_to_sentences(self, text):
        return nltk.sent_tokenize(text)

    def _create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ):
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format

        try:
            return self.openai_client.chat.completions.create(**request)
        except Exception:
            if response_format is None:
                raise
            request.pop("response_format", None)
            return self.openai_client.chat.completions.create(**request)

    def _extract_json_object(self, raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {}

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        if start == -1:
            return {}

        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        return {}
        return {}

    def _coerce_likert_score(self, value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        return max(1, min(5, numeric))

    def _normalize_likert_score(self, value: int | None) -> float | None:
        if value is None:
            return None
        return round((value - 1) / 4, 4)

    def sentence_to_atomic_claims(self, sentence):
        SENTENCES_TO_CLAIMS_PROMPT = """
        Your task is to decompose the text into atomic claims.
        Let's define a function named decompose(input:str).
        The returned value should be a list of strings, where each string should be a context-independent claim, representing one fact.
        For example, if a user call decompose("Mary is a five-year old girl, she likes playing piano and she doesn't like cookies.").
        You should return a python list without any other words: 
        ["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
        Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

        decompose("{doc}")
        """
        prompt = SENTENCES_TO_CLAIMS_PROMPT.format(doc=sentence)
        attempt = 0
        max_retries = 3
        delay = 1  # seconds

        while attempt < max_retries:
            response = self.call_large_language_model(prompt)
            try:
                result = ast.literal_eval(response)
                return result  # Return the result if eval is successful
            except:
                attempt += 1
                print("Retrying sentence to claims")
                if attempt < max_retries:
                    time.sleep(delay)  # Wait for a bit before retrying
                else:
                    raise  # Re-raise the last exception if out of retries

    def find_most_relevant_sentence(self, sentence, sentences_list):
        vectorizer = TfidfVectorizer().fit_transform([sentence] + sentences_list)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
        most_relevant_index = cosine_similarities.argmax()
        return sentences_list[most_relevant_index]

    def apply_nli(self, premise, hypothesis):
        if self.tokenizer is None or self.model is None:
            logger.warning("NLI model unavailable; returning empty scores")
            return {}
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.softmax(outputs["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        return prediction

    def call_large_language_model(self, prompt):
        if not Config.OPENAI_API_KEY:
            return "[]"
        response = self._create_chat_completion(
            model=Config.DEFAULT_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def apply_g_eval(self, input_text, actual_output):
        if not Config.OPENAI_API_KEY:
            return {}

        prompt = f"""
You are grading whether a candidate response correctly restores a manipulated news article back to the reference article.

Reference article:
<reference>
{input_text}
</reference>

Candidate response:
<candidate>
{actual_output}
</candidate>

Score each criterion on a 1-5 integer scale where 1 is very poor and 5 is excellent.

Criteria:
1. factual_consistency: The candidate does not contradict or invent facts relative to the reference article.
2. correction_completeness: The candidate restores the reference facts instead of preserving manipulated claims.
3. content_preservation: The candidate keeps important supported information from the reference article instead of dropping it.
4. output_cleanliness: The candidate mostly returns the corrected article itself instead of meta commentary, bullet-point audits, or extra instructions.
5. overall: Holistic quality as a corrected article.

Return strict JSON only with this shape:
{{
  "factual_consistency": 1,
  "correction_completeness": 1,
  "content_preservation": 1,
  "output_cleanliness": 1,
  "overall": 1,
  "explanation": "One or two short sentences."
}}
""".strip()

        try:
            response = self._create_chat_completion(
                model=Config.DEFAULT_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict evaluation judge. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            parsed = self._extract_json_object(raw_content)
            if not parsed:
                logger.warning("G-Eval returned unparsable output")
                return {}
        except Exception as exc:
            logger.warning("G-Eval request failed: %s", exc)
            return {}

        factual_consistency = self._coerce_likert_score(parsed.get("factual_consistency"))
        correction_completeness = self._coerce_likert_score(parsed.get("correction_completeness"))
        content_preservation = self._coerce_likert_score(parsed.get("content_preservation"))
        output_cleanliness = self._coerce_likert_score(parsed.get("output_cleanliness"))
        overall = self._coerce_likert_score(parsed.get("overall"))

        if overall is None:
            available_scores = [
                score
                for score in (
                    factual_consistency,
                    correction_completeness,
                    content_preservation,
                    output_cleanliness,
                )
                if score is not None
            ]
            if available_scores:
                overall = int(round(sum(available_scores) / len(available_scores)))

        if overall is None:
            return {}

        explanation = str(parsed.get("explanation") or "").strip()
        if len(explanation) > 400:
            explanation = explanation[:397].rstrip() + "..."

        return {
            "method": "g_eval",
            "judge_model": Config.DEFAULT_JUDGE_MODEL,
            "factual_consistency": factual_consistency,
            "normalized_factual_consistency": self._normalize_likert_score(factual_consistency),
            "correction_completeness": correction_completeness,
            "normalized_correction_completeness": self._normalize_likert_score(correction_completeness),
            "content_preservation": content_preservation,
            "normalized_content_preservation": self._normalize_likert_score(content_preservation),
            "output_cleanliness": output_cleanliness,
            "normalized_output_cleanliness": self._normalize_likert_score(output_cleanliness),
            "overall": overall,
            "normalized_overall": self._normalize_likert_score(overall),
            "explanation": explanation,
        }

    def cosine_similarity(self, s1, s2):
        if self.sentence_transformer_model is None:
            return 0.0
        embedding = self.sentence_transformer_model.encode([s1, s2])
        v1 = embedding[0]
        v2 = embedding[1]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def calculate_semantic_similarity(self, preds, target):
        sts = []
        for s1, s2 in zip(preds, target):
            cos_sim = self.cosine_similarity(s1, s2)
            sts.append(cos_sim)
        return sts

    def word_overlap(self, s1, s2):
        La, Lb = [], []
        tokens1 = [token.strip() for token in s1.split() if token.strip() != ""]
        tokens2 = [token.strip() for token in s2.split() if token.strip() != ""]
        if len(tokens1) < len(tokens2):
            La, Lb = tokens2, tokens1
        else:
            La, Lb = tokens1, tokens2
        intersection = [token for token in La if token in Lb]
        return round(len(intersection)/len(La), 3)

    def lexical_distance_two_text(self, s1, s2):
        distance = {
            "edit_distance": self.levenshtein.distance(s1, s2),
            "norm_edit_distance": self.normalized_levenshtein.distance(s1, s2),
            "norm_edit_similarity": self.normalized_levenshtein.similarity(s1, s2),
            "bigram_distance": self.bigram.distance(s1, s2),
            "word_overlap": self.word_overlap(s1, s2)
        }
        return distance

    def calculate_text_distances(self, S1, S2):
        if len(S1) != len(S2):
            raise ValueError("S1 and S2 must contain the same number of items")
        distance = {}
        for i, (s1, s2) in enumerate(zip(S1, S2)):
            try:
                temp = self.lexical_distance_two_text(s1, s2)
                for k, v in temp.items():
                    if distance.get(k) is None:
                        distance[k] = []
                    distance[k].append(round(float(v), 2))
            except Exception as error:
                logger.warning("Example %d failed during distance calculation: %s", i, error)
        return distance 
