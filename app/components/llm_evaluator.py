import logging
from typing import Dict
from app.components.evaluation import Evaluation
from app.utils.benchmark_scoring import compute_surface_alignment_score

logger = logging.getLogger(__name__)

class LLMEvaluator:
    def __init__(self):
        self.evaluator = Evaluation()
        
    def _evaluate_response(self, original_text: str, modified_text: str) -> Dict:
        """Evaluate the modified text against the original using various metrics.
        
        Args:
            original_text: The original news article text
            modified_text: The modified version of the text
            
        Returns:
            Dictionary containing evaluation metrics:
            - nli_scores (entailment, contradiction)
            - geval_scores (factuality)
            - semantic_similarity
            - lexical_distance
        """
        logger.info("Starting evaluation...")
        
        try:
            # Get NLI scores
            logger.info("Getting NLI scores...")
            nli_scores = self.evaluator.apply_nli(original_text, modified_text)
            logger.info(f"NLI scores: {nli_scores}")
            
            # Get G-Eval scores
            logger.info("Getting G-Eval scores...")
            geval_scores = self.evaluator.apply_g_eval(original_text, modified_text)
            logger.info(f"G-Eval scores: {geval_scores}")
            
            # Calculate semantic similarity
            logger.info("Calculating semantic similarity...")
            semantic_similarity = self.evaluator.cosine_similarity(original_text, modified_text)
            logger.info(f"Semantic similarity: {semantic_similarity}")
            
            # Calculate lexical distance
            logger.info("Calculating lexical distance...")
            lexical_distance = self.evaluator.lexical_distance_two_text(original_text, modified_text)
            logger.info(f"Lexical distance: {lexical_distance}")

            surface_alignment_score = compute_surface_alignment_score(
                {
                    "nli_scores": nli_scores,
                    "geval_scores": geval_scores,
                    "semantic_similarity": semantic_similarity,
                    "lexical_distance": lexical_distance,
                }
            )
            logger.info(f"Surface alignment score: {surface_alignment_score}")
            
            result = {
                "nli_scores": nli_scores,
                "geval_scores": geval_scores,
                "semantic_similarity": semantic_similarity,
                "lexical_distance": lexical_distance,
                "surface_alignment_score": surface_alignment_score,
            }
            logger.info(f"Evaluation complete. Result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return {
                "nli_scores": {},
                "geval_scores": {},
                "semantic_similarity": 0,
                "lexical_distance": {},
                "surface_alignment_score": 0,
            }
