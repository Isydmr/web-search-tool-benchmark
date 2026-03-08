import logging
from typing import Tuple
import json
from openai import OpenAI
from app.config import Config

logger = logging.getLogger(__name__)


class LLMModifier:
    def __init__(self):
        self.client = OpenAI()
        self.client.api_key = Config.OPENAI_API_KEY
        self.model = Config.DEFAULT_MODIFICATION_MODEL
        
    def modify_news(self, original_text: str) -> Tuple[str, str, str, str, str]:
        """Modify news content using LLM-based entity detection and modification.
        
        Returns:
            Tuple containing:
            - modified_text: The text with entity replaced
            - modification_type: Type of modification (entity or no_modification)
            - original_entity: The original entity that was replaced
            - modified_entity: The new entity that replaced the original
            - entity_type: The type of entity (PERSON, ORGANIZATION, DATE)
        """
        try:
            logger.info(f"Starting modification with text: {original_text[:100]}...")
            
            # Single prompt to detect and modify entity
            prompt = f"""Given this text: '{original_text}'
            Find a significant entity (person, organization, date, etc.) in the text and suggest a plausible but different value for it.
            The modification should be subtle but manipulative, changing the meaning or implications of the text.
            
            Return your response in this JSON format:
            {{
                "entity_type": "PERSON/ORGANIZATION/DATE/etc.",
                "original_entity": "the entity you found",
                "suggested_entity": "your suggested replacement",
                "explanation": "brief explanation of why this change is manipulative"
            }}
            
            Only return the JSON, nothing else."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            try:
                # Parse the JSON response
                result = json.loads(response.choices[0].message.content.strip())
                logger.info(f"LLM suggestion: {result}")
                
                # Validate the response has all required fields
                if all(k in result for k in ["entity_type", "original_entity", "suggested_entity"]):
                    original_entity = result["original_entity"]
                    new_entity = result["suggested_entity"]
                    entity_type = result["entity_type"]
                    
                    # Replace the entity in the text
                    modified_text = original_text.replace(original_entity, new_entity)
                    
                    if modified_text != original_text:  # Ensure the replacement actually happened
                        logger.info(f"Successfully modified {entity_type}: {original_entity} -> {new_entity}")
                        return modified_text, "entity", original_entity, new_entity, entity_type
                    else:
                        logger.warning("Replacement failed - original text not found")
                else:
                    logger.warning("Invalid JSON structure in LLM response")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            
            return original_text, "no_modification", "", "", ""
            
        except Exception as e:
            logger.error(f"Error modifying news: {str(e)}")
            return original_text, "no_modification", "", "", ""
