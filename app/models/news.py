from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float
from datetime import datetime
from .database import Base

class News(Base):
    __tablename__ = "news"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500))
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(500))
    published_date = Column(DateTime)
    modified_content = Column(Text)
    modification_type = Column(String(50))
    original_entity = Column(String(500))
    modified_entity = Column(String(500))
    entity_type = Column(String(50))  # PERSON, ORGANIZATION, or GPE
    created_at = Column(DateTime, default=datetime.utcnow)
    
class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    news_id = Column(Integer)
    llm_model = Column(String(100))
    llm_response = Column(Text)
    nli_scores = Column(JSON)
    geval_scores = Column(JSON)
    semantic_similarity = Column(Float)
    lexical_distance = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow) 