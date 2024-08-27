from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from enum import Enum


class StateAction(str, Enum):
    COORDINATE = "Coordinate"
    SCORE_QUERY = "Query"
    BUILD_QUESTION = "BuildQuestion"
    CHALLENGE = "Challenge"
    WEB_SEARCH = "WebSearch"
    QUERY_LLM = "QueryLLM"
    SOURCE_APPROVE = "SourceApprove"


class MessageType(str, Enum):
    PROCESS = "process"
    REFINED_QUESTION = "refined queston"


class EduModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Message(EduModel):
    message_type: MessageType
    state_action: StateAction
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class SeedQuestions(EduModel):
    questions: List[str] = Field(default_factory=list)


class Question(EduModel):
    question_text: str
    priority: int = 1


class QuestionsScore(EduModel):
    score: int
    score_comment: str


class Questions(EduModel):
    questions: list[Question]
    questions_score: QuestionsScore | None


class RefinedQuestions(EduModel):
    refined_questions: list[str]
    comment_to_the_original_question: str


class Query(EduModel):
    query_text: str


class WebSearchResult(EduModel):
    snippet: str
    title: str
    link: str


class WebSearchResults(EduModel):
    web_search_results: list[WebSearchResult]


class Sources(EduModel):
    links: list[str]
    query_tuning_suggestion: str | None
    removed_links_explaination: str | None


class WebSearchError(EduModel):
    message: str


class Prompt(EduModel):
    prompt_text: str


class Answer(EduModel):
    answer_text: str


class ConversationState(EduModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    questions: Optional[Questions] = None
    refined_questions: Optional[RefinedQuestions] = None
    query: Optional[Query] = None
    web_search_results: Optional[WebSearchResults] = None
    sources: Optional[Sources] = None
    prompt: Optional[Prompt] = None
    answer: Optional[Answer] = None
    messages: List[Message] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now())
    last_action: StateAction


class QuestionsRequest(BaseModel):
    questions_list: list[str] | str


def create_initial_state(conversation_id: str) -> ConversationState:
    """
    Create initial state with seeded questions
    """
    state = ConversationState(
        conversation_id=conversation_id, last_action=StateAction.BUILD_QUESTION
    )
    return state
