from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from enum import Enum


class StateAction(str, Enum):
    COORDINATE = "Coordinate"
    BUILD_QUESTION = "BuildQuestion"
    CHALLENGE = "Challenge"
    WEB_SEARCH = "WebSearch"
    QUERY_LLM = "QueryLLM"


class MessageType(str, Enum):
    PROCESS = "process"
    REFINED_QUESTION = "refined queston"


class Message(BaseModel):
    message_type: MessageType
    state_action: StateAction
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class SeedQuestions(BaseModel):
    questions: List[str] = Field(default_factory=list)


class Question(BaseModel):
    question_text: str


class RefinedQuestions(BaseModel):
    refined_questions: list[str]
    comment_to_the_original_question: str


class Query(BaseModel):
    query_text: str


class WebSearchResult(BaseModel):
    data: str


class Prompt(BaseModel):
    prompt_text: str


class Answer(BaseModel):
    answer_text: str


class ConversationState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    seed_questions: SeedQuestions = Field(default_factory=SeedQuestions)
    question: Optional[Question] = None
    refined_questions: Optional[RefinedQuestions] = None
    query: Optional[Query] = None
    some_data: Optional[WebSearchResult] = None
    prompt: Optional[Prompt] = None
    answer: Optional[Answer] = None
    messages: List[Message] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now())
    last_action: StateAction


def create_initial_state(
    conversation_id: str, initial_questions: List[str]
) -> ConversationState:
    """
    Create initial state with seeded questions
    """
    state = ConversationState(
        conversation_id=conversation_id, last_action=StateAction.BUILD_QUESTION
    )
    state.seed_questions = SeedQuestions(questions=initial_questions)
    state.messages.append(
        Message(
            message_type=MessageType.REFINED_QUESTION,
            state_action=StateAction.BUILD_QUESTION,
            content="Can you write an essay about Napoleon?",
        )
    )
    return state
