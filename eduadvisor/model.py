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
    WEB_SCRAPE = "WebScrape"
    ANSWER_QUESTION = "AnswerQuestion"


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

    def immutable_copy_questions_score(
        self, questions_score: QuestionsScore | None
    ) -> "Questions":
        return self.model_copy(update={"questions_score": questions_score}, deep=True)


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


class WebData(EduModel):
    link: str
    data: str


class WebDataCollection(EduModel):
    web_data_collection: list[WebData]
    web_data_retrival_errors: list[str]


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
    web_data_collection: Optional[WebDataCollection] = None
    sources: Optional[Sources] = None
    prompt: Optional[Prompt] = None
    answer: Optional[Answer] = None
    messages: List[Message] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now())
    last_action: StateAction

    def immutable_copy_conversation_id(
        self, conversation_id: str
    ) -> "ConversationState":
        return self.model_copy(update={"conversation_id": conversation_id}, deep=True)

    def immutable_copy_questions(self, questions: Questions) -> "ConversationState":
        return self.model_copy(update={"questions": questions}, deep=True)

    def immutable_copy_refined_questions(
        self, refined_questions: RefinedQuestions | None
    ) -> "ConversationState":
        return self.model_copy(
            update={"refined_questions": refined_questions}, deep=True
        )

    def immutable_copy_query(self, query: Query | None) -> "ConversationState":
        return self.model_copy(update={"query": query}, deep=True)

    def immutable_copy_web_search_results(
        self, web_search_results: WebSearchResults
    ) -> "ConversationState":
        return self.model_copy(
            update={"web_search_results": web_search_results}, deep=True
        )

    def immutable_copy_web_data_collection(
        self, web_data_collection: WebDataCollection
    ) -> "ConversationState":
        return self.model_copy(
            update={"web_data_collection": web_data_collection}, deep=True
        )

    def immutable_copy_sources(self, sources: Sources | None) -> "ConversationState":
        return self.model_copy(update={"sources": sources}, deep=True)

    def immutable_copy_prompt(self, prompt: Prompt) -> "ConversationState":
        return self.model_copy(update={"prompt": prompt}, deep=True)

    def immutable_copy_answer(self, answer: Answer | None) -> "ConversationState":
        return self.model_copy(update={"answer": answer}, deep=True)

    def immutable_copy_messages(self, messages: List[Message]) -> "ConversationState":
        return self.model_copy(update={"messages": messages}, deep=True)

    def immutable_copy_last_updated(
        self, last_updated: datetime
    ) -> "ConversationState":
        return self.model_copy(update={"last_updated": last_updated}, deep=True)

    def immutable_copy_last_action(
        self, last_action: StateAction
    ) -> "ConversationState":
        return self.model_copy(update={"last_action": last_action}, deep=True)


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
