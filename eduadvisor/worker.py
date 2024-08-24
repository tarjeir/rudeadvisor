from collections.abc import Callable
from celery import Celery
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import ValidationError
import re
import openai
from eduadvisor.model import (
    ConversationState,
    Message,
    MessageType,
    Query,
    Questions,
    QuestionsScore,
    RefinedQuestions,
    StateAction,
    WebSearchError,
    WebSearchResult,
    WebSearchResults,
)
import redis

celery_app = Celery("tasks", broker="redis://localhost:6379/0")
redis_client = redis.Redis(host="localhost", port=6379, db=0)
openai_client = openai.OpenAI()


def send_process_message_to_user(
    state: ConversationState, action: StateAction, message_content: str
):
    state = state.model_copy(
        update={
            "messages": state.messages
            + [
                Message(
                    message_type=MessageType.PROCESS,
                    state_action=action,
                    content=message_content,
                    timestamp=datetime.now(),
                )
            ]
        },
        deep=True,
    )

    channel_name = f"conversation:{state.conversation_id}"
    redis_client.publish(channel_name, state.model_dump_json())


@celery_app.task(name="process_action")
def process_action(
    conversation_state: ConversationState | str,
    previous_action: StateAction | None,
    action: StateAction,
):

    if isinstance(conversation_state, str):
        state = ConversationState.model_validate_json(conversation_state)
    else:
        state = conversation_state

    send_process_message_to_user(state, action, f"Processing your {action} request")

    state = state_process(state, previous_action, action, send_process_message_to_user)
    redis_client.publish(
        f"conversation:{state.conversation_id}", state.model_dump_json()
    )


def coordination(
    state: ConversationState,
    previous_action: StateAction | None,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match previous_action:
        case None:
            send_process_message_to_user(
                state,
                StateAction.COORDINATE,
                f"Thanks ... ",
            )
            return state_process(
                state,
                StateAction.COORDINATE,
                StateAction.SCORE_QUERY,
                send_state_to_user,
            )
        case StateAction.SCORE_QUERY:
            if not state.questions:
                send_process_message_to_user(
                    state, StateAction.COORDINATE, "Error state"
                )
                return state

            quality_score = state.questions.questions_score
            if quality_score and quality_score.score < 80:
                send_process_message_to_user(
                    state,
                    StateAction.COORDINATE,
                    quality_score.score_comment
                    + f", I scored it {quality_score.score}",
                )
                return state_process(
                    state,
                    StateAction.COORDINATE,
                    StateAction.CHALLENGE,
                    send_state_to_user,
                )
            elif quality_score:

                send_process_message_to_user(
                    state,
                    StateAction.COORDINATE,
                    f"The quality is good. We will continue creating prompts",
                )
                return state_process(
                    state,
                    StateAction.COORDINATE,
                    StateAction.QUERY_LLM,
                    send_state_to_user,
                )
            else:
                send_process_message_to_user(
                    state,
                    StateAction.COORDINATE,
                    "Failed to score the query. Please retry...",
                )
                return state
        case StateAction.CHALLENGE:
            if state.refined_questions:
                state = state.model_copy(
                    update={
                        "messages": state.messages
                        + [
                            Message(
                                message_type=MessageType.REFINED_QUESTION,
                                state_action=StateAction.CHALLENGE,
                                content=state.refined_questions.comment_to_the_original_question
                                + ".. Here are some suggestions: "
                                + ", ".join(state.refined_questions.refined_questions),
                                timestamp=datetime.now(),
                            )
                        ]
                    },
                    deep=True,
                )
            else:
                send_process_message_to_user(
                    state,
                    StateAction.COORDINATE,
                    "Failed to refine questions please.. please retry",
                )
            return state

    return state


def score_query(
    state: ConversationState,
    previous_action: StateAction | None,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match previous_action:
        case StateAction.COORDINATE:
            if not state.questions:
                send_process_message_to_user(
                    state, StateAction.SCORE_QUERY, "Error state"
                )
                return state

            send_process_message_to_user(
                state,
                StateAction.SCORE_QUERY,
                "Asserting the quality of your questions",
            )
            quality_score = quality_check_your_questions(state.questions)
            state = state.model_copy(
                update={
                    "questions": state.questions.model_copy(
                        update={"questions_score": quality_score}, deep=True
                    )
                },
                deep=True,
            )

            return state_process(
                state,
                StateAction.SCORE_QUERY,
                StateAction.COORDINATE,
                send_state_to_user,
            )

    return state


def challenge(
    state: ConversationState,
    previous_action: StateAction | None,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match previous_action:
        case StateAction.COORDINATE:
            send_process_message_to_user(
                state, StateAction.CHALLENGE, "Trying to challenge you.."
            )
            if not state.questions:
                # Send error here
                return state
            else:
                refined_questions = challenge_llm(state.questions)
                state = state.model_copy(
                    update={"refined_questions": refined_questions}, deep=True
                )

                return state_process(
                    state,
                    StateAction.CHALLENGE,
                    StateAction.COORDINATE,
                    send_state_to_user,
                )
    return state


def query_llm(
    state: ConversationState,
    previous_action: StateAction | None,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match previous_action:
        case StateAction.COORDINATE:
            if state.questions:
                query = extract_search_query(state.questions)
                state = state.model_copy(update={"query": query}, deep=True)
                return state_process(
                    state,
                    StateAction.QUERY_LLM,
                    StateAction.COORDINATE,
                    send_state_to_user,
                )
            else:
                return state_process(
                    state,
                    StateAction.QUERY_LLM,
                    StateAction.COORDINATE,
                    send_state_to_user,
                )
    return state


def state_process(
    state: ConversationState,
    previous_action: StateAction | None,
    action: StateAction,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match (previous_action, action):
        case (transition_from, StateAction.COORDINATE):
            return coordination(state, transition_from, send_state_to_user)
        case (transition_from, StateAction.SCORE_QUERY):
            return score_query(state, transition_from, send_state_to_user)
        case (transition_from, StateAction.CHALLENGE):
            return challenge(state, transition_from, send_state_to_user)
        case (transition_from, StateAction.QUERY_LLM):
            return query_llm(state, transition_from, send_state_to_user)
        case _:
            print(f"Nothing MATCHED: {previous_action}{action}")
            return state


def quality_check_your_questions(questions: Questions) -> QuestionsScore | None:

    question_as_numbered_prompt = ",".join(
        [f"{i}. " + q.question_text for i, q in enumerate(questions.questions)]
    )

    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """I am a question reviewer. 
                The questions that I will review should be reviewed after the following criterias
                1. if more than one... Are they too similar? In essence will the answers be too overlapping and repeating
                2. Are they too leading? In what way is it too leading?
                3. Clarity and specifity: Is it too vague or too specific
                4. Is it relevant and significant: Is it any scholarly interest? 
                5. Is it original? Can you find if this is a quite regular question to answers
                6. Can it be answered ?
                7. Criterion validity
                8. Is this a question at all? (Score 0 if so)
                
                Assert the questions, not the question domain. That means the question can be good but the domain assumptions wrong
                Please score the question between 0 - 100 (0 the question is bad - 100 is fantastic)
                Add a small comment that can be sent back to the user that describe the score (it is not going to be shown side by side)

                """,
            },
            {
                "role": "user",
                "content": f"Here is the question for review: {question_as_numbered_prompt}",
            },
        ],
        response_format=QuestionsScore,
    )

    return completions.choices[0].message.parsed


def challenge_llm(question: Questions) -> RefinedQuestions | None:
    contradiction = (
        [
            {
                "role": "system",
                "content": f"I have already scored these questions  and asserted it. Do not contradict the feedback already given: {question.questions_score.score_comment}",
            }
        ]
        if question.questions_score
        else []
    )
    messages = [
        {
            "role": "system",
            "content": """I am a question challenger.
        My job is to challenge your questions and generate more questions based on your initial question. 
        There are some rules I will endforce:
        Always generate minimum 3 refined questions at maximum 6
        I will try to put you off and add one relevant but bogus question, but I can not guarantee that I will
        Create a comment to the original question. You are allowed to be a bit 'mansplaining'when you comment the question. The challenger is not the nicest bot out there.
        """,
        },
        {
            "role": "user",
            "content": f"Here are the questions I want you to refine: {", ".join(list(map(lambda q: q.question_text,question.questions)))}",
        },
    ]
    messages.extend(contradiction)
    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=RefinedQuestions,
    )

    return completions.choices[0].message.parsed


def extract_search_query(questions: Questions) -> Query | None:
    """
    Extracts the most important topics for a search query from the given questions.
    """
    # Formulating the system message that guides the AI to generate a search query
    instructions = {
        "role": "system",
        "content": """I am an extraction bot.
        My job is to extract the most important topics from your questions to create a search query.
        There are some rules I will enforce:
        1. Focus on the main topics.
        2. Generate a concise and relevant search query.
        3. Use one word per topic 
        Below is the logic expressions allowed
        1. cats dogs 	Results about cats or dogs
        2. "cats and dogs" 	Results for exact term "cats and dogs". If no or few results are found, we'll try to show related result
        3. ~"cats and dogs" 	Experimental syntax: more results that are semantically similar to "cats and dogs", like "cats & dogs" and "dogs and cats" in addition to "cats and dogs".
        4.cats -dogs 	Fewer dogs in results
        5.cats +dogs 	More dogs in results
        6.cats filetype:pdf 	PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
        7.dogs site:example.com 	Pages about dogs from example.com
        8.cats -site:example.com 	Pages about cats, excluding example.com
        9.intitle:dogs 	Page title includes the word "dogs"
        10. inurl:cats 	Page URL includes the word "cats"
        """,
    }

    # Collecting user questions to be included in the prompt
    user_message = {
        "role": "user",
        "content": f"Here are the questions I want you to extract topics from: {', '.join([q.question_text for q in questions.questions])}",
    }

    # Combining the instructional and user messages into a single prompt
    messages = [instructions, user_message]

    # Making an API call to the AI model to generate the search query
    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=Query,
    )

    return completions.choices[0].message.parsed if completions.choices else None


def clean_and_parse(input_string: str) -> list[dict[str, str]]:
    # Regular expression to match the pattern within brackets
    pattern = re.compile(r"\[snippet:\s*(.+?),\s*title:\s*(.+?),\s*link:\s*(.+?)\]")

    # Find all matches in the input string
    matches = pattern.findall(input_string)

    # Construct the list of dictionaries from matches
    parsed_results = [
        {"snippet": match[0], "title": match[1], "link": match[2]} for match in matches
    ]

    return parsed_results


def query_duckduckgo(query: Query) -> WebSearchResults | WebSearchError:
    """
    Queries DuckDuckGo and returns snippets with URLs.
    """
    try:
        search = DuckDuckGoSearchResults()
        results = search.invoke(query.query_text)
        results_list = clean_and_parse(results)
        web_result_list = [WebSearchResult(**r) for r in results_list]
        return WebSearchResults(web_search_results=web_result_list)
    except ValidationError:
        return WebSearchError(
            message="We failed to return proper data from the search engine"
        )
    except Exception:
        return WebSearchError(message="We failed to search the web for relevant info")
