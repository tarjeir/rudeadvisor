from eduadvisor import model as edu_model
from datetime import datetime
from typing import Callable
from eduadvisor import llm


def state_process(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    action: edu_model.StateAction,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Process the state transition based on the current action.
    """
    match (previous_action, action):
        case (transition_from, edu_model.StateAction.COORDINATE):
            return coordination(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.SCORE_QUERY):
            return score_query(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.CHALLENGE):
            return challenge(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.QUERY_LLM):
            return query_llm(state, transition_from, send_state_to_user)
        case _:
            print(f"Nothing MATCHED: {previous_action}{action}")
            return state


def coordination(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Coordinate the state based on the previous action.
    """
    match previous_action:
        case None:
            send_state_to_user(
                state,
                edu_model.StateAction.COORDINATE,
                f"Thanks ... ",
            )
            return state_process(
                state,
                edu_model.StateAction.COORDINATE,
                edu_model.StateAction.SCORE_QUERY,
                send_state_to_user,
            )
        case edu_model.StateAction.SCORE_QUERY:
            if not state.questions:
                send_state_to_user(
                    state, edu_model.StateAction.COORDINATE, "Error state"
                )
                return state

            quality_score = state.questions.questions_score
            if quality_score and quality_score.score < 80:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    quality_score.score_comment
                    + f", I scored it {quality_score.score}",
                )
                return state_process(
                    state,
                    edu_model.StateAction.COORDINATE,
                    edu_model.StateAction.CHALLENGE,
                    send_state_to_user,
                )
            elif quality_score:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    f"The quality is good. {quality_score.score}/100. The comment is {quality_score.score_comment}. We will continue creating prompts",
                )
                return state_process(
                    state,
                    edu_model.StateAction.COORDINATE,
                    edu_model.StateAction.QUERY_LLM,
                    send_state_to_user,
                )
            else:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    "Failed to score the query. Please retry...",
                )
                return state
        case edu_model.StateAction.CHALLENGE:
            if state.refined_questions:
                state = state.model_copy(
                    update={
                        "messages": state.messages
                        + [
                            edu_model.Message(
                                message_type=edu_model.MessageType.REFINED_QUESTION,
                                state_action=edu_model.StateAction.CHALLENGE,
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
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    "Failed to refine questions, please retry",
                )
            return state

    return state


def score_query(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Score the query based on the state and previous action.
    """
    match previous_action:
        case edu_model.StateAction.COORDINATE:
            if not state.questions:
                send_state_to_user(
                    state, edu_model.StateAction.SCORE_QUERY, "Error state"
                )
                return state

            send_state_to_user(
                state,
                edu_model.StateAction.SCORE_QUERY,
                "Asserting the quality of your questions",
            )
            quality_score = llm.quality_check_your_questions(state.questions)
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
                edu_model.StateAction.SCORE_QUERY,
                edu_model.StateAction.COORDINATE,
                send_state_to_user,
            )

    return state


def challenge(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Challenge the current state based on the previous action.
    """
    match previous_action:
        case edu_model.StateAction.COORDINATE:
            send_state_to_user(
                state, edu_model.StateAction.CHALLENGE, "Trying to challenge you.."
            )
            if not state.questions:
                # Send error here
                return state
            else:
                refined_questions = llm.challenge_llm(state.questions)
                state = state.model_copy(
                    update={"refined_questions": refined_questions}, deep=True
                )

                return state_process(
                    state,
                    edu_model.StateAction.CHALLENGE,
                    edu_model.StateAction.COORDINATE,
                    send_state_to_user,
                )
    return state


def query_llm(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Query the language model based on the current and previous action.
    """
    match previous_action:
        case edu_model.StateAction.COORDINATE:
            if state.questions:
                query = llm.extract_search_query(state.questions)
                state = state.model_copy(update={"query": query}, deep=True)
                return state_process(
                    state,
                    edu_model.StateAction.QUERY_LLM,
                    edu_model.StateAction.COORDINATE,
                    send_state_to_user,
                )
            else:
                return state_process(
                    state,
                    edu_model.StateAction.QUERY_LLM,
                    edu_model.StateAction.COORDINATE,
                    send_state_to_user,
                )
    return state
