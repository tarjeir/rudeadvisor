import logging
from eduadvisor import model as edu_model
from typing import Callable
from eduadvisor import llm


def transition(
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
    logging.debug(
        f"Transitioning from {previous_action} to {action} with state: {state}"
    )

    match (previous_action, action):
        case (transition_from, edu_model.StateAction.COORDINATE):
            return coordination_agent(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.SCORE_QUERY):
            return score_query_agent(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.CHALLENGE):
            return challenge_agent(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.QUERY_LLM):
            return query_llm_agent(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.WEB_SEARCH):
            return web_search_agent(state, transition_from, send_state_to_user)
        case (transition_from, edu_model.StateAction.SOURCE_APPROVE):
            return source_approve_agent(state, transition_from, send_state_to_user)
        case _:
            logging.debug(
                f"No matching transition found for {previous_action} and {action}"
            )
            return state


def source_approve_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    logging.debug(
        f"Processing source approval with state: {state} and previous action: {previous_action}"
    )

    if state.web_search_results and state.query:
        send_state_to_user(
            state,
            edu_model.StateAction.SOURCE_APPROVE,
            "Let me do a sanity check on the URLs you've found.",
        )
        sources = llm.evaluate_the_sources(state.web_search_results, state.query)
        state = state.model_copy(update={"sources": sources}, deep=True)
        logging.debug(f"Updated state with sources: {sources}")

        if sources and len(sources.links) < 2:
            send_state_to_user(
                state,
                edu_model.StateAction.SOURCE_APPROVE,
                "Too few results retrieved from the web. You really should improve the query.",
            )
            return transition(
                state,
                edu_model.StateAction.SOURCE_APPROVE,
                edu_model.StateAction.QUERY_LLM,
                send_state_to_user,
            )
        elif sources and len(sources.links):
            explaination_of_removed_links = (
                sources.removed_links_explaination
                if sources.removed_links_explaination
                else ""
            )
            send_state_to_user(
                state,
                edu_model.StateAction.SOURCE_APPROVE,
                f"Okay, I guess we can use the links {sources.links} to generate a prompt and create an answer. "
                + explaination_of_removed_links,
            )
            return state
        else:
            send_state_to_user(
                state,
                edu_model.StateAction.SOURCE_APPROVE,
                "The links are not good enough.",
            )
            return state

    return state


def web_search_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    logging.debug(
        f"Processing web search with state: {state} and previous action: {previous_action}"
    )

    match previous_action:
        case edu_model.StateAction.QUERY_LLM:
            if state.query:
                search_results = llm.query_duckduckgo(state.query)
                if isinstance(search_results, edu_model.WebSearchError):
                    send_state_to_user(
                        state,
                        edu_model.StateAction.WEB_SEARCH,
                        "Unfortunately, the web search failed.",
                    )
                    return state
                else:
                    state = state.model_copy(
                        update={"web_search_results": search_results}, deep=True
                    )
                    logging.debug(
                        f"Updated state with search results: {search_results}"
                    )

                    send_state_to_user(
                        state,
                        edu_model.StateAction.WEB_SEARCH,
                        f"You only got {len(search_results.web_search_results)} search results.",
                    )
                    return transition(
                        state,
                        edu_model.StateAction.WEB_SEARCH,
                        edu_model.StateAction.SOURCE_APPROVE,
                        send_state_to_user,
                    )

    return state


def coordination_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Coordinate the state based on the previous action.
    """
    logging.debug(
        f"Coordinating with state: {state} and previous action: {previous_action}"
    )

    match previous_action:
        case None:
            send_state_to_user(
                state,
                edu_model.StateAction.COORDINATE,
                "Thanks ...",
            )
            return transition(
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
            logging.debug(f"Quality score: {quality_score}")

            if quality_score and quality_score.score < 80:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    quality_score.score_comment
                    + f", I scored it {quality_score.score}",
                )
                return transition(
                    state,
                    edu_model.StateAction.COORDINATE,
                    edu_model.StateAction.CHALLENGE,
                    send_state_to_user,
                )
            elif quality_score:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    f"The quality is fine. {quality_score.score}/100. The comment is {quality_score.score_comment}. We will continue creating prompts.",
                )
                return transition(
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
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    state.refined_questions.comment_to_the_original_question
                    + ".. Here are some suggestions: "
                    + ", ".join(state.refined_questions.refined_questions),
                )
            else:
                send_state_to_user(
                    state,
                    edu_model.StateAction.COORDINATE,
                    "Failed to refine questions, please retry",
                )
            return state

    return state


def score_query_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Score the query based on the state and previous action.
    """
    logging.debug(
        f"Scoring query with state: {state} and previous action: {previous_action}"
    )

    if not state.questions:
        send_state_to_user(state, edu_model.StateAction.SCORE_QUERY, "Error state")
        return state

    send_state_to_user(
        state,
        edu_model.StateAction.SCORE_QUERY,
        "Let me assess the quality of your questions.",
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
    logging.debug(f"Updated state with quality score: {quality_score}")

    return transition(
        state,
        edu_model.StateAction.SCORE_QUERY,
        edu_model.StateAction.COORDINATE,
        send_state_to_user,
    )


def challenge_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Challenge the current state based on the previous action.
    """
    logging.debug(
        f"Challenging with state: {state} and previous action: {previous_action}"
    )

    send_state_to_user(
        state, edu_model.StateAction.CHALLENGE, "I'm going to challenge you.."
    )
    if not state.questions:
        logging.error("No questions added")
        return state
    else:
        refined_questions = llm.challenge_llm(state.questions)
        state = state.model_copy(
            update={"refined_questions": refined_questions}, deep=True
        )
        logging.debug(f"Updated state with refined questions: {refined_questions}")

        return transition(
            state,
            edu_model.StateAction.CHALLENGE,
            edu_model.StateAction.COORDINATE,
            send_state_to_user,
        )


def query_llm_agent(
    state: edu_model.ConversationState,
    previous_action: edu_model.StateAction | None,
    send_state_to_user: Callable[
        [edu_model.ConversationState, edu_model.StateAction, str], None
    ],
) -> edu_model.ConversationState:
    """
    Query the language model based on the current and previous action.
    """
    logging.debug(
        f"Querying LLM with state: {state} and previous action: {previous_action}"
    )

    if state.questions:
        query = llm.extract_search_query(state.questions, state.query, state.sources)
        state = state.model_copy(update={"query": query}, deep=True)
        logging.debug(f"Updated state with query: {query}")

        return transition(
            state,
            edu_model.StateAction.QUERY_LLM,
            edu_model.StateAction.WEB_SEARCH,
            send_state_to_user,
        )

    send_state_to_user(
        state,
        edu_model.StateAction.QUERY_LLM,
        "There are no questions to analyze. Please try asking some first.",
    )
    return state
