from collections.abc import Callable
from celery import Celery
from datetime import datetime
import openai
from eduadvisor.model import (
    ConversationState,
    Message,
    MessageType,
    Questions,
    QuestionsScore,
    RefinedQuestions,
    StateAction,
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


def state_process(
    state: ConversationState,
    previous_action: StateAction | None,
    action: StateAction,
    send_state_to_user: Callable[[ConversationState, StateAction, str], None],
) -> ConversationState:
    match (previous_action, action):
        case (None, StateAction.COORDINATE):
            if state.questions:
                send_process_message_to_user(
                    state, action, "Asserting the quality of your questions"
                )
                quality_score = quality_check_your_questions(state.questions)
                if quality_score and quality_score.score < 80:
                    state = state.model_copy(
                        update={
                            "questions": state.questions.model_copy(
                                update={"questions_score": quality_score}, deep=True
                            )
                        },
                        deep=True,
                    )
                    send_process_message_to_user(
                        state,
                        action,
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
                    state = state.model_copy(
                        update={"question_score": quality_score}, deep=True
                    )
                    send_process_message_to_user(
                        state,
                        action,
                        f"The quality is good: Score: {quality_score.score} The comment is {quality_score.score_comment}",
                    )

            send_process_message_to_user(
                state,
                action,
                f"The quality is good. We will continue creating prompts",
            )
            return state

        case (StateAction.CHALLENGE, StateAction.COORDINATE):
            # TODO ADD SOME LOGIC IF THE CHALLENGE IS NOT THAT GOOD
            # Append result to the conversation
            # TODO MAKE THIS IMMUTABLE
            if state.refined_questions:
                state.messages.append(
                    Message(
                        message_type=MessageType.REFINED_QUESTION,
                        state_action=StateAction.CHALLENGE,
                        content=state.refined_questions.comment_to_the_original_question
                        + ".. Here are some suggestions: "
                        + ", ".join(state.refined_questions.refined_questions),
                        timestamp=datetime.now(),
                    )
                )

            return state

        case (StateAction.COORDINATE, StateAction.CHALLENGE):
            send_process_message_to_user(state, action, "Trying to challenge you..")
            if not state.questions:
                # Send error here
                return state
            else:
                refined_questions = challenge(state.questions)

                state = state.model_copy(
                    update={"refined_questions": refined_questions}, deep=True
                )

                return state_process(
                    state,
                    StateAction.CHALLENGE,
                    StateAction.COORDINATE,
                    send_state_to_user,
                )
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


def challenge(question: Questions) -> RefinedQuestions | None:
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
