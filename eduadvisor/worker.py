from collections.abc import Callable
from celery import Celery
from datetime import datetime
from fastapi.datastructures import State
import openai
from eduadvisor.model import (
    ConversationState,
    Message,
    MessageType,
    Question,
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
    state.messages.append(
        Message(
            message_type=MessageType.PROCESS,
            state_action=action,
            content=message_content,
            timestamp=datetime.now(),
        )
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
            return state_process(
                state, StateAction.COORDINATE, StateAction.CHALLENGE, send_state_to_user
            )
        case (StateAction.CHALLENGE, StateAction.COORDINATE):
            # TODO ADD SOME LOGIC IF THE CHALLENGE IS NOT THAT GOOD
            # Append result to the conversation
            # TODO MAKE THIS IMMUTABLE
            if state.refined_questions:
                state.messages.append(
                    Message(
                        message_type=MessageType.REFINED_QUESTION,
                        state_action=StateAction.CHALLENGE,
                        content=state.refined_questions.comment_to_the_original_question,
                        timestamp=datetime.now(),
                    )
                )

            return state

        case (StateAction.COORDINATE, StateAction.CHALLENGE):
            send_process_message_to_user(state, action, "Trying to challenge you..")
            if not state.question:
                # Send error here
                return state
            else:
                state.refined_questions = challenge(state.question)

                return state_process(
                    state,
                    StateAction.CHALLENGE,
                    StateAction.COORDINATE,
                    send_state_to_user,
                )
        case _:
            print(f"Nothing MATCHED: {previous_action}{action}")
            return state


def challenge(question: Question) -> RefinedQuestions | None:

    completions = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """I am a question challenger.
        My job is to challenge your questions and generate more questions based on your initial question. 
        There are some rules I will endforce:
        Always generate minimum 3 refined questions at maximum 6
        I will try to put you off and add one relevant but bogus question, but I can not guarantee that I will
        Create a comment to the original question. You are allowed to be a bit 'mansplaining'when you comment the question. The challenger is not the nicest bot out there""",
            },
            {
                "role": "user",
                "content": f"Here is the question I want you to refine: {question.question_text}",
            },
        ],
        response_format=RefinedQuestions,
    )

    return completions.choices[0].message.parsed
