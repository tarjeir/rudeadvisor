from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from uuid import uuid4
import redis
import asyncio
import inspect
from rudeadvisor import model as edu_model
from rudeadvisor import worker as edu_worker


app = FastAPI()
templates = Jinja2Templates(directory="templates")

redis_client = redis.StrictRedis(
    decode_responses=True, host="localhost", port=6379, db=0
)


# Helper functions
def set_state_in_cache(conversation_id: str, state: edu_model.ConversationState):
    redis_client.set(conversation_id, state.model_dump_json())


async def get_state_json(conversation_id: str) -> None | edu_model.ConversationState:
    state_json = redis_client.get(conversation_id)
    if inspect.isawaitable(state_json):
        state_json = await state_json

    if state_json:
        return edu_model.ConversationState.model_validate_json(state_json)
    return None


def delete_state_from_cache(conversation_id):
    redis_client.delete(conversation_id)


def template_based_on_message(
    message: edu_model.Message, jinja2_env: Jinja2Templates
) -> str:
    match message.message_type:
        case edu_model.MessageType.PROCESS:
            tmpl = jinja2_env.get_template("process_message.html")
            template_str = tmpl.render(
                content=message.content, time=message.timestamp.isoformat()
            )
            return template_str
        case edu_model.MessageType.REFINED_QUESTION:
            tmpl = jinja2_env.get_template("refined_question_message.html")
            template_str = tmpl.render(
                content=message.content, time=message.timestamp.isoformat()
            )
            return template_str


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.get("/conversation")
def create_conversation(request: Request):
    conversation_id = str(uuid4())
    state = edu_model.create_initial_state(conversation_id)
    set_state_in_cache(conversation_id, state)
    return templates.TemplateResponse(
        "conversation.html",
        context={"request": request, "conversation_id": conversation_id},
    )


@app.get("/conversation/{conversation_id}")
async def stream_conversation(conversation_id: str):
    async def event_generator():
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"conversation:{conversation_id}")
        try:
            while True:
                message = pubsub.get_message()
                if message and message["type"] == "message":
                    message_data = message["data"]
                    state = edu_model.ConversationState.model_validate_json(
                        message_data
                    )
                    message_template = template_based_on_message(
                        state.messages[-1], templates
                    )
                    yield {
                        "event": "message",
                        "data": message_template.replace("\n", ""),
                    }
                await asyncio.sleep(0.1)
        finally:
            pubsub.unsubscribe()

    return EventSourceResponse(event_generator())


@app.post("/conversation/{conversation_id}", status_code=201)
async def handle_action(
    conversation_id: str,
    questions_request: edu_model.QuestionsRequest,
    background_task: BackgroundTasks,
):
    state = await get_state_json(conversation_id)
    if state is None:
        raise HTTPException(
            status_code=404, detail=f"Conversaton with {conversation_id} does not exist"
        )

    questions = edu_model.Questions(
        questions=list(
            map(
                lambda q_str: edu_model.Question(question_text=q_str),
                (
                    questions_request.questions_list
                    if isinstance(questions_request.questions_list, list)
                    else [questions_request.questions_list]
                ),
            )
        ),
        questions_score=None,
    )

    state = state.model_copy(update={"questions": questions}, deep=True)
    model_json = state.model_dump_json()
    background_task.add_task(
        edu_worker.process_action, model_json, None, edu_model.StateAction.COORDINATE
    )

    return {"status": "Action is being processed"}
