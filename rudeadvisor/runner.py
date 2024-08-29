import uvicorn
import typer

app = typer.Typer()


@app.command()
def start_server():
    """
    Start the Uvicorn server with FastAPI application.
    """
    uvicorn.run("rudeadvisor.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    app()
