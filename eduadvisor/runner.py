import uvicorn
import subprocess
import typer

app = typer.Typer()


@app.command()
def start_server():
    """
    Start the Uvicorn server with FastAPI application.
    """
    uvicorn.run("eduadvisor.api:app", host="127.0.0.1", port=8000, reload=True)


@app.command()
def start_worker():
    """
    Start the Celery worker.
    """
    subprocess.run(["celery", "-A", "eduadvisor.worker", "worker", "--loglevel=info"])


if __name__ == "__main__":
    app()
