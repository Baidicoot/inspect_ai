from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate
from inspect_ai import eval
from dotenv import load_dotenv

load_dotenv()

@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                id=i,
                input="Just reply with Hello World",
                target="Hello World",
            )
        for i in range(1000)],
        solver=[
            generate(),
        ],
        scorer=exact(),
    )


if __name__ == "__main__":
    eval(hello_world(), model="openai/gpt-4o", display="web")