import pathlib

import requests
from invoke.context import Context
from invoke.tasks import task
from tqdm import tqdm

project_folder = pathlib.Path(__file__).parent.resolve()


def download_with_progress_bar(url: str, filepath: pathlib.Path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


@task
def download_mistral_7b(ctx: Context):
    print("Downloading Mistral 7B Q4 model...")

    url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    filepath = project_folder / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    download_with_progress_bar(url, filepath)
