from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from funcs import crop_img, create_html_content
import os

IMAGE_DIR = "images"
cropped_path = os.path.join(IMAGE_DIR, "cropped.png")

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("recognition.html", {"request": request})


@app.post("/result", response_class=HTMLResponse)
def process_img(request: Request, file: UploadFile = File(...)):
    crop_img(file)
    return templates.TemplateResponse("result.html", {"request": request,
                                                      "content": create_html_content(file.filename)})