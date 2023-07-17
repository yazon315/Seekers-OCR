from fastapi import FastAPI, Request, UploadFile, File, status
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import funcs
import os
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
import argparse

IMAGE_DIR = "images"
cropped_path = os.path.join(IMAGE_DIR, "cropped.png")
acc_filename = "cropped_out.png"
acc_path = os.path.join(IMAGE_DIR, acc_filename)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("recognition.html", {"request": request})


@app.post("/result", response_class=HTMLResponse)
def process_raw_img(request: Request, file: UploadFile = File(...)):
    funcs.remove_images()
    funcs.crop_img(file)
    return templates.TemplateResponse("cropped_result.html", {"request": request,
                                                              "content": funcs.create_html_content(cropped_path,
                                                                                                   file.filename)})


@app.post("/enhanced_result", response_class=HTMLResponse)
def process_cropped_img(request: Request):
    funcs.acc_img()
    redirect_url = request.url_for("enhanced_result", **{'filename': acc_filename})
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/enhanced_result/{filename}", response_class=HTMLResponse)
def enhanced_result(request: Request, filename: str):
    while True:
        try:
            content = funcs.create_html_content(acc_path, filename)
            break

        except:
            time.sleep(5)

    return templates.TemplateResponse("enhanced_result.html", {"request": request, "content": content})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
