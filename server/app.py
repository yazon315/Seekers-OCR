from fastapi import FastAPI, Request, UploadFile, File, status
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from funcs import crop_img, acc_img, create_html_content
import os
from fastapi.middleware.cors import CORSMiddleware

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
    crop_img(file)
    return templates.TemplateResponse("cropped_result.html", {"request": request,
                                                              "content": create_html_content(cropped_path,
                                                                                             file.filename)})


@app.post("/enhanced_result", response_class=HTMLResponse)
def process_cropped_img(request: Request):
    acc_img()
    redirect_url = request.url_for("enhanced_result", **{'filename': acc_filename})
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/enhanced_result/{filename}", response_class=HTMLResponse)
def enhanced_result(request: Request, filename: str):
    return templates.TemplateResponse("enhanced_result.html", {"request": request,
                                                               "content": create_html_content(acc_path,
                                                                                              filename)})
