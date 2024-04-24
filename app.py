from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Speaking_Silence.utils.common import decodeImage
from Speaking_Silence.pipeline.prediction import PredictionPipeline

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientApp:
    def __init__(self):
        # Initialize with a placeholder or default filename
        self.filename = "video.mp4"
        self.classifier = PredictionPipeline(self.filename)

    def update_filename(self, new_filename):
        self.filename = new_filename
        self.classifier.update_filename(self.filename) 
clApp = ClientApp()

@app.get("/")
async def home():
    return {"message": "Welcome to the Speaking Silence API!"}

@app.get("/train")
async def train_route():
    return {"message": "Training done successfully!"}

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    original_filename = file.filename
    clApp.filename = original_filename
    clApp.update_filename(clApp.filename)
    if file.content_type.startswith('video'):
        contents = await file.read()
        with open(clApp.filename, "wb") as f:
            f.write(contents)
        result = clApp.classifier.predict()
        print(result)
        return JSONResponse(content=result)
    else:
        contents = await file.read()
        decodeImage(contents, clApp.filename)
        result = clApp.classifier.predict()
        return JSONResponse(content=result)

# Swagger docs path
@app.get("/docs", include_in_schema=False)
async def get_swagger():
    from fastapi.openapi.utils import get_openapi
    return JSONResponse(content=get_openapi(title="Speaking Silence API", version="1.0.0", routes=app.routes))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
