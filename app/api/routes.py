from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from PIL import Image
import io

from app.services.preprocessing_service import preprocess_image
from app.services.inference_service import predict

router = APIRouter()


@router.get("/")
def health():
    return {
        "status": "healthy"
    }


@router.post("/predict")
async def prediction(
        file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    processed_image = preprocess_image(
        image
    )

    return predict(
        processed_image
    )