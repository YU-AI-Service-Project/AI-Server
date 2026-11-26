"""Internal AI HTTP service. Run: uvicorn api_server:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated,Any
import os,hmac,time
from fastapi import FastAPI,Header,HTTPException,Response
from pydantic import BaseModel,ConfigDict,Field
from predict_v9 import ETAPredictorV9

@asynccontextmanager
async def lifespan(app):
    app.state.predictor=ETAPredictorV9(os.environ.get('WATTUP_MODEL_DIR',str(Path(__file__).parent/'results')))
    yield

app=FastAPI(title='WattUp ETA v9 Arrival Forecast',version='9.0',lifespan=lifespan)

class Item(BaseModel):
    model_config=ConfigDict(extra='forbid')
    requestItemId:str=Field(min_length=1,max_length=100)
    stationId:str=Field(min_length=1,max_length=100)
    profile:str
    asOf:str
    etaSeconds:float=Field(ge=0,allow_inf_nan=False,strict=True)
    etaGeneratedAt:str|None=None
    roadDistanceM:float|None=Field(default=None,ge=0,allow_inf_nan=False)
    features:dict[str,Any]

class Batch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    requestId:str=Field(min_length=1,max_length=100)
    requestAt:str
    featureVersion:str
    items:list[Item]=Field(min_length=1,max_length=10)

@app.get('/health/ready')
def ready():
    return {'ready':True,'status':'UP','modelVersion':app.state.predictor.card['model_version'],
            'featureVersion':app.state.predictor.card['feature_version']}

@app.post('/internal/v3/arrival-predictions/batch')
def predict(batch:Batch,response:Response,authorization:Annotated[str|None,Header()]=None,
            x_internal_api_key:Annotated[str|None,Header()]=None):
    secret=os.environ.get('WATTUP_INTERNAL_API_KEY')
    if secret:
        # Authorization is used by Spring Boot and takes precedence when supplied.
        token=x_internal_api_key if authorization is None else authorization.removeprefix('Bearer ')
        if authorization is not None and not authorization.startswith('Bearer '):
            raise HTTPException(401,'INVALID_INTERNAL_API_KEY')
        if not token or not hmac.compare_digest(secret.encode('utf-8'),token.encode('utf-8')):
            raise HTTPException(401,'INVALID_INTERNAL_API_KEY')
    start=time.perf_counter_ns()
    try:result=app.state.predictor.predict_batch(batch.model_dump())
    except ValueError as e:raise HTTPException(422,str(e)) from None
    response.headers['X-AI-Handler-Ms']=str(round((time.perf_counter_ns()-start)/1e6,3))
    return result
