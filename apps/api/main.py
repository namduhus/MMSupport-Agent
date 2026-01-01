from fastapi import FastAPI
from apps.api.router import router

app = FastAPI(title="MMSuport-Agent API", version="1.0.0")
app.include_router(router)
