from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

web_router = APIRouter(tags=["Web Interface"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

@web_router.get("/dashboard", response_class=HTMLResponse)
async def render_dashboard(request: Request):
    """Serves the VERDICT AI Dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")