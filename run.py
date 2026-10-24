import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=get_settings().port, reload=True)
