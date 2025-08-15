from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Hello World API", version="1.0.0")

@app.get("/hello-world")
async def hello_world():
    
    return {
        "message": "Hello World!",

    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)