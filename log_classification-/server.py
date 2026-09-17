"""
server.py
---------
Exposes the classifier as a REST API so external tools (like n8n) can
call it over HTTP.

Run with:
    uvicorn server:app --reload

Endpoint:
    POST /classify/
        - form-data file: a CSV with 'source' and 'log_message' columns
        - header 'X-Groq-Api-Key' (optional): needed only for LegacyCRM
          logs / RAG fallback
        - returns: the same CSV with added 'target_label', 'confidence',
          and 'explanation' columns
"""

import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse

from classify import classify

app = FastAPI(title="Log Classifier API")


@app.post("/classify/")
async def classify_logs(file: UploadFile, x_groq_api_key: str = Header(default=None)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        try:
            results = classify(
                list(zip(df["source"], df["log_message"])),
                api_key=x_groq_api_key,
            )
        except RuntimeError as e:
            # e.g. Groq API exhausted its retries
            raise HTTPException(status_code=502, detail=f"Classification backend error: {e}")

        df["target_label"] = [label for label, _, _ in results]
        df["confidence"] = [round(conf, 3) for _, conf, _ in results]
        df["explanation"] = [expl for _, _, expl in results]

        output_file = "resources/output.csv"
        df.to_csv(output_file, index=False)
        return FileResponse(output_file, media_type="text/csv", filename="output.csv")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.get("/health")
async def health():
    return {"status": "ok"}
