from fastapi import  HTTPException, Body
from app.utils.session import SESSIONS
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi,get_token
import io

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

api = HfApi()


async def clean_data(
    session_id: str = Body(...),
    instruction: str = Body(...)
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    df = SESSIONS[session_id]

    prompt = f"""
    You are a Pandas coding assistant.
    Apply this cleaning instruction to a DataFrame called df.
    Instruction: {instruction}
    Output only valid Python code (no markdown).
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Pandas coding assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    code = response.choices[0].message.content.strip()

    if code.startswith("```"):
        code = code.split("```")[1]
        code = code.replace("python\n", "").strip()

    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        df = local_vars["df"]
        SESSIONS[session_id] = df
    except Exception as e:
        return {"error": f"Error running code: {e}", "code": code}
    
    try:
    # Convert DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    # Upload the cleaned file to your Hugging Face repo
        repo_id = "prthm20/ZeoMl"  # <-- change this
        filename = f"{session_id}_cleaned.csv"

        api.upload_file(
        path_or_fileobj=csv_bytes,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Updated cleaned data for session {session_id}"
     )

        hf_status = f"Saved cleaned data to {repo_id}/{filename}"
    except Exception as e:
        hf_status = f"Error saving to Hugging Face: {e}"
    return {"message": "Cleaning applied", "code": code, "preview":  df.head().to_json(orient="records"), "huggingface_status": hf_status}
