from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import httpx
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define CORS configuration
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Env file schema
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    model_name: str
    ollama_url: str

# Response Schema for plain text
class ResponseModel(BaseModel):
    response: str

settings = Settings()

# Unique delimiter to separate diff and user instructions
DELIMITER = "-----END_OF_DIFF-----"

prompt_template = """
    You are tasked with generating a professional and concise commit message for a code change. 
    The commit message should follow best practices, such as being descriptive, summarizing the 
    purpose of the change, and adhering to the conventional commit format (if applicable).

    Below, you will find:
    1. **Code Diff**: A representation of the changes made in the code.
    2. **Optional User Instruction**: Additional context or specific guidelines provided by the user. 
       This field may be blank.

    If the **Optional User Instruction** is blank, rely solely on the **Code Diff** to infer the intent 
    of the change and generate the commit message. If the instruction is provided, use it to enhance 
    or guide the commit message.

    **Input Format:**

    1. **Code Diff**:  
    ```
    {code_diff}
    ```

    2. **Optional User Instruction**:  
    ```
    {user_instruction}
    ```

    **Output Format:**

    Generate a commit message in the following format:

    ```
    <type>: <summary>

    <body>

    <footer>
    ```

    - `<type>`: A keyword indicating the type of change (e.g., `feat`, `fix`, `docs`, `refactor`, `test`).
    - `<summary>`: A brief description of the change, written in the imperative mood.
    - `<body>`: A detailed explanation of the change, if necessary.
    - `<footer>`: References to issues, breaking changes, or other relevant notes.

    If no specific format is required, ensure the commit message is still clear, concise, and professional.
    Do not return anything other than the commit message, not even any identifier, or even the thought process.
    """

async def get_response(prompt: str):
    payload = {
        "model": settings.model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            print("Sending request to Ollama API...") 
            response = await client.post(url=settings.ollama_url, json=payload, timeout=None)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"HTTP error! Status: {response.status_code}")

            combined_response = ""

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed_response = json.loads(line)
                    if "response" in parsed_response:
                        response_text = parsed_response["response"]
                        combined_response += response_text
                        if parsed_response.get("done", False):
                            break

                except json.JSONDecodeError:
                    print("Failed to parse line as JSON:", line)
                    continue

            return combined_response
        
    except Exception as e:
        print("Error fetching response:", e)
        raise HTTPException(status_code=500, detail="Error fetching response")

def generate_prompt(code_diff: str, user_instruction: str):    
    # Format the prompt with the provided inputs
    prompt = prompt_template.format(code_diff=code_diff, user_instruction=user_instruction)
    return prompt.strip()

@app.post("/generate", response_model=str)  # Return plain text
async def generate_text(request: Request):
    try:
        # Read the raw text body of the request
        raw_text = await request.body()
        raw_text = raw_text.decode('utf-8')  # Decode bytes to string

        # Split the input into code_diff and user_instruction using the delimiter
        parts = raw_text.split(DELIMITER, 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid input format. Expected 'diff' and 'user_instruction' separated by a delimiter.")

        code_diff, user_instruction = parts

        # Create the prompt
        prompt = generate_prompt(code_diff.strip(), user_instruction.strip())
        
        # Capture the response
        response = await get_response(prompt)
        print(response)
        response = response.strip()

        if len(response) == 0:
            print("No response... Trying again")
            response = await get_response(prompt)

        # Return the generated response as plain text
        return response
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)