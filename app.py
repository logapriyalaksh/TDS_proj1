# /// srcipt
# requires-python = ">=3.12"
# dependencies = [
#       "fastapi", 
#       "uvicorn", 
#       "requests"
# ]
# ///


from urllib import response
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import subprocess
import json
from datetime import datetime
import sqlite3
import base64
import re
import httpx
import numpy as np
import markdown
import speech_recognition as sr
from pydub import AudioSegment
import os
from fastapi.responses import PlainTextResponse
from bs4 import BeautifulSoup
import pandas as pd
import duckdb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def install_uv():
    subprocess.run(["pip", "install", "uv"])

def run_script(script_url, email):
    #install_uv()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    command = f"uv run {script_url} {email}"
    subprocess.run(command, shell=True)

def format_file(file_path):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path])

def parse_date(date_str):
    for fmt in ("%b %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"time data '{date_str}' does not match any known format")

def count_wednesdays(file_path, output_path, day_name):
    with open(file_path, "r") as f:
        dates = f.readlines()
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    wednesdays = sum(1 for date in dates if parse_date(date.strip()).weekday() == day_mapping[day_name])
    with open(output_path, "w") as f:
        f.write(str(wednesdays))

def sort_contacts(input_path, output_path):
    with open(input_path, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    with open(output_path, "w") as f:
        json.dump(sorted_contacts, f, indent=4)

def extract_logs(input_dir, output_path):
    log_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".log")], key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    with open(output_path, "w") as f:
        for log_file in log_files[:10]:
            with open(os.path.join(input_dir, log_file), "r") as lf:
                f.write(lf.readline())

def create_index(input_dir, output_path):
    index = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file] = line[2:].strip()
                            break
    with open(output_path, "w") as f:
        json.dump(index, f, indent=4)

def extract_email(input_path, output_path):
    with open(input_path, "r") as f:
        email_content = f.read()
    # Define the prompt for the LLM
    prompt = f"Extract the sender's email address from the following email content:\n\n{email_content}"
    
    # Send the prompt to the LLM
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Extract the sender's email address from the following email content"
            },
            {
                "role": "user",
                "content": email_content
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "math_response",
            "strict": True,
            "schema":{
                "type": "object",
                "properties": {
                    "sendersEmailAdress": {
                    "type": "string"
                    }
                },
                "required": ["sendersEmailAdress"],
                "additionalProperties": False
                }}
            }
            }

    response = requests.post(url=url, headers=headers, json=data)
    response_json = response.json()
    email_address_dict = json.loads(response_json['choices'][0]['message']['content'].strip())

    # Extract the email address
    email_address = email_address_dict['sendersEmailAdress']
    
    # Write the extracted email address to the output file
    with open(output_path, "w") as f:
        f.write(email_address)
def extract_credit_card(input_path, output_path):
    # Assuming LLM is used to extract credit card number
    with open(input_path, 'rb') as f:
        binary_data = f.read()
        image_b64 = base64.b64encode(binary_data).decode()

# Data URI example (embed images in HTML/CSS)
    data_uri = f"data:image/png;base64,{image_b64}"
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data={
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Extract the number from the image"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": data_uri
                }
                }
            ]
            }
        ]
        }

    response = requests.post(url=url, headers=headers, json=data)
    response_json=response.json()
    content = response_json['choices'][0]['message']['content']
    number = re.search(r'\*\*(\d{4} \d{4} \d{4} \d{3})\*\*', content).group(1)

    # Remove spaces from the number
    number = number.replace(' ', '')

    with open(output_path, "w") as f:
        f.write(number)

def get_openai_embeddings(texts,model="text-embedding-3-small"):
    """Fetches embeddings for a list of texts using OpenAI's embedding API in batch mode."""

    data = {"input": texts, "model": model}
    embedding_url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
}
    
    response = requests.post(url=embedding_url, headers=headers, json=data)
    print(response.json())
    if response.status_code == 200:
        return [item["embedding"] for item in response.json()["data"]]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def cosine_similarity_matrix(embeddings):
    """Computes cosine similarity matrix for a set of embeddings."""
    embeddings = np.array(embeddings)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def find_most_similar_comments(input_path, output_path):
    """Finds the most similar pair of comments using embeddings and writes them to a file."""
    with open(input_path, "r") as file:
        comments = [line.strip() for line in file.readlines() if line.strip()]
    
    if len(comments) < 2:
        raise ValueError("Not enough comments to compare.")
    
    # Fetch embeddings in one batch request
    embeddings = get_openai_embeddings(comments)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity_matrix(embeddings)

    # Find the most similar pair (excluding diagonal)
    np.fill_diagonal(similarity_matrix, -1)  # Avoid self-comparison
    max_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

    most_similar_pair = (comments[max_index[0]], comments[max_index[1]])

    # Write to output file
    with open(output_path, "w") as file:
        file.write(most_similar_pair[0] + "\n")
        file.write(most_similar_pair[1] + "\n")
    


def calculate_sales(input_path, output_path):
    conn = sqlite3.connect(input_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = 'gold'")
    total_sales = cursor.fetchone()[0]
    with open(output_path, "w") as f:
        f.write(str(total_sales))
    conn.close()

def convert_markdown_to_html(markdown_content):
    # Convert Markdown to HTML using the markdown library
    html_content = markdown.markdown(markdown_content)
    return html_content


# Function to read the input Markdown file, convert it to HTML, and save to the output file
def convert_markdown_file(input_path, output_path):
    try:
        # Read the content from the input Markdown file
        with open(input_path, 'r') as md_file:
            markdown_content = md_file.read()
        
        # Convert Markdown to HTML
        html_content = convert_markdown_to_html(markdown_content)
        
        # Save the HTML content to the output file
        with open(output_path, 'w') as html_file:
            html_file.write(html_content)
        

    
    except Exception as e:
        print (e)

def transcribe_mp3_to_text(input_path: str, output_path: str):
    try:
        # Convert MP3 file to WAV format
        wav_path = '/tmp/temp_audio.wav'  # Temporary location for WAV file
        audio = AudioSegment.from_mp3(input_path)
        audio.export(wav_path, format="wav")
        
        # Use SpeechRecognition to transcribe the audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        
        # Save the transcription to the specified output file
        with open(output_path, 'w') as f:
            f.write(transcription)
        
        # Clean up the temporary WAV file
        os.remove(wav_path)
        
        print(f"Transcription saved to {output_path}")
    
    except Exception as e:
        print(e)

def B3(url, save_path):
    
    import requests
    response = requests.get(url)
    with open(save_path, 'w') as file:
        file.write(response.text)

def scrape_webpage(url: str, output_path: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    with open(output_path, "w") as file:
        file.write(soup.prettify())


def run_sql_query(input_path, output_path, query):
    print(f"Running SQL query: {query}, {input_path}, {output_path}")
    if not input_path or not query:
        raise HTTPException(status_code=400, detail="Invalid input parameters: input_path and query are required.")
    
    """
    Runs a SQL query on a SQLite or DuckDB database and saves the result.
    
    Parameters:
        db_path (str): Path to the SQLite (.db) or DuckDB (.duckdb) database file.
        query (str): SQL query to execute.
        output_file (str): File to save the results.
        output_format (str): "csv" or "json" (default: "csv").
    """
    # Determine database type (SQLite or DuckDB)
    is_duckdb = input_path.endswith(".duckdb")
    
    if not output_path:
        output_path = "./data/output_B5.csv"
    elif output_path.startswith("/"):
        output_path = f".{output_path}"
    
    output_format = "csv" if output_path.endswith(".csv") else "json" if output_path.endswith(".json") else "txt"

    # Connect to the database
    conn = duckdb.connect(input_path) if is_duckdb else sqlite3.connect(input_path)
    
    try:
        # Execute the query and fetch results into a DataFrame
        df = pd.read_sql_query(query, conn)

        # Save results
        if output_format == "json":
            df.to_json(output_path, orient="records", indent=4)
        elif output_format == "txt":
            df.to_csv(output_path, sep="\t", index=False)
        else:  # Default is CSV
            df.to_csv(output_path, index=False)

        print(f"Query executed successfully. Results saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        conn.close()

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "Install uv and run a script from a URL with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of arguments to pass to the script."
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a file using prettier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to format."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_wednesdays",
            "description": "Count the number of Wednesdays in a file and write the result to another file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file containing dates."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the result."
                    },
                    "day_name": {
                    "type": "string",
                    "enum": [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday"
                    ],
                    "description": "The name of the day to count."
                }

                },
                
                "required": ["file_path", "output_path","day_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts by last name and first name and write the result to another file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing contacts."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the sorted contacts."
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_logs",
            "description": "Extract the first line of the 10 most recent log files and write to another file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_dir": {
                        "type": "string",
                        "description": "The directory containing log files."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the extracted lines."
                    }
                },
                "required": ["input_dir", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_index",
            "description": "Create an index of Markdown files and their titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_dir": {
                        "type": "string",
                        "description": "The directory containing Markdown files."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the index."
                    }
                },
                "required": ["input_dir", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email",
            "description": "Extract the sender's email address from an email file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing the email."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the extracted email address."
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card",
            "description": "Extract the credit card number from an image file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the image file containing the credit card number."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the extracted credit card number."
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_similar_comments",
            "description": "Find the most similar pair of comments using embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing comments."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the similar comments."
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sales",
            "description": "Calculate the total sales of Gold tickets from a SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the SQLite database file."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path of the file to write the total sales."
                    }
                },
                "required": ["input_path", "output_path"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "convert_markdown_file",
        "description": "Convert a Markdown file to HTML and save to the output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "The path of the input Markdown file."
                },
                "output_path": {
                    "type": "string",
                    "description": "The path of the output HTML file."
                }
            },
            "required": ["input_path", "output_path"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "transcribe_mp3_to_text",
        "description": "Transcribe audio from an MP3 file and save the transcription to the output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "The path of the input MP3 file."
                },
                "output_path": {
                    "type": "string",
                    "description": "The path of the output transcription file."
                }
            },
            "required": ["input_path", "output_path"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "B3",
        "description": "Fetch data from an API and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch data from."
                },
                "output_path": {
                    "type": "string",
                    "description": "The path to save the fetched data."
                }
            },
            "required": ["url", "output_path"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "scrape_webpage",
        "description": "Extract data from (i.e. scrape) a website and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to scrape."
                },
                "output_path": {
                    "type": "string",
                    "description": "The path to save the scraped data."
                }
            },
            "required": ["url", "output_path"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Run a SQL query on a SQLite or DuckDB database and save the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "The path of the SQLite or DuckDB database file."
                },
                "output_path": {
                    "type": "string",
                    "description": "The path of the file to save the results."
                },
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute."
                }
            },
            "required": ["input_path", "output_path", "query"]
        }
    }
}
]

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

    # Remove the raise statement since we're providing a default token

@app.get("/")
def home ():
    return {"Yay TDS is awesome!"}

@app.get("/read")
def read_file(path: str):
    """
    Reads the content of a specified file.
    - Returns 200 OK if successful.
    - Returns 404 Not Found if the file does not exist.
    """

    try:
        # Ensure the file is inside the /data/ directory
        if not path.startswith("/data/"):
            raise HTTPException(status_code=400, detail="Access restricted to /data/ directory.")

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found.")

        # Read file content
        with open(path, "r", encoding="utf-8") as file:
            content = file.read().strip()  

        return PlainTextResponse(content)

    except HTTPException as e:
        raise e 

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.post("/run")
def task_runner(task: str):
    # Define forbidden patterns (including variations like os . remove, shutil . rmtree)
    FORBIDDEN_PATTERNS = [
        r"\bdelete\b", r"\bremove\b", r"\berase\b", r"\bdestroy\b", r"\bdrop\b", r"\brm -rf\b",
        r"\bunlink\b", r"\brmdir\b", r"\bdel\b", 
        r"\bos\s*\.\s*remove\b", 
        r"\bos\s*\.\s*unlink\b", 
        r"\bshutil\s*\.\s*rmtree\b", 
        r"\bpathlib\s*\.\s*Path\s*\.\s*unlink\b"
    ]

    # Check if any forbidden pattern is in the task string
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Task contains forbidden operations (deletion is not allowed).")
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": task
            },
            {
                "role": "system",
                "content": """
You are an assistant who has to perform various tasks based on the user's request.
You have access to the following tools:
1. run_script: Install uv and run a script from a URL with provided arguments.
2. format_file: Format a file using prettier.
3. count_wednesdays: Count the number of Wednesdays or any week days in a file and write the result to another file.
4. sort_contacts: Sort contacts by last name and first name and write the result to another file.
5. extract_logs: Extract the first line of the 10 most recent log files and write to another file.
6. create_index: Create an index of Markdown files and their titles.
7. extract_email: Extract the sender's email address from an email file.
8. extract_credit_card: Extract the credit card number from an image file.
9. find_most_similar_comments: Find the most similar pair of comments using embeddings.
10. calculate_sales: Calculate the total sales of Gold tickets from a SQLite database.
11. convert_markdown_file: Convert a Markdown file to HTML and save to the output file.
12. transcribe_mp3_to_text: Transcribe audio from an MP3 file and save the transcription to the output file.
13. B3: Fetch data from an API and save it.
14. scrape_webpage: Extract data from (i.e. scrape) a website and save it.
15. run_sql_query: Run a SQL query on a SQLite or DuckDB database and save the result.

Use the appropriate tool based on the task description provided by the user.
                """
            }
        ],
        "tools": tools,
        "tool_choice": "auto"
    }

    
    response = requests.post(url=url, headers=headers, json=data)
    response_json = response.json()
    function_details = response_json['choices'][0]['message']['tool_calls'][0]['function']

    print (function_details)
    
    # Execute the function based on the function details
    function_name = function_details['name']
    arguments = json.loads(function_details['arguments'])
    path_keys = ["input_path", "output_path", "input_dir", "file_path"]

    # Check if any path-related key exists and validate it
    for key in path_keys:
        if key in arguments:
            if not arguments[key].startswith("/data"):
                raise HTTPException(status_code=400, detail=f"Invalid path '{arguments[key]}': Access outside /data is not allowed.")
    
    if function_name == "run_script":
        run_script(arguments['script_url'], arguments['args'][0])
    elif function_name == "format_file":
        format_file(arguments['file_path'])
    elif function_name == "count_wednesdays":
        count_wednesdays(arguments['file_path'], arguments['output_path'],arguments['day_name'])
    elif function_name == "sort_contacts":
        sort_contacts(arguments['input_path'], arguments['output_path'])
    elif function_name == "extract_logs":
        extract_logs(arguments['input_dir'], arguments['output_path'])
    elif function_name == "create_index":
        create_index(arguments['input_dir'], arguments['output_path'])
    elif function_name == "extract_email":
        extract_email(arguments['input_path'], arguments['output_path'])
    elif function_name == "extract_credit_card":
        extract_credit_card(arguments['input_path'], arguments['output_path'])
    elif function_name == "find_most_similar_comments":
        find_most_similar_comments(arguments['input_path'], arguments['output_path'])
    elif function_name == "calculate_sales":
        calculate_sales(arguments['input_path'], arguments['output_path'])
    elif function_name == "convert_markdown_file":
        convert_markdown_file(arguments['input_path'], arguments['output_path'])
    elif function_name == "transcribe_mp3_to_text":
        transcribe_mp3_to_text(arguments['input_path'], arguments['output_path'])
    elif function_name == "B3":
        B3(arguments['url'], arguments['output_path'])
    elif function_name == "scrape_webpage":
        scrape_webpage(arguments['url'], arguments['output_path'])
    elif function_name == "run_sql_query":
        run_sql_query(arguments['input_path'], arguments['output_path'], arguments['query'])
    else:
        raise HTTPException(status_code=500, detail=f"Function  not found.")
    
    return function_details

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)