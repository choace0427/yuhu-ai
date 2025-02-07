import os
import json
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["POST", "OPTIONS"]}})

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create or connect to Pinecone index
INDEX_NAME = "therapist-profiles"
if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

# Get the index
index = pc.Index(INDEX_NAME)

def get_file_path_from_url(url):
    """Extract the file path from Supabase storage URL"""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    try:
        resume_index = path_parts.index('resume')
        return '/'.join(path_parts[resume_index:])
    except ValueError:
        return None

def download_pdf(file_url, destination_path):
    """Download PDF from Supabase storage with simple path extraction"""
    try:
        file_path = file_url.split('/resume/')[-1] if '/resume/' in file_url else None
        
        if not file_path:
            print(f"Could not extract file path from URL: {file_url}")
            return False
        
        response = supabase.storage.from_('resume').download(file_path)
        # os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        with open(destination_path, 'wb+') as f:
            f.write(response)
        return True
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with error handling"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def generate_embedding(text):
    try:
        """Generate embedding using OpenAI's text-embedding-ada-002 model"""
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
    except Exception as e:
        print("Error generating embeddings", str(e))

    embedding_data = response.model_dump()["data"][0]["embedding"]
    return embedding_data

def format_services(services):
    """Format services list into a string"""
    return " ".join([
        f"{service.get('service', '')} - {service.get('summary', '')}" 
        for service in services
    ])

@app.route("/api/train-therapist", methods=['POST'])
def train_therapist():
    try:
        profile_data = request.get_json()
        therapist_id = profile_data.get('therapist_id')
        
        # Combine all text data for embedding
        text_components = []
        
        # 1. Process PDF if provided
        if pdf_url := profile_data.get('resume_url'):
            pdf_path = f'temp_{therapist_id}.pdf'
            if download_pdf(pdf_url, pdf_path):
                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text:
                    text_components.append(pdf_text)
                # Clean up temporary file
                try:
                    os.remove(pdf_path)
                except:
                    pass
        
        # 2. Add services information
        if services := profile_data.get('services'):
            services_text = format_services(services)
            text_components.append(services_text)
        
        # 3. Add summary
        if summary := profile_data.get('summary'):
            text_components.append(summary)
        
        # Combine all text components with spaces
        combined_text = " ".join(text_components).strip()
        
        if not combined_text:
            return jsonify({
                "success": False,
                "message": "No content available for embedding"
            }), 400 

        embeddings = generate_embedding(combined_text)

        if not embeddings:
            return jsonify({
                "success": False,
                "message": "Failed to generate embeddings"
            }), 500

        vectors = []
        vectors.append({
            "id": profile_data.get('therapist_id', ''),
            "values": embeddings,
            "metadata": {
                "therapist_id": profile_data.get('therapist_id', ''),
                "name": profile_data.get('name', ''), 
                "avatar_url": profile_data.get('avatar_url', ''), 
                "email": profile_data.get('email', ''), 
                "hourly_rate": str(profile_data.get('hourly_rate', '')), 
                "role": profile_data.get('role', ''),
                "services": [
                    f"{service.get('service_subcategory', '')} - {service.get('service_description', '')}" 
                    for service in profile_data.get('services', [])
                ],
                "summary": profile_data.get('summary', '')
            }
        })
        try:
            index.upsert(vectors=vectors, namespace="therapist_profiles")
        except Exception as e:
            print("Error upserting into Pinecone", str(e))
        
        return jsonify({
            "success": True, 
            "therapist_id": therapist_id,
            "message": "Therapist data successfully trained"
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500
@app.route("/api/get-best-therapist", methods=['POST', 'OPTIONS'])
def get_therapist():
    if request.method == 'OPTIONS':
        return '', 200
    else:
        try:
            promptData = request.get_json()

            embeddingsVectors = generate_embedding(promptData)
            therapistData = index.query(
                namespace="therapist_profiles",
                vector=embeddingsVectors,
                top_k=3,
                include_metadata=True,
            )

            if therapistData is None:
                return jsonify({
                    "success": False, 
                    "message": "No results found from Pinecone query"
                }), 404

            filtered_results = [match for match in therapistData.get('matches', []) if match.get('score', 0) > 0.75]

            filtered_json = []
            for item in filtered_results:
                scored_vector_dict = item.to_dict()
                filtered_json.append(json.dumps(scored_vector_dict))
            
            return jsonify({
                "success": True, 
                "therapistData": filtered_json,
                "message": "Therapist data successfully searched"
            }), 200

        except Exception as e:
            print("POST get-best-thierapist => ", e)
            return jsonify({
                "success": False, 
                "message": str(e)
            }), 500

def main():
    # Use host='0.0.0.0' to make the server accessible from other machines
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()