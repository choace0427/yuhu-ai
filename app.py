import os
import json
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["POST", "OPTIONS"]}})

load_dotenv()

# Configure OpenAI and Pinecone
# openai.api_key = os.getenv("OPENAI_API_KEY")

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

@app.route("/api/train-therapist", methods=['POST'])
def train_therapist():
    try:
        profile_data = request.get_json()
        therapist_id = profile_data.get('therapist_id')
        text_to_embed = f"{profile_data.get('services', '')} {profile_data.get('summary', '')}"
        
        embeddings = generate_embedding(text_to_embed)
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

if __name__ == '__main__':
    app.run(debug=True)
