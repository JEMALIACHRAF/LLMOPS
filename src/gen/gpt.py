
import nest_asyncio
nest_asyncio.apply()
import asyncio
import aiohttp
import os




OPENAI_API_KEY = "***********************"
model_name = "gpt-4o-2024-05-13"  # Ensure this is the correct model name for chat


def get_model(model_name, generation_config=None, safety_settings=None):
    model = model_name
    return model



async def generate_async(model, prompt):
    url = f"https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                if result.get('choices') and result['choices'][0].get('message'):
                    return result['choices'][0]['message']['content'].strip()
                else:
                    return "Received an unexpected response format."
            else:
                print(f"Error: {response.status} - {await response.text()}")
                return None
