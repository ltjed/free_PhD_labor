import os
import requests
from litellm import completion
import litellm
# litellm.set_verbose = True

os.environ['LITELLM_LOG'] = 'DEBUG'

def test_deepseek_api_direct():
    """Test DeepSeek API directly using requests"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Respond with 'API operational'"}],
        "temperature": 0.1
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        # print(response)
        response.raise_for_status()
        return f"Direct API OK (HTTP {response.status_code})"
    except requests.exceptions.RequestException as e:
        return f"Direct API Error: {str(e)}"

def test_deepseek_via_litellm():
    """Test DeepSeek API through LiteLLM"""
    try:
        response = completion(
            model="deepseek/deepseek-reasoner",
            messages=[{"role": "user", "content": "Respond with 'API operational'"}],
            temperature=0.1
        )
        print('reasoner response:')
        print(response)
        response = completion(
            model="deepseek/deepseek-coder",
            messages=[{"role": "user", "content": "Respond with 'API operational'"}],
            temperature=0.1
        )
        print('coder response:')
        print(response)
        return f"LiteLLM Proxy OK (Response: {response.choices[0].message.content})"
    except Exception as e:
        return f"LiteLLM Proxy Error: {str(e)}"

if __name__ == "__main__":
    print("Running DeepSeek API status check...\n")
    print("Direct API Test:", test_deepseek_api_direct())
    print("LiteLLM Proxy Test:", test_deepseek_via_litellm())