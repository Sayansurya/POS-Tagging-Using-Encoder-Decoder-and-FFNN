import requests

resp = requests.post(
    "http://127.0.0.1:5000/predict",
    params={
        'sentence': "my dog's name is brian haha"
    }
)

print(resp)
