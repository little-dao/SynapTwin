import google.generativeai as genai
genai.configure(api_key="AIzaSyBPOI74hfCYbDDrkSFUH-tTiJivcnndmxs")

models = genai.list_models()
for model in models:
    print(model.name, "supports generation:", "generateContent" in model.supported_generation_methods)
