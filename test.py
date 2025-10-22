import google.generativeai as genai
genai.configure(api_key="AIzaSyCtjW1oim7SmMgDVMuTXKeGkJAKYFtAuE0")

for m in genai.list_models():
    print(m.name)
