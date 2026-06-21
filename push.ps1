Set-Location "C:\Users\codem\documents\rag-car-chatbot"
git add -A
git commit -m "initial commit"
git branch -M main
gh repo create rag-car-chatbot --public --source . --push
