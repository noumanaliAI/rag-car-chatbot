@echo off
cd /d C:\Users\codem\documents\rag-car-chatbot
git add -A
git commit -m "initial commit - RAG car manual chatbot"
git branch -M main
gh repo create rag-car-chatbot --public --source . --push
