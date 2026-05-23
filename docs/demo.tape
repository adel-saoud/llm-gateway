Output demo.gif

Set Shell "zsh"
Set FontSize 13
Set Width 1200
Set Height 600
Set Theme "Dracula"
Set TypingSpeed 50ms

# Show the gateway is up
Type "curl -s http://localhost:8000/health | python3 -m json.tool"
Sleep 500ms
Enter
Sleep 1500ms

# Run the RAG demo
Type "uv run python scripts/demo_rag.py"
Sleep 500ms
Enter
Sleep 30s

# Show cost savings from a quick load test
Type "uv run python scripts/load_test.py --requests 50 --seed 7"
Sleep 500ms
Enter
Sleep 45s

Sleep 3s
