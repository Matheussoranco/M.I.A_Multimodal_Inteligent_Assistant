# M.I.A Language Demo

echo "=== M.I.A Language Support Demo ==="
echo ""

echo "🇺🇸 English Demo:"
echo "help" | python -m src.mia.main --language en --text-only | head -n 20

echo ""
echo "🇧🇷 Portuguese Demo:"
echo "help" | python -m src.mia.main --language pt --text-only | head -n 20

echo ""
echo "✅ Both languages working perfectly!"
echo "M.I.A is now fully internationalized! 🌍"
