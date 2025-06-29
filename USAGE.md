# Usage Example

After installing dependencies, you can run M.I.A from the command line:

```bash
python -m main_modules.main --model-id mistral:instruct --api-key <YOUR_API_KEY>
```

Or, if installed as a package:

```bash
mia --model-id mistral:instruct --api-key <YOUR_API_KEY>
```

You can also pass additional arguments:

- `--image-input <path>`: Process an image file
- `--enable-reasoning`: Enable advanced reasoning

**Note:**
- Make sure to copy `.env.example` to `.env` and fill in your API keys.
- For more configuration, see the comments in `main_modules/main.py`.
