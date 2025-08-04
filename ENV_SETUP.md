# Setting Up Environment Variables

## Using a .env File

### 1. Create .env File
Create a file named `.env` in the project directory:

```bash
# Create .env file
touch .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-actual-key-here" >> .env
```

Or create it manually with content:
```
# .env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 2. Load Environment Variables

#### Option A: Using bash (Linux/Mac)
```bash
# Load variables from .env
source load_env.sh

# Or manually:
export $(cat .env | grep -v '^#' | xargs)
```

#### Option B: Using Python script
```bash
# This automatically loads .env
python run_with_env.py --auto_k
```

#### Option C: Using python-dotenv
```bash
# Install python-dotenv
pip install python-dotenv

# Then in Python:
from dotenv import load_dotenv
load_dotenv()
```

### 3. Direct Export (Without .env file)

```bash
# Temporary (current session only)
export OPENAI_API_KEY="sk-your-key-here"

# Run analysis
python run_analysis_improved.py --auto_k
```

### 4. Pass as Argument

```bash
# No need to export
python run_analysis_improved.py --auto_k --openai_key "sk-your-key-here"
```

## Security Best Practices

1. **Never commit .env to git**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use .env.example**
   - Create `.env.example` with dummy values
   - Commit this instead of actual .env

3. **Validate key is loaded**
   ```bash
   # Check if loaded
   echo $OPENAI_API_KEY
   
   # Or in Python
   python -c "import os; print('Key loaded:', bool(os.environ.get('OPENAI_API_KEY')))"
   ```

## Complete Example

```bash
# 1. Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
EOF

# 2. Load and run analysis
source load_env.sh
python run_analysis_improved.py --auto_k

# Or use the all-in-one script
python run_with_env.py --auto_k
```

## Troubleshooting

### Key not loading?
- Check .env file exists: `ls -la .env`
- Check format: `cat .env`
- No spaces around `=`
- No quotes needed (but OK if used)

### Still getting "NoneType" errors?
- The key might be invalid
- Check key starts with `sk-`
- Verify with: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`